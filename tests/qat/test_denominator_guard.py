"""Positive division guards must survive INT8 fake quantization."""

import onnx
import onnxruntime
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from sima_qat import (
    sima_export_onnx,
    sima_finalize_qat_model,
    sima_freeze_qat,
    sima_prepare_qat_model,
)


pytestmark = pytest.mark.regression


class Normalize(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.linspace(0.8, 1.2, 4))

    def forward(self, x):
        return F.normalize(x * self.gain, dim=-1, eps=1e-6)


class SumNormalize(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.linspace(0.9, 1.1, 4).view(1, 4, 1))

    def forward(self, x):
        attention = torch.softmax(x * self.gain, dim=-1)
        return attention / (attention.sum(dim=1, keepdim=True) + self.eps)


class SignedDenominator(nn.Module):
    def forward(self, x, y):
        return x / (y + 1e-6)


class UnguardedDivision(nn.Module):
    def forward(self, x, y):
        return x / y


def _normalize_input():
    return torch.tensor([[1e-7, 1e-7, 0.0, 0.0], [0.25, -0.5, 0.75, -1.0]])


def _softmax_input():
    x = torch.randn(2, 4, 64)
    x[..., :16] = -12.0
    return x


def _guards(model):
    return [node for node in model.graph.nodes if "sima_denominator_guard" in node.meta]


@pytest.mark.parametrize(
    "model, example",
    [(Normalize(), _normalize_input), (SumNormalize(), _softmax_input)],
    ids=["clamp_min", "additive"],
)
def test_guarded_division_is_finite_through_training(model, example) -> None:
    torch.manual_seed(0)
    x = example()
    qat_model = sima_prepare_qat_model(model, (x,), "cpu")
    assert len(_guards(qat_model)) == 1

    optimizer = torch.optim.SGD(qat_model.parameters(), lr=1e-2)
    for _ in range(3):
        output = qat_model(x)
        assert torch.isfinite(output).all()
        loss = output.square().mean()
        optimizer.zero_grad()
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in qat_model.parameters() if p.grad is not None)
        optimizer.step()

    sima_freeze_qat(qat_model)
    assert torch.isfinite(qat_model(x)).all()


def test_floor_is_one_step_of_the_denominator_grid() -> None:
    x = _normalize_input()
    qat_model = sima_prepare_qat_model(Normalize(), (x,), "cpu")
    qat_model(x)
    guard = _guards(qat_model)[0]
    quantizer = qat_model.get_submodule(guard.meta["sima_denominator_guard"])
    assert int(quantizer.zero_point) == quantizer.activation_post_process.quant_min

    captured = {}
    handle = quantizer.register_forward_hook(lambda module, args, output: captured.update(value=output))
    qat_model(x)
    handle.remove()
    assert float(captured["value"].min()) == 0.0
    # The row of tiny values divides 0 by exactly one step, giving 0.
    assert torch.equal(qat_model(x)[0], torch.zeros(4))


def test_frozen_floor_exports_as_scalar_clip(tmp_path) -> None:
    x = _normalize_input()
    qat_model = sima_prepare_qat_model(Normalize(), (x,), "cpu")
    qat_model(x)
    sima_freeze_qat(qat_model)
    guard = _guards(qat_model)[0]
    step = float(qat_model.get_submodule(guard.meta["sima_denominator_guard"]).scale)

    final = sima_finalize_qat_model(qat_model)
    expected = final(x)
    path = tmp_path / "normalize.onnx"
    sima_export_onnx(final, (x,), str(path), input_names=["x"], output_names=["y"])

    graph = onnx.load(str(path)).graph
    assert "Max" not in [node.op_type for node in graph.node]
    constants = {item.name: onnx.numpy_helper.to_array(item) for item in graph.initializer}
    for node in graph.node:
        if node.op_type == "Constant":
            constants[node.output[0]] = onnx.numpy_helper.to_array(node.attribute[0].t)
    # F.normalize exports its own Clip at 1e-6; the guard adds one at the frozen step.
    floors = sorted(
        float(constants[name])
        for node in graph.node
        if node.op_type == "Clip"
        for name in node.input[1:2]
        if name in constants
    )
    assert floors == [pytest.approx(1e-6), pytest.approx(step)]

    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"x": x.numpy()})[0]
    assert torch.allclose(torch.from_numpy(actual), expected, atol=1e-6)


def test_checkpoint_reload_keeps_guarded_outputs() -> None:
    x = _softmax_input()
    qat_model = sima_prepare_qat_model(SumNormalize(), (x,), "cpu")
    qat_model(x)
    sima_freeze_qat(qat_model)
    restored = sima_prepare_qat_model(SumNormalize(), (x,), "cpu")
    restored.load_state_dict(qat_model.state_dict())
    assert torch.equal(restored(x), qat_model(x))


def test_signed_denominator_is_not_floored() -> None:
    x = torch.ones(2, 8)
    y = torch.linspace(-1.0, 1.0, 16).view(2, 8)
    qat_model = sima_prepare_qat_model(SignedDenominator(), (x, y), "cpu")
    assert len(_guards(qat_model)) == 1
    output = qat_model(x, y)
    assert (output < 0).any()

    sima_freeze_qat(qat_model)
    final = sima_finalize_qat_model(qat_model)
    assert not any(node.target == torch.ops.aten.clamp_min.default for node in final.graph.nodes)


def test_unguarded_division_is_left_unchanged() -> None:
    x = torch.ones(2, 8)
    y = torch.rand(2, 8) + 0.5
    qat_model = sima_prepare_qat_model(UnguardedDivision(), (x, y), "cpu")
    assert _guards(qat_model) == []
