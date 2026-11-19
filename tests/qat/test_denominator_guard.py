"""Fake-quantized division denominators that the model guards must not round to zero."""

import onnx
import onnxruntime
import pytest
import torch
from torch import nn
from torch.ao.quantization import disable_fake_quant, disable_observer

from sima_qat import (
    sima_export_onnx,
    sima_finalize_qat_model,
    sima_freeze_qat,
    sima_prepare_qat_model,
)


pytestmark = pytest.mark.regression


class SumNormalize(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.linspace(0.9, 1.1, 4).view(1, 4, 1))

    def forward(self, x):
        attention = torch.softmax(x * self.gain, dim=-1)
        return attention / (attention.sum(dim=1, keepdim=True) + 1e-6)


class Divide(nn.Module):
    def forward(self, x, y):
        return x / y


class ClampDivide(nn.Module):
    def forward(self, x, y):
        return x / y.clamp_min(1e-6)


class SelfRatio(nn.Module):
    def __init__(self, clone: bool):
        super().__init__()
        self.clone = clone

    def forward(self, x):
        d = x.clamp_min(1e-6)
        return d / (d.clone() if self.clone else d)


class SumRatio(nn.Module):
    def forward(self, x):
        total = torch.relu(x).sum()
        return total / (total + 1.0)


def _softmax_input():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 64)
    x[..., :16] = -12.0
    return x


def _floors(model) -> int:
    return sum(1 for node in model.graph.nodes if "sima_denominator_floor" in node.meta)


def test_small_denominator_stays_finite_through_training() -> None:
    x = _softmax_input()
    qat_model = sima_prepare_qat_model(SumNormalize(), (x,), "cpu")
    optimizer = torch.optim.SGD(qat_model.parameters(), lr=1e-2)
    for _ in range(3):
        output = qat_model(x)
        assert torch.isfinite(output).all()
        optimizer.zero_grad()
        output.square().mean().backward()
        optimizer.step()

    sima_freeze_qat(qat_model)
    assert torch.isfinite(qat_model(x)).all()


def test_frozen_floor_exports_as_clip_at_one_step(tmp_path) -> None:
    x = _softmax_input()
    qat_model = sima_prepare_qat_model(SumNormalize(), (x,), "cpu")
    qat_model(x)
    sima_freeze_qat(qat_model)
    (quantizer_name,) = [
        node.meta["sima_denominator_floor"]
        for node in qat_model.graph.nodes
        if "sima_denominator_floor" in node.meta
    ]
    step = float(qat_model.get_submodule(quantizer_name).scale)
    final = sima_finalize_qat_model(qat_model)
    expected = final(x)
    assert torch.isfinite(expected).all()

    path = tmp_path / "sum_normalize.onnx"
    sima_export_onnx(final, (x,), str(path), input_names=["x"], output_names=["y"])
    graph = onnx.load(str(path)).graph
    constants = {item.name: onnx.numpy_helper.to_array(item) for item in graph.initializer}
    for node in graph.node:
        if node.op_type == "Constant":
            constants[node.output[0]] = onnx.numpy_helper.to_array(node.attribute[0].t)
    clips = [node for node in graph.node if node.op_type == "Clip"]
    assert len(clips) == 1
    assert float(constants[clips[0].input[1]]) == pytest.approx(step)

    session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"x": x.numpy()})[0]
    assert torch.allclose(torch.from_numpy(actual), expected, atol=1e-6)


@pytest.mark.parametrize(
    "denominator",
    [[0.0, 0.1, 0.5, 1.0], [-1e-4, 0.1, 0.5, 1.0]],
    ids=["raw_zero", "small_negative"],
)
def test_unguarded_denominator_is_not_floored(denominator) -> None:
    x = torch.ones(1, 4)
    y = torch.tensor([denominator])
    qat_model = sima_prepare_qat_model(Divide(), (x, y), "cpu")
    assert _floors(qat_model) == 0
    assert not torch.isfinite(qat_model(x, y)[0, 0])


def test_frozen_checkpoint_loads_into_unguarded_model() -> None:
    x = torch.ones(2, 8)
    y = torch.linspace(0.1, 1.0, 16).view(2, 8)
    trained = sima_prepare_qat_model(Divide(), (x, y), "cpu")
    trained(x, y)
    sima_freeze_qat(trained)
    expected = sima_finalize_qat_model(trained)(x, y)

    loaded = sima_prepare_qat_model(Divide(), (x, y), "cpu")
    loaded.load_state_dict(trained.state_dict())
    sima_freeze_qat(loaded)
    final = sima_finalize_qat_model(loaded)
    assert torch.ops.aten.clamp_min.default not in [node.target for node in final.graph.nodes]
    torch.testing.assert_close(final(x, y), expected)


def test_self_division_floors_only_the_denominator() -> None:
    x = torch.tensor([[0.0, 0.1, 0.5, 1.0]])
    shared = sima_prepare_qat_model(SelfRatio(clone=False), (x,), "cpu")
    cloned = sima_prepare_qat_model(SelfRatio(clone=True), (x,), "cpu")
    assert _floors(shared) == _floors(cloned) == 1
    output = shared(x)
    assert output[0, 0] == 0
    torch.testing.assert_close(output, cloned(x))


def test_scalar_denominator_keeps_its_shape() -> None:
    x = torch.rand(2, 8)
    qat_model = sima_prepare_qat_model(SumRatio(), (x,), "cpu")
    assert _floors(qat_model) == 1
    qat_model(x)
    assert qat_model(x).shape == torch.Size([])


def test_disabled_fake_quant_divides_in_float() -> None:
    x = torch.ones(2, 8)
    y = torch.linspace(1e-4, 1.0, 16).view(2, 8)
    qat_model = sima_prepare_qat_model(ClampDivide(), (x, y), "cpu")
    assert _floors(qat_model) == 1
    qat_model(x, y)

    qat_model.apply(disable_fake_quant)
    qat_model.apply(disable_observer)
    torch.testing.assert_close(qat_model(x, y), x / y)
