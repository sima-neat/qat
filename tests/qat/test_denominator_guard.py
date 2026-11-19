"""Fake-quantized division denominators must not round to zero."""

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


class SumRatio(nn.Module):
    def forward(self, x):
        total = x.sum()
        return total / (total + 1.0)


def _softmax_input():
    torch.manual_seed(0)
    x = torch.randn(2, 4, 64)
    x[..., :16] = -12.0
    return x


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


def test_signed_denominator_is_not_floored() -> None:
    x = torch.ones(2, 8)
    y = torch.linspace(-1.0, 1.0, 16).view(2, 8)
    qat_model = sima_prepare_qat_model(Divide(), (x, y), "cpu")
    assert (qat_model(x, y) < 0).any()

    sima_freeze_qat(qat_model)
    final = sima_finalize_qat_model(qat_model)
    assert torch.ops.aten.clamp_min.default not in [node.target for node in final.graph.nodes]


def test_scalar_denominator_keeps_its_shape() -> None:
    x = torch.rand(2, 8)
    qat_model = sima_prepare_qat_model(SumRatio(), (x,), "cpu")
    qat_model(x)
    assert qat_model(x).shape == torch.Size([])


def test_disabled_fake_quant_divides_in_float() -> None:
    x = torch.ones(2, 8)
    y = torch.linspace(1e-4, 1.0, 16).view(2, 8)
    qat_model = sima_prepare_qat_model(Divide(), (x, y), "cpu")
    qat_model(x, y)

    qat_model.apply(disable_fake_quant)
    qat_model.apply(disable_observer)
    torch.testing.assert_close(qat_model(x, y), x / y)
