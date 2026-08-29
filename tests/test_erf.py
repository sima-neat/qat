"""Coverage for the Erf boundary used by target-realizable GELU."""

import torch
import pytest

from sima_qat.qat_api import sima_prepare_qat_model


class ExplicitGelu(torch.nn.Module):
    def forward(self, x):
        scaled = x * 0.7071067811865476
        cdf = torch.erf(scaled)
        cdf = cdf + 1.0
        return (x * cdf) * 0.5


class ConvExplicitGelu(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 1)

    def forward(self, x):
        return ExplicitGelu()(self.conv(x))


@pytest.mark.regression
def test_erf_has_input_and_output_fake_quant_boundaries():
    model = sima_prepare_qat_model(
        ExplicitGelu(), (torch.randn(1, 3, 4, 4),), torch.device("cpu")
    )
    erf = next(
        node for node in model.graph.nodes
        if node.op == "call_function" and node.target == torch.ops.aten.erf.default
    )
    assert erf.args[0].op == "call_module"
    assert "FakeQuant" in type(model.get_submodule(str(erf.args[0].target))).__name__
    assert erf.next.op == "call_module"
    assert "FakeQuant" in type(model.get_submodule(str(erf.next.target))).__name__


@pytest.mark.regression
def test_gelu_scale_multiply_is_not_fused_into_preceding_conv():
    model = sima_prepare_qat_model(
        ConvExplicitGelu(), (torch.randn(1, 3, 4, 4),), torch.device("cpu")
    )
    erf = next(
        node for node in model.graph.nodes
        if node.op == "call_function" and node.target == torch.ops.aten.erf.default
    )
    scale_input_fq = erf.args[0]
    scale_mul = scale_input_fq.args[0]
    conv_output_fq = scale_mul.args[0]
    assert scale_mul.target == torch.ops.aten.mul.Tensor
    assert conv_output_fq.op == "call_module"
    assert "FakeQuant" in type(model.get_submodule(str(conv_output_fq.target))).__name__
    assert conv_output_fq.args[0].target == torch.ops.aten.conv2d.default
