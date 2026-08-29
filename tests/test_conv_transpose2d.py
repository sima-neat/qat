"""Strict W8A8 scaffolding coverage for ConvTranspose2d."""

import pytest
import torch
from torch.ao.quantization.fake_quantize import FakeQuantizeBase

from sima_qat.qat_api import sima_prepare_qat_model


class TransposedConvolution(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample = torch.nn.ConvTranspose2d(3, 4, 3, stride=2)

    def forward(self, value):
        return self.upsample(value)


@pytest.mark.regression
def test_conv_transpose2d_has_int8_activation_and_weight_fake_quantizers():
    prepared = sima_prepare_qat_model(
        TransposedConvolution(),
        (torch.randn(1, 3, 4, 4),),
        "cpu",
    )
    node = next(
        node
        for node in prepared.graph.nodes
        if node.op == "call_function"
        and node.target == torch.ops.aten.conv_transpose2d.input
    )
    input_fake_quant = prepared.get_submodule(str(node.args[0].target))
    weight_fake_quant = prepared.get_submodule(str(node.args[1].target))

    assert isinstance(input_fake_quant, FakeQuantizeBase)
    assert input_fake_quant.dtype == torch.int8
    assert isinstance(weight_fake_quant, FakeQuantizeBase)
    assert weight_fake_quant.dtype == torch.int8
    assert weight_fake_quant.qscheme == torch.per_tensor_symmetric
    assert (weight_fake_quant.quant_min, weight_fake_quant.quant_max) == (-127, 127)
