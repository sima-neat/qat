import pytest
import torch
from torch.ao.quantization.fake_quantize import FakeQuantizeBase

from sima_qat.qat_api import sima_prepare_qat_model


class ReusedConv(torch.nn.Module):
    """One Conv module invoked twice, as in an unrolled recurrent graph."""

    def __init__(self):
        super().__init__()
        self.shared = torch.nn.Conv2d(4, 4, 1, bias=False)

    def forward(self, value):
        return self.shared(torch.relu(self.shared(value)))


@pytest.mark.regression
def test_every_reused_conv_invocation_is_qat_annotated():
    prepared = sima_prepare_qat_model(
        ReusedConv().eval(), (torch.randn(1, 4, 8, 8),), torch.device("cpu")
    )
    convs = [
        node
        for node in prepared.graph.nodes
        if node.op == "call_function" and node.target == torch.ops.aten.conv2d.default
    ]
    assert len(convs) == 2
    modules = dict(prepared.named_modules(remove_duplicate=False))
    for conv in convs:
        for argument in conv.args[:2]:
            assert argument.op == "call_module"
            assert isinstance(modules[str(argument.target)], FakeQuantizeBase)
