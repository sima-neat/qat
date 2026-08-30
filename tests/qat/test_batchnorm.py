"""BatchNorm folding, freezing, and finalization regressions."""

import pytest
import torch

from sima_qat import sima_finalize_qat_model, sima_freeze_qat, sima_prepare_qat_model
from sima_qat.qat_api import _fake_quant_module

from tests.operators.cases import BatchNormModel, ConvBnModel


pytestmark = pytest.mark.regression


class Conv1dBnModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(3, 5, 3, padding=1)
        self.bn = torch.nn.BatchNorm1d(5)

    def forward(self, inputs):
        return self.bn(self.conv(inputs))


class GroupedConvBnModel(torch.nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, 3, padding=1, groups=groups)
        self.bn = torch.nn.BatchNorm2d(4)

    def forward(self, inputs):
        return self.bn(self.conv(inputs))


def test_folded_weight_lock_uses_current_parameters() -> None:
    inputs = torch.randn(2, 3, 8, 8)
    model = sima_prepare_qat_model(ConvBnModel(), (inputs,), "cpu")
    model(inputs)
    conv = next(
        node
        for node in model.graph.nodes
        if node.op == "call_function" and node.target == torch.ops.aten.conv2d.default
    )
    weight_fq = _fake_quant_module(model, conv.args[1])
    assert weight_fq is not None
    stale_capacity = weight_fq.scale.detach().clone() * 127.0
    with torch.no_grad():
        model.conv.weight.mul_(4.0)
    bn_scale = model.bn.weight.detach() / torch.sqrt(model.bn.running_var.detach() + 1e-5)
    folded_weight = model.conv.weight.detach() * bn_scale.reshape(-1, 1, 1, 1)
    current_max = folded_weight.abs().amax(dim=(1, 2, 3))
    assert bool((current_max > stale_capacity).any())

    sima_freeze_qat(model)

    assert bool((weight_fq.scale * 127.0 >= current_max).all())


def test_running_statistics_stay_frozen_during_recovery() -> None:
    inputs = torch.randn(2, 3, 8, 8)
    model = sima_prepare_qat_model(ConvBnModel(), (inputs,), "cpu")
    model(inputs)
    sima_freeze_qat(model)
    running_mean = model.bn.running_mean.detach().clone()
    running_var = model.bn.running_var.detach().clone()
    batches = model.bn.num_batches_tracked.detach().clone()

    model.eval()
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    model(inputs).square().mean().backward()
    optimizer.step()

    torch.testing.assert_close(model.bn.running_mean, running_mean, rtol=0, atol=0)
    torch.testing.assert_close(model.bn.running_var, running_var, rtol=0, atol=0)
    torch.testing.assert_close(model.bn.num_batches_tracked, batches, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("source", "inputs", "operator"),
    [
        (Conv1dBnModel(), (torch.randn(2, 3, 12),), torch.ops.aten.conv1d.default),
        (
            GroupedConvBnModel(groups=2),
            (torch.randn(2, 4, 8, 8),),
            torch.ops.aten.conv2d.default,
        ),
        (
            GroupedConvBnModel(groups=4),
            (torch.randn(2, 4, 8, 8),),
            torch.ops.aten.conv2d.default,
        ),
    ],
    ids=["conv1d", "grouped_conv2d", "depthwise_conv2d"],
)
def test_folded_weight_variants_freeze_without_clipping(source, inputs, operator) -> None:
    model = sima_prepare_qat_model(source, inputs, "cpu")
    model(*inputs)
    conv = next(
        node
        for node in model.graph.nodes
        if node.op == "call_function" and node.target == operator
    )
    weight_fq = _fake_quant_module(model, conv.args[1])
    assert weight_fq is not None

    sima_freeze_qat(model)

    bn_scale = model.bn.weight.detach() / torch.sqrt(model.bn.running_var.detach() + 1e-5)
    scale_shape = (-1,) + (1,) * (model.conv.weight.ndim - 1)
    folded_weight = model.conv.weight.detach() * bn_scale.reshape(scale_shape)
    reduce_dims = tuple(range(1, folded_weight.ndim))
    current_max = folded_weight.abs().amax(dim=reduce_dims)
    assert bool((weight_fq.scale * 127.0 >= current_max).all())


def test_standalone_batchnorm_is_replaced_during_finalization() -> None:
    inputs = torch.randn(2, 4, 8, 8)
    model = sima_prepare_qat_model(BatchNormModel(), (inputs,), "cpu")
    model(inputs)
    sima_freeze_qat(model)

    finalized = sima_finalize_qat_model(model)

    assert torch.isfinite(finalized(inputs)).all()
    assert all(
        node.target != torch.ops.aten._native_batch_norm_legit_no_training.default
        for node in finalized.graph.nodes
    )
