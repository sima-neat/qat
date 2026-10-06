"""BatchNorm folding, freezing, and finalization regressions."""

import copy
import warnings

import pytest
import torch
from torch.ao.quantization import disable_fake_quant, disable_observer
from torch.nn.utils.fusion import fuse_conv_bn_weights

from sima_qat import (
    sima_export_onnx, sima_finalize_qat_model, sima_freeze_qat, sima_prepare_qat_model,
)
from sima_qat.qat_api import _fake_quant_module, _resolve_attr

from tests.operators.cases import BatchNormModel, ConvBnModel


pytestmark = pytest.mark.regression


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable"
))]


class CustomBatchNormModel(torch.nn.Module):
    def __init__(self, dimension=2, bias=False, groups=1, standalone=False):
        super().__init__()
        conv = getattr(torch.nn, f"Conv{dimension}d")
        bn = getattr(torch.nn, f"BatchNorm{dimension}d")
        self.conv = torch.nn.Identity() if standalone else conv(
            2, 2, 1, bias=bias, groups=groups
        )
        self.bn = bn(2, eps=1e-3, momentum=0.03)
        with torch.no_grad():
            if not standalone:
                self.conv.weight.fill_(0.5)
                if bias:
                    self.conv.bias.copy_(torch.tensor([0.125, -0.25]))
            self.bn.running_mean.copy_(torch.tensor([0.125, -0.125]))
            self.bn.running_var.copy_(torch.tensor([1e-4, 2e-4]))
            self.bn.weight.copy_(torch.tensor([0.75, 1.25]))
            self.bn.bias.copy_(torch.tensor([0.25, -0.5]))

    def forward(self, inputs):
        return self.bn(self.conv(inputs))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("standalone", [False, True])
def test_custom_batchnorm_settings_survive_preparation_and_mode_changes(device, standalone):
    source = CustomBatchNormModel(standalone=standalone).to(device)
    initial_state = copy.deepcopy(source.state_dict())
    inputs = torch.linspace(-1, 1, 64, device=device).reshape(2, 2, 4, 4)
    model = sima_prepare_qat_model(source, (inputs,), device)
    for name, value in source.state_dict().items():
        torch.testing.assert_close(value, initial_state[name], rtol=0, atol=0)
    model.apply(disable_fake_quant).apply(disable_observer)
    for _ in range(2):
        for training in (False, True):
            model.train(training)
            source.train(training)
            bn = next(n for n in model.graph.nodes if n.target == torch.ops.aten.batch_norm.default)
            assert bn.args[6:8] == (0.03, 1e-3)
            with torch.no_grad():
                torch.testing.assert_close(model(inputs), source(inputs), rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(model.bn.running_mean, source.bn.running_mean)
            torch.testing.assert_close(model.bn.running_var, source.bn.running_var)
    assert not torch.equal(source.bn.running_mean, initial_state['bn.running_mean'])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dimension,bias,groups", [(1, False, 1), (2, False, 1), (2, True, 1), (2, True, 2)])
def test_custom_batchnorm_fold_preserves_weights_bias_and_qparams(device, dimension, bias, groups):
    source = CustomBatchNormModel(dimension, bias, groups).to(device)
    inputs = torch.linspace(-1, 1, 2 * 2 * 4 ** dimension, device=device).reshape(
        2, 2, *([4] * dimension)
    )
    model = sima_prepare_qat_model(source, (inputs,), device).eval()
    model(inputs)
    sima_freeze_qat(model)
    model.train().eval()
    with torch.no_grad():
        # Finalization must fold the current recovery weights, not a prepare-time copy.
        model.conv.weight.mul_(0.99)
        model.bn.weight.mul_(0.97)
    conv_op = getattr(torch.ops.aten, f"conv{dimension}d").default
    conv = next(n for n in model.graph.nodes if n.target == conv_op)
    weight_fq = _fake_quant_module(model, conv.args[1])
    expected_weight, expected_bias = fuse_conv_bn_weights(
        model.conv.weight, model.conv.bias if bias else None,
        model.bn.running_mean, model.bn.running_var, source.bn.eps,
        model.bn.weight, model.bn.bias,
    )
    expected_weight = weight_fq(expected_weight).detach()
    scale = weight_fq.scale.detach().clone()
    zero_point = weight_fq.zero_point.detach().clone()
    with torch.no_grad():
        expected_output = model(inputs)

    finalized = sima_finalize_qat_model(model)
    conv = next(n for n in finalized.graph.nodes if n.target == conv_op)
    dq = conv.args[1]
    args = [
        _resolve_attr(finalized, arg.target) if isinstance(arg, torch.fx.Node) else arg
        for arg in dq.args
    ]
    torch.testing.assert_close(dq.target(*args), expected_weight, rtol=0, atol=0)
    torch.testing.assert_close(args[1], scale, rtol=0, atol=0)
    torch.testing.assert_close(args[2], zero_point, rtol=0, atol=0)
    torch.testing.assert_close(_resolve_attr(finalized, conv.args[2].target), expected_bias, rtol=0, atol=0)
    assert all(n.target != torch.ops.aten.batch_norm.default for n in finalized.graph.nodes)
    torch.testing.assert_close(finalized(inputs), expected_output, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("standalone", [False, True])
def test_custom_batchnorm_onnx_parity(tmp_path, standalone):
    ort = pytest.importorskip("onnxruntime")
    inputs = torch.linspace(-1, 1, 64).reshape(2, 2, 4, 4)
    source = CustomBatchNormModel(standalone=standalone)
    model = sima_prepare_qat_model(source, (inputs,), "cpu").eval()
    model(inputs)
    sima_freeze_qat(model)
    expected = model(inputs).detach()
    finalized = sima_finalize_qat_model(model)
    if standalone:
        bn = next(n for n in finalized.graph.nodes if n.target == torch.ops.aten.batch_norm.default)
        assert bn.args[6:8] == (0.03, 1e-3)
    torch.testing.assert_close(finalized(inputs), expected)
    output = tmp_path / "custom_batchnorm.onnx"
    sima_export_onnx(finalized, (inputs,), str(output), device="cpu")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        str(output), sess_options=options, providers=["CPUExecutionProvider"]
    )
    actual = session.run(None, {session.get_inputs()[0].name: inputs.numpy()})[0]
    torch.testing.assert_close(torch.from_numpy(actual), expected, rtol=1e-5, atol=1e-5)


def test_distinct_batchnorm_settings_survive_checkpoint_and_finalization():
    source = torch.nn.Sequential(CustomBatchNormModel(), CustomBatchNormModel(bias=True))
    source[1].bn.eps = 0.02
    source[1].bn.momentum = 0.2
    inputs = torch.linspace(-1, 1, 64).reshape(2, 2, 4, 4)
    model = sima_prepare_qat_model(source, (inputs,), "cpu").eval()
    model(inputs)
    sima_freeze_qat(model)
    restored = sima_prepare_qat_model(source, (inputs,), "cpu")
    restored.load_state_dict(copy.deepcopy(model.state_dict()))
    restored.eval().train().eval()
    settings = [n.args[6:8] for n in restored.graph.nodes if n.target == torch.ops.aten.batch_norm.default]
    assert settings == [(0.03, 1e-3), (0.2, 0.02)]
    torch.testing.assert_close(restored(inputs), model(inputs), rtol=0, atol=0)
    expected = restored(inputs).detach()
    finalized = sima_finalize_qat_model(restored)
    torch.testing.assert_close(finalized(inputs), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_reused_convolution_preserves_distinct_batchnorm_folds(device):
    class ReusedConvModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(2, 2, 1, bias=False)
            self.bn1 = torch.nn.BatchNorm2d(2, eps=1e-3, momentum=0.03)
            self.bn2 = torch.nn.BatchNorm2d(2, eps=2e-2, momentum=0.2)
            with torch.no_grad():
                self.conv.weight.copy_(torch.tensor([
                    [[[0.5]], [[-0.25]]], [[[0.2]], [[0.7]]],
                ]))
                self.bn1.running_mean.copy_(torch.tensor([0.1, -0.2]))
                self.bn1.running_var.copy_(torch.tensor([0.01, 0.03]))
                self.bn2.running_mean.copy_(torch.tensor([-0.3, 0.4]))
                self.bn2.running_var.copy_(torch.tensor([0.02, 0.04]))

        def forward(self, inputs):
            return self.bn1(self.conv(inputs)) + self.bn2(self.conv(inputs * 0.75))

    inputs = torch.linspace(-1, 1, 64, device=device).reshape(2, 2, 4, 4)
    model = sima_prepare_qat_model(ReusedConvModel().to(device), (inputs,), device).eval()
    model(inputs)
    sima_freeze_qat(model)
    expected = model(inputs).detach()
    finalized = sima_finalize_qat_model(model)
    torch.testing.assert_close(finalized(inputs), expected, rtol=1e-5, atol=1e-5)


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


def test_conv_batchnorm_finalization_does_not_double_erase_nodes() -> None:
    inputs = torch.randn(2, 3, 8, 8)
    model = sima_prepare_qat_model(ConvBnModel(), (inputs,), "cpu")
    model(inputs)
    sima_freeze_qat(model)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        finalized = sima_finalize_qat_model(model)

    assert torch.isfinite(finalized(inputs)).all()
    assert not any("already erased node" in str(item.message) for item in caught)
