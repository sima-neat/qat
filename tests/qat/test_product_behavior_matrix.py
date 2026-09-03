"""Production-facing behavioral matrix for capture, freeze, and export."""

from __future__ import annotations

from collections import Counter
import operator

import numpy as np
import onnx
import pytest
import torch
from onnx import numpy_helper
from torch import nn
from torch.ao.quantization.fake_quantize import FakeQuantizeBase

from sima_qat import (
    sima_export_onnx,
    sima_finalize_qat_model,
    sima_freeze_qat,
    sima_prepare_qat_model,
)
from sima_qat.qat_api import (
    _capture_inputs_to_cpu,
    _fake_quant_module,
    _find_output_fake_quant,
    _resolve_static_weight_tensor,
)
from tests.operators.cases import Conv2dModel


pytestmark = pytest.mark.regression


class BatchTokenLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token = nn.Parameter(torch.randn(1, 1, 4))
        self.linear = nn.Linear(4, 4)

    def forward(self, inputs):
        token = self.token.expand(inputs.shape[0], -1, -1)
        return self.linear(torch.cat((token, inputs), dim=1))


class FoldedDirection(nn.Module):
    def forward(self, inputs):
        torch._assert(inputs.shape[0] == 1, "batch is folded into direction")
        return inputs[:, None].expand(-1, 2, -1).reshape(1, 8)


class BatchDependentBranch(nn.Module):
    """A shape-valid capture whose batch-two branch is wrong for batch one."""

    def forward(self, inputs):
        if inputs.shape[0] > 1:
            return inputs + 10
        return inputs - 10


class DynamicDropout(nn.Module):
    def forward(self, inputs):
        return torch.nn.functional.dropout(
            inputs,
            p=0.5,
            training=self.training,
        )


class NestedDynamicOutput(nn.Module):
    def forward(self, inputs):
        return {
            "prediction": inputs * 2,
            "auxiliary": (
                inputs.mean(dim=-1),
                inputs.argmax(dim=-1),
            ),
        }


class MultiInputBatch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, values, mask, static_table):
        return self.linear(values + mask) + static_table[0]


class FailingDynamicBatchNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(3)

    def forward(self, inputs):
        torch._assert(inputs.shape[0] == 1, "static folded batch")
        return self.bn(inputs)


class DynamicBatchNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(3)

    def forward(self, inputs):
        return self.bn(inputs)


class ConvChain(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Conv2d(3, 4, 1, bias=False)
        self.second = nn.Conv2d(4, 4, 1, bias=False)

    def forward(self, inputs):
        return self.second(self.first(inputs))


class ConvFanout(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.source = nn.Conv2d(3, 4, 1, bias=False)
        self.left = nn.Conv2d(4, 4, 1, bias=False)
        self.right = nn.Conv2d(4, 4, 1, bias=False)

    def forward(self, inputs):
        shared = self.source(inputs)
        return self.left(shared) + self.right(shared)


class ValidThenDynamicConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.bn = nn.BatchNorm2d(4)

    def forward(self, inputs, runtime_weight):
        value = self.bn(self.conv(inputs))
        return torch.nn.functional.conv2d(value, runtime_weight, padding=1)


class ViewSlicedWeightConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(8, 3, 3, 3))

    def forward(self, inputs):
        weight = self.weight.view(8, 3, 3, 3)[:4].clone()
        return torch.nn.functional.conv2d(inputs, weight, padding=1)


class UnsupportedWeightCallable(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, 3, 3, 3))

    def forward(self, inputs):
        return torch.nn.functional.conv2d(inputs, torch.sin(self.weight), padding=1)


class NonPersistableObserver(nn.Module):
    quant_min = -128
    quant_max = 127

    def __init__(self, scale: float, zero_point: int) -> None:
        super().__init__()
        self.register_buffer("fixed_scale", torch.tensor([scale]))
        self.register_buffer("fixed_zero_point", torch.tensor([zero_point], dtype=torch.int32))

    def calculate_qparams(self):
        return self.fixed_scale, self.fixed_zero_point

    def forward(self, value):
        return value


def _conv_contracts(model):
    contracts = []
    for node in model.graph.nodes:
        if node.op != "call_function" or node.target != torch.ops.aten.conv2d.default:
            continue
        contracts.append(
            (
                node,
                _fake_quant_module(model, node.args[0]),
                _fake_quant_module(model, node.args[1]),
                _find_output_fake_quant(model, node),
            )
        )
    assert contracts
    assert all(all(item is not None for item in row[1:]) for row in contracts)
    return contracts


def _set_grid(fake_quant, scale: float, zero_point: int = 0) -> None:
    observer = fake_quant.activation_post_process
    lower = (observer.quant_min - zero_point) * scale
    upper = (observer.quant_max - zero_point) * scale
    observer.min_val.fill_(lower)
    observer.max_val.fill_(upper)


def _snapshot_freeze_state(model):
    fake_quantizers = {}
    for name, module in model.named_modules():
        if not isinstance(module, FakeQuantizeBase):
            continue
        observer = module.activation_post_process
        fake_quantizers[name] = {
            "scale": module.scale.detach().clone(),
            "zero_point": module.zero_point.detach().clone(),
            "observer_enabled": module.observer_enabled.detach().clone(),
            "minimum": (
                observer.min_val.detach().clone()
                if hasattr(observer, "min_val")
                else None
            ),
            "maximum": (
                observer.max_val.detach().clone()
                if hasattr(observer, "max_val")
                else None
            ),
        }
    batchnorm = {
        name: value.detach().clone()
        for name, value in model.named_buffers()
        if any(token in name for token in ("running_mean", "running_var", "num_batches_tracked"))
    }
    return {
        "fake_quantizers": fake_quantizers,
        "batchnorm": batchnorm,
        "qat_frozen": model.qat_frozen.detach().clone(),
    }


def _assert_freeze_state_equal(model, snapshot) -> None:
    current_modules = dict(model.named_modules())
    for name, expected in snapshot["fake_quantizers"].items():
        module = current_modules[name]
        observer = module.activation_post_process
        torch.testing.assert_close(module.scale, expected["scale"], rtol=0, atol=0)
        torch.testing.assert_close(module.zero_point, expected["zero_point"], rtol=0, atol=0)
        torch.testing.assert_close(
            module.observer_enabled, expected["observer_enabled"], rtol=0, atol=0
        )
        if expected["minimum"] is not None:
            torch.testing.assert_close(
                observer.min_val,
                expected["minimum"],
                rtol=0,
                atol=0,
                equal_nan=True,
            )
        if expected["maximum"] is not None:
            torch.testing.assert_close(
                observer.max_val,
                expected["maximum"],
                rtol=0,
                atol=0,
                equal_nan=True,
            )
    current_buffers = dict(model.named_buffers())
    for name, expected in snapshot["batchnorm"].items():
        torch.testing.assert_close(current_buffers[name], expected, rtol=0, atol=0)
    torch.testing.assert_close(model.qat_frozen, snapshot["qat_frozen"], rtol=0, atol=0)


def _onnx_value(model, name):
    initializers = {value.name: numpy_helper.to_array(value) for value in model.graph.initializer}
    if name in initializers:
        return initializers[name]
    producers = {output: node for node in model.graph.node for output in node.output}
    node = producers[name]
    if node.op_type == "Identity":
        return _onnx_value(model, node.input[0])
    if node.op_type == "Constant":
        return numpy_helper.to_array(next(attr.t for attr in node.attribute if attr.name == "value"))
    raise RuntimeError(f"cannot resolve {name} from {node.op_type}")


def _qparam_key(scale, zero_point):
    scale = np.ascontiguousarray(np.asarray(scale, dtype=np.float32))
    # Fake quant stores zero points as int32 while ONNX Q/DQ serializes the
    # same logical code in int8. Compare the logical value, not carrier width.
    zero_point = np.ascontiguousarray(np.asarray(zero_point, dtype=np.int64))
    return (scale.shape, scale.tobytes(), zero_point.shape, zero_point.tobytes())


def _snapshot_source(model):
    return {
        "state": {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        },
        "modes": {name: module.training for name, module in model.named_modules()},
        "devices": {
            name: value.device for name, value in model.state_dict().items()
        },
        "parameter_ids": {name: id(value) for name, value in model.named_parameters()},
        "buffer_ids": {name: id(value) for name, value in model.named_buffers()},
        "requires_grad": {
            name: value.requires_grad for name, value in model.named_parameters()
        },
        "gradients": {
            name: None if value.grad is None else value.grad.detach().clone()
            for name, value in model.named_parameters()
        },
    }


def _assert_source_equal(model, snapshot) -> None:
    assert {name: module.training for name, module in model.named_modules()} == snapshot["modes"]
    state = model.state_dict()
    assert state.keys() == snapshot["state"].keys()
    for name, expected in snapshot["state"].items():
        torch.testing.assert_close(state[name], expected, rtol=0, atol=0)
        assert state[name].device == snapshot["devices"][name]
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    assert {name: id(value) for name, value in parameters.items()} == snapshot["parameter_ids"]
    assert {name: id(value) for name, value in buffers.items()} == snapshot["buffer_ids"]
    assert {
        name: value.requires_grad for name, value in parameters.items()
    } == snapshot["requires_grad"]
    for name, expected in snapshot["gradients"].items():
        if expected is None:
            assert parameters[name].grad is None
        else:
            torch.testing.assert_close(parameters[name].grad, expected, rtol=0, atol=0)


def test_static_default_preserves_ordinary_and_folded_geometry() -> None:
    ordinary = torch.randn(1, 2, 4)
    prepared = sima_prepare_qat_model(BatchTokenLinear(), (ordinary,), "cpu")
    assert prepared(ordinary).shape == (1, 3, 4)
    with pytest.raises((RuntimeError, AssertionError)):
        prepared(torch.randn(2, 2, 4))

    folded = torch.randn(1, 4)
    prepared_folded = sima_prepare_qat_model(FoldedDirection(), (folded,), "cpu")
    assert prepared_folded(folded).shape == (1, 8)


def test_explicit_dynamic_batch_handles_multi_input_shared_and_static_leading_dims() -> None:
    values = torch.randn(1, 2, 4)
    mask = torch.randn(1, 2, 4)
    static_table = torch.randn(3, 4)
    source = MultiInputBatch()
    source_state = {name: value.detach().clone() for name, value in source.state_dict().items()}
    prepared = sima_prepare_qat_model(
        source,
        (values, mask, static_table),
        "cpu",
        dynamic_batch=True,
    )
    for batch in (1, 2, 4):
        output = prepared(
            torch.randn(batch, 2, 4),
            torch.randn(batch, 2, 4),
            static_table,
        )
        assert output.shape == (batch, 2, 4)
    for name, expected in source_state.items():
        torch.testing.assert_close(source.state_dict()[name], expected, rtol=0, atol=0)


def test_prepare_normalizes_nested_capture_inputs_without_alias_or_mutation() -> None:
    source = torch.randn(1, 3, requires_grad=True)
    nested = (source, {"same": source, "mask": source[:, :1], "label": ["sample"]})
    captured = _capture_inputs_to_cpu(nested)

    assert captured[0].device.type == "cpu"
    assert captured[1]["mask"].device.type == "cpu"
    assert not captured[0].requires_grad
    assert captured[0].data_ptr() != source.data_ptr()
    assert captured[0] is captured[1]["same"]
    assert captured[1]["mask"].data_ptr() != nested[1]["mask"].data_ptr()
    torch.testing.assert_close(captured[0], source.detach(), rtol=0, atol=0)
    captured[0].zero_()
    assert bool((source != 0).any())
    assert nested[1]["label"] == ["sample"]


def test_successful_dynamic_prepare_leaves_source_mode_state_and_device_unchanged() -> None:
    source = DynamicBatchNorm().eval()
    with torch.no_grad():
        source.bn.weight.fill_(1.25)
        source.bn.bias.fill_(-0.5)
        source.bn.running_mean.copy_(torch.tensor([0.1, -0.2, 0.3]))
        source.bn.running_var.copy_(torch.tensor([0.8, 1.1, 1.4]))
    source.bn.weight.grad = torch.full_like(source.bn.weight, 0.25)
    example = torch.randn(1, 3, 4, 4)
    snapshot = _snapshot_source(source)

    prepared = sima_prepare_qat_model(
        source, (example,), "cpu", dynamic_batch=True
    )

    _assert_source_equal(source, snapshot)
    for batch in (1, 2, 4):
        assert prepared(torch.randn(batch, 3, 4, 4)).shape == (batch, 3, 4, 4)


def test_dynamic_batch_failure_leaves_source_and_batchnorm_unchanged() -> None:
    source = FailingDynamicBatchNorm().train()
    example = torch.randn(1, 3, 4, 4)
    snapshot = _snapshot_source(source)

    with pytest.raises(RuntimeError, match="Dynamic-batch QAT capture"):
        sima_prepare_qat_model(source, (example,), "cpu", dynamic_batch=True)

    _assert_source_equal(source, snapshot)


def test_dynamic_batch_rejects_semantically_different_batch_branch() -> None:
    example = torch.ones(1, 4)
    source = BatchDependentBranch()
    expected = source(example)

    with pytest.raises(RuntimeError, match="does not preserve.*semantics"):
        sima_prepare_qat_model(
            source,
            (example,),
            "cpu",
            dynamic_batch=True,
        )

    torch.testing.assert_close(source(example), expected, rtol=0, atol=0)


def test_dynamic_batch_parity_reuses_rng_for_stochastic_outputs() -> None:
    prepared = sima_prepare_qat_model(
        DynamicDropout().train(),
        (torch.randn(1, 8),),
        "cpu",
        dynamic_batch=True,
    )

    assert prepared(torch.randn(3, 8)).shape == (3, 8)


def test_dynamic_batch_parity_supports_nested_output_pytrees() -> None:
    prepared = sima_prepare_qat_model(
        NestedDynamicOutput(),
        (torch.randn(1, 4),),
        "cpu",
        dynamic_batch=True,
    )

    output = prepared(torch.randn(3, 4))
    assert output["prediction"].shape == (3, 4)
    assert output["auxiliary"][0].shape == (3,)
    assert output["auxiliary"][1].shape == (3,)


def test_solver_propagates_coarsened_grid_through_multi_op_chain() -> None:
    inputs = torch.randn(1, 3, 4, 4)
    model = sima_prepare_qat_model(ConvChain(), (inputs,), "cpu")
    model(inputs)
    first, second = _conv_contracts(model)
    assert first[3] is second[1]
    _set_grid(first[1], 1.0)
    _set_grid(first[3], 0.01)
    _set_grid(second[3], 1.0)
    with torch.no_grad():
        model.first.weight.fill_(127.0)
        model.second.weight.fill_(127.0)

    sima_freeze_qat(model)

    assert len(model.meta["qat_activation_retargets"]) >= 2
    for _, input_fq, weight_fq, output_fq in (first, second):
        assert bool((input_fq.scale * weight_fq.scale / output_fq.scale <= 1.0).all())


def test_solver_handles_shared_fanout_grid_for_every_consumer() -> None:
    inputs = torch.randn(1, 3, 4, 4)
    model = sima_prepare_qat_model(ConvFanout(), (inputs,), "cpu")
    model(inputs)
    source, left, right = _conv_contracts(model)
    assert source[3] is left[1] is right[1]
    _set_grid(source[1], 1.0)
    _set_grid(source[3], 0.01)
    _set_grid(left[3], 1.0)
    _set_grid(right[3], 1.0)
    with torch.no_grad():
        model.source.weight.fill_(127.0)
        model.left.weight.fill_(127.0)
        model.right.weight.fill_(127.0)

    sima_freeze_qat(model)

    for _, input_fq, weight_fq, output_fq in (source, left, right):
        assert bool((input_fq.scale * weight_fq.scale / output_fq.scale <= 1.0).all())


def test_solver_coarsens_exact_shift_zero_boundary() -> None:
    inputs = torch.randn(1, 3, 4, 4)
    model = sima_prepare_qat_model(Conv2dModel(), (inputs,), "cpu")
    model(inputs)
    _, input_fq, weight_fq, output_fq = _conv_contracts(model)[0]
    _set_grid(input_fq, 1.0)
    _set_grid(output_fq, 1.0)
    with torch.no_grad():
        model.conv.weight.fill_(127.0)

    sima_freeze_qat(model)

    assert float(output_fq.scale) > 1.0
    assert bool((weight_fq.scale * 127.0 >= 127.0).all())
    assert bool((input_fq.scale * weight_fq.scale / output_fq.scale <= 1.0).all())


def test_asymmetric_retargeted_grid_persists_to_onnx(tmp_path) -> None:
    inputs = torch.randn(1, 3, 4, 4)
    model = sima_prepare_qat_model(Conv2dModel(), (inputs,), "cpu")
    model(inputs)
    _, input_fq, _, output_fq = _conv_contracts(model)[0]
    _set_grid(input_fq, 1.0)
    _set_grid(output_fq, 0.01, zero_point=-37)
    expected_zero_point = output_fq.activation_post_process.calculate_qparams()[1].clone()
    with torch.no_grad():
        model.conv.weight.fill_(127.0)
    sima_freeze_qat(model)
    frozen_scale = output_fq.scale.detach().cpu().numpy().copy()
    frozen_zero_point = output_fq.zero_point.detach().cpu().numpy().copy()
    np.testing.assert_array_equal(
        frozen_zero_point,
        expected_zero_point.detach().cpu().numpy(),
    )
    retarget = model.meta["qat_activation_retargets"][-1]
    assert retarget["zero_point_before"] == -37
    assert retarget["zero_point_after"] == -37

    finalized = sima_finalize_qat_model(model)
    output = tmp_path / "asymmetric.onnx"
    sima_export_onnx(finalized, (inputs,), str(output), device="cpu")
    exported = onnx.load(output)
    conv = next(node for node in exported.graph.node if node.op_type == "Conv")
    quantize = next(
        node for node in exported.graph.node
        if node.op_type == "QuantizeLinear" and node.input[0] == conv.output[0]
    )
    np.testing.assert_array_equal(_onnx_value(exported, quantize.input[1]), frozen_scale)
    np.testing.assert_array_equal(_onnx_value(exported, quantize.input[2]), frozen_zero_point.astype(np.int8))


def test_nonpersistable_observer_fails_without_mutation() -> None:
    inputs = torch.randn(1, 3, 4, 4)
    model = sima_prepare_qat_model(Conv2dModel(), (inputs,), "cpu")
    model(inputs)
    _, input_fq, _, output_fq = _conv_contracts(model)[0]
    _set_grid(input_fq, 1.0)
    output_fq.activation_post_process = NonPersistableObserver(0.001, 0)
    with torch.no_grad():
        model.conv.weight.fill_(1.0)
    snapshot = _snapshot_freeze_state(model)

    with pytest.raises(RuntimeError, match="cannot persist a retargeted grid"):
        sima_freeze_qat(model)

    _assert_freeze_state_equal(model, snapshot)


def test_nonfinite_activation_grid_fails_atomically_without_coarsening() -> None:
    inputs = torch.randn(1, 3, 4, 4)
    model = sima_prepare_qat_model(Conv2dModel(), (inputs,), "cpu")
    model(inputs)
    _, input_fq, _, _ = _conv_contracts(model)[0]
    input_fq.activation_post_process.min_val.fill_(float("nan"))
    input_fq.activation_post_process.max_val.fill_(float("nan"))
    snapshot = _snapshot_freeze_state(model)

    with pytest.raises(RuntimeError, match="valid activation qparams"):
        sima_freeze_qat(model)

    _assert_freeze_state_equal(model, snapshot)
    assert "qat_activation_retargets" not in model.meta


def test_late_dynamic_contract_failure_is_fully_atomic() -> None:
    inputs = torch.randn(1, 3, 4, 4)
    runtime_weight = torch.randn(4, 4, 3, 3)
    model = sima_prepare_qat_model(
        ValidThenDynamicConv(),
        (inputs, runtime_weight),
        "cpu",
    )
    model(inputs, runtime_weight)
    snapshot = _snapshot_freeze_state(model)

    with pytest.raises(RuntimeError, match="could not determine complete"):
        sima_freeze_qat(model)

    _assert_freeze_state_equal(model, snapshot)


def test_static_weight_views_and_slices_freeze_but_unsafe_sources_fail() -> None:
    inputs = torch.randn(1, 3, 4, 4)
    accepted = sima_prepare_qat_model(ViewSlicedWeightConv(), (inputs,), "cpu")
    accepted(inputs)
    sima_freeze_qat(accepted)
    assert bool(accepted.qat_frozen.item())

    rejected = sima_prepare_qat_model(UnsupportedWeightCallable(), (inputs,), "cpu")
    rejected(inputs)
    with pytest.raises(RuntimeError, match="unsupported operation aten.sin.default"):
        sima_freeze_qat(rejected)

    with pytest.raises(RuntimeError, match="produced tuple, expected Tensor"):
        _resolve_static_weight_tensor(accepted, (operator.add, 1))


def test_cpu_wrapper_buffers_follow_graph_device() -> None:
    inputs = torch.randn(1, 3, 4, 4)
    prepared = sima_prepare_qat_model(Conv2dModel(), (inputs,), "cpu")
    assert prepared.qat_state.device.type == "cpu"
    assert prepared.qat_frozen.device.type == "cpu"
    prepared(inputs)
    sima_freeze_qat(prepared)
    finalized = sima_finalize_qat_model(prepared)
    assert finalized.qat_state.device.type == "cpu"
    assert finalized.qat_frozen.device.type == "cpu"


def test_corrupted_observers_preserve_prepared_finalized_and_onnx_qparams(tmp_path) -> None:
    inputs = torch.randn(1, 3, 4, 4)
    model = sima_prepare_qat_model(Conv2dModel(), (inputs,), "cpu")
    model(inputs)
    sima_freeze_qat(model)
    fake_quantizers = [module for module in model.modules() if isinstance(module, FakeQuantizeBase)]
    frozen = Counter(
        _qparam_key(
            module.scale.detach().cpu().numpy(),
            module.zero_point.detach().cpu().numpy(),
        )
        for module in fake_quantizers
    )
    for module in fake_quantizers:
        module.activation_post_process.min_val.fill_(-10000.0)
        module.activation_post_process.max_val.fill_(10000.0)
        scale, zero_point = module.calculate_qparams()
        torch.testing.assert_close(scale, module.scale, rtol=0, atol=0)
        torch.testing.assert_close(zero_point, module.zero_point, rtol=0, atol=0)
    with torch.no_grad():
        prepared_output = model(inputs).clone()
    finalized = sima_finalize_qat_model(model)
    with torch.no_grad():
        finalized_output = finalized(inputs)
    torch.testing.assert_close(finalized_output, prepared_output, rtol=0, atol=0)

    output = tmp_path / "corrupted_observers.onnx"
    sima_export_onnx(finalized, (inputs,), str(output), device="cpu")
    exported = onnx.load(output)
    exported_qparams = Counter()
    quantized_values = set()
    for node in exported.graph.node:
        if node.op_type != "QuantizeLinear":
            continue
        quantized_values.update(node.output)
        exported_qparams[_qparam_key(
            _onnx_value(exported, node.input[1]),
            _onnx_value(exported, node.input[2]),
        )] += 1
    for node in exported.graph.node:
        if node.op_type != "DequantizeLinear" or node.input[0] in quantized_values:
            continue
        exported_qparams[_qparam_key(
            _onnx_value(exported, node.input[1]),
            _onnx_value(exported, node.input[2]),
        )] += 1
    assert exported_qparams == frozen
