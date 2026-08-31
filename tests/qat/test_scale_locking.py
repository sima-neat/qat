"""Numerical and failure-mode tests for shift-aware scale locking."""

import pytest
import torch
from torch.ao.quantization.fake_quantize import FakeQuantizeBase

from sima_qat import sima_finalize_qat_model, sima_freeze_qat, sima_prepare_qat_model
from sima_qat.qat_api import (
    _fake_quant_module,
    _find_output_fake_quant,
    _safe_power_of_two_weight_scale,
)

from tests.operators.cases import Conv2dModel


pytestmark = pytest.mark.regression


class DynamicWeightLinear(torch.nn.Module):
    def forward(self, inputs, weight):
        return torch.nn.functional.linear(inputs, weight)


class DynamicWeightConv(torch.nn.Module):
    def forward(self, inputs, weight):
        return torch.nn.functional.conv2d(inputs, weight, padding=1)


class UnsupportedStaticWeightExpression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(4, 3, 3, 3))

    def forward(self, inputs):
        return torch.nn.functional.conv2d(inputs, torch.sin(self.weight), padding=1)


class TinyWeightLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 4, bias=False)
        with torch.no_grad():
            self.linear.weight.uniform_(-1e-3, 1e-3)

    def forward(self, inputs):
        return self.linear(inputs)


def test_exact_power_of_two_minimum_uses_next_coarser_grid() -> None:
    required_scale = torch.tensor([2.0**-4], dtype=torch.float32)

    locked_scale = _safe_power_of_two_weight_scale(
        torch.tensor([1.0]),
        torch.tensor([1.0]),
        required_scale,
    )

    assert locked_scale[0] >= required_scale[0]
    assert 2.0**-4 < locked_scale[0] < 2.0**-3


def test_scale_larger_than_shift_zero_grid_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="shift 0"):
        _safe_power_of_two_weight_scale(
            torch.tensor([1.0]),
            torch.tensor([1.0]),
            torch.tensor([2.0]),
        )


def test_all_frozen_fake_quantizers_use_stored_qparams_exactly() -> None:
    inputs = torch.randn(2, 8)
    model = sima_prepare_qat_model(TinyWeightLinear(), (inputs,), "cpu")
    model(inputs)
    sima_freeze_qat(model)
    fake_quantizers = [
        module for module in model.modules() if isinstance(module, FakeQuantizeBase)
    ]
    assert fake_quantizers

    with torch.no_grad():
        prepared_output = model(inputs).clone()

    for fake_quantizer in fake_quantizers:
        stored_scale = fake_quantizer.scale.detach().clone()
        stored_zero_point = fake_quantizer.zero_point.detach().clone()
        fake_quantizer.activation_post_process.min_val.fill_(-1000.0)
        fake_quantizer.activation_post_process.max_val.fill_(1000.0)

        converted_scale, converted_zero_point = fake_quantizer.calculate_qparams()
        torch.testing.assert_close(converted_scale, stored_scale, rtol=0, atol=0)
        torch.testing.assert_close(
            converted_zero_point,
            stored_zero_point,
            rtol=0,
            atol=0,
        )

    finalized = sima_finalize_qat_model(model)
    with torch.no_grad():
        finalized_output = finalized(inputs)
    torch.testing.assert_close(finalized_output, prepared_output, rtol=0, atol=0)


def test_lock_uses_weights_newer_than_observer_state() -> None:
    inputs = torch.randn(2, 3, 8, 8)
    model = sima_prepare_qat_model(Conv2dModel(), (inputs,), "cpu")
    model(inputs)
    conv = next(
        node
        for node in model.graph.nodes
        if node.op == "call_function" and node.target == torch.ops.aten.conv2d.default
    )
    input_fq = _fake_quant_module(model, conv.args[0])
    weight_fq = _fake_quant_module(model, conv.args[1])
    output_fq = _find_output_fake_quant(model, conv)
    assert input_fq is not None and weight_fq is not None and output_fq is not None

    input_fq.scale.fill_(1.0)
    output_fq.scale.fill_(1.0)
    weight_fq.scale.fill_(2.0**-4)
    stale_lock = _safe_power_of_two_weight_scale(
        input_fq.scale,
        output_fq.scale,
        weight_fq.scale,
    )
    with torch.no_grad():
        model.conv.weight[0].fill_(float(stale_lock[0]) * 127.0 * 1.01)
    current_max = model.conv.weight[0].detach().abs().max()

    sima_freeze_qat(model)

    assert weight_fq.scale[0] > stale_lock[0]
    assert weight_fq.scale[0] * 127.0 >= current_max


def test_sub_epsilon_weight_qparams_survive_finalization() -> None:
    torch.manual_seed(0)
    inputs = torch.randn(3, 8)
    model = sima_prepare_qat_model(TinyWeightLinear(), (inputs,), "cpu")
    model(inputs)
    linear = next(
        node
        for node in model.graph.nodes
        if node.op == "call_function" and node.target == torch.ops.aten.linear.default
    )
    weight_fq = _fake_quant_module(model, linear.args[1])
    assert weight_fq is not None

    sima_freeze_qat(model)
    locked_scale = weight_fq.scale.detach().clone()
    assert bool((locked_scale < 2**-12).all())
    converted_scale, converted_zero_point = weight_fq.calculate_qparams()
    torch.testing.assert_close(converted_scale, locked_scale, rtol=0, atol=0)
    assert bool((converted_zero_point == weight_fq.zero_point).all())

    with torch.no_grad():
        prepared_output = model(inputs).clone()
    finalized = sima_finalize_qat_model(model)
    weight_dq = next(
        node
        for node in finalized.graph.nodes
        if node.op == "call_function"
        and node.target
        == torch.ops.quantized_decomposed.dequantize_per_channel.default
    )
    finalized_scale = getattr(finalized, weight_dq.args[1].target)
    torch.testing.assert_close(finalized_scale, locked_scale, rtol=0, atol=0)
    with torch.no_grad():
        finalized_output = finalized(inputs)
    torch.testing.assert_close(finalized_output, prepared_output, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("source", "inputs"),
    [
        (DynamicWeightLinear(), (torch.randn(2, 8), torch.randn(4, 8))),
        (
            DynamicWeightConv(),
            (torch.randn(2, 3, 8, 8), torch.randn(4, 3, 3, 3)),
        ),
    ],
    ids=["linear", "conv2d"],
)
def test_dynamic_weights_are_rejected_without_mutating_observers(source, inputs) -> None:
    model = sima_prepare_qat_model(source, inputs, "cpu")
    model(*inputs)
    fake_quantizers = [
        module for module in model.modules() if isinstance(module, FakeQuantizeBase)
    ]
    observer_states = [module.observer_enabled.detach().clone() for module in fake_quantizers]

    with pytest.raises(RuntimeError, match="depends on runtime input"):
        sima_freeze_qat(model)

    assert not bool(model.qat_frozen.item())
    for module, observer_state in zip(fake_quantizers, observer_states):
        torch.testing.assert_close(module.observer_enabled, observer_state, rtol=0, atol=0)


def test_unknown_static_weight_operation_is_rejected() -> None:
    inputs = torch.randn(2, 3, 8, 8)
    model = sima_prepare_qat_model(
        UnsupportedStaticWeightExpression(),
        (inputs,),
        "cpu",
    )
    model(inputs)

    with pytest.raises(RuntimeError, match="unsupported operation aten.sin.default"):
        sima_freeze_qat(model)

    assert not bool(model.qat_frozen.item())


def test_freeze_is_atomic_when_a_layer_has_incomplete_qparams() -> None:
    inputs = torch.randn(2, 3, 8, 8)
    model = sima_prepare_qat_model(Conv2dModel(), (inputs,), "cpu")
    model(inputs)
    conv = next(
        node
        for node in model.graph.nodes
        if node.op == "call_function" and node.target == torch.ops.aten.conv2d.default
    )
    output_fq = _find_output_fake_quant(model, conv)
    assert output_fq is not None
    output_fq_node = next(
        node for node in model.graph.nodes if _fake_quant_module(model, node) is output_fq
    )
    model.set_submodule(output_fq_node.target, torch.nn.Identity())
    input_fq = _fake_quant_module(model, conv.args[0])
    weight_fq = _fake_quant_module(model, conv.args[1])
    assert input_fq is not None and weight_fq is not None
    original_weight_scale = weight_fq.scale.detach().clone()

    with pytest.raises(RuntimeError, match="could not determine complete"):
        sima_freeze_qat(model)

    assert not bool(model.qat_frozen.item())
    assert bool(input_fq.observer_enabled.item())
    torch.testing.assert_close(weight_fq.scale, original_weight_scale, rtol=0, atol=0)
