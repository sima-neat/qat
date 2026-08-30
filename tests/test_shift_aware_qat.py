#**************************************************************************
#||                        SiMa.ai CONFIDENTIAL                          ||
#||   Unpublished Copyright (c) 2024 SiMa.ai, All Rights Reserved.       ||
#**************************************************************************
import onnxruntime
import pytest
import torch
from torch.ao.quantization.fake_quantize import FakeQuantizeBase
from torch.ao.quantization.observer import PerChannelMinMaxObserver

from sima_qat.qat_api import (
    _SHIFT_AWARE_OPS,
    _fake_quant_module,
    _find_output_fake_quant,
    _safe_power_of_two_weight_scale,
    sima_export_onnx,
    sima_finalize_qat_model,
    sima_freeze_qat,
    sima_prepare_qat_model,
)
from sima_qat.sima_quantizer import get_sima_quantization_config


class TinyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(4, 2, 1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        return self.conv2(x).mean(dim=(-2, -1))


def _weight_fake_quantizers(model):
    return [
        module
        for module in model.modules()
        if isinstance(module, FakeQuantizeBase)
        and module.qscheme in (torch.per_channel_affine, torch.per_channel_symmetric)
    ]


def _activation_fake_quantizers(model):
    return [
        module
        for module in model.modules()
        if isinstance(module, FakeQuantizeBase)
        and module.qscheme not in (torch.per_channel_affine, torch.per_channel_symmetric)
    ]


@pytest.mark.regression
def test_shift_aware_weight_fake_quant_is_default_and_legacy_is_available():
    default_config = get_sima_quantization_config(is_qat=True)
    default_weight_module = default_config.weight.observer_or_fake_quant_ctr()
    assert isinstance(default_weight_module, FakeQuantizeBase)

    legacy_config = get_sima_quantization_config(is_qat=True, shift_aware=False)
    legacy_weight_module = legacy_config.weight.observer_or_fake_quant_ctr()
    assert isinstance(legacy_weight_module, PerChannelMinMaxObserver)
    assert not isinstance(legacy_weight_module, FakeQuantizeBase)

    legacy_model = sima_prepare_qat_model(
        TinyClassifier(),
        (torch.randn(2, 3, 8, 8),),
        "cpu",
        shift_aware=False,
    )
    assert not bool(legacy_model.shift_aware_qat.item())
    assert _weight_fake_quantizers(legacy_model) == []
    assert any(isinstance(module, PerChannelMinMaxObserver) for module in legacy_model.modules())


@pytest.mark.regression
def test_two_epoch_cpu_qat_locks_model_compiler_compatible_scales(tmp_path):
    torch.manual_seed(7)
    inputs = torch.randn(8, 3, 8, 8)
    targets = torch.randn(8, 2)
    model = sima_prepare_qat_model(TinyClassifier(), (inputs[:2],), "cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    # Epoch 1: observers and trainable weights warm up together.
    for batch in range(0, len(inputs), 2):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(inputs[batch:batch + 2]), targets[batch:batch + 2])
        loss.backward()
        optimizer.step()

    sima_freeze_qat(model)
    locked_scales = [module.scale.detach().clone() for module in _weight_fake_quantizers(model)]

    # Epoch 2: fake quantization remains active, gradients flow, and locked scales stay fixed.
    for batch in range(0, len(inputs), 2):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(inputs[batch:batch + 2]), targets[batch:batch + 2])
        loss.backward()
        optimizer.step()

    assert torch.isfinite(loss)
    assert model.conv1.weight.grad is not None
    assert model.conv1.weight.grad.abs().sum() > 0
    assert bool(model.shift_aware_qat.item())
    assert bool(model.qat_frozen.item())
    for module, locked_scale in zip(_weight_fake_quantizers(model), locked_scales):
        torch.testing.assert_close(module.scale, locked_scale, rtol=0, atol=0)

    # Reproduce the Model Compiler's normalization boundary calculation. The
    # normalized correction must be effectively 1, so folding it preserves every INT8 code.
    checked = 0
    for node in model.graph.nodes:
        if node.op != "call_function" or node.target not in _SHIFT_AWARE_OPS:
            continue
        input_fq = _fake_quant_module(model, node.args[0])
        weight_fq = _fake_quant_module(model, node.args[1])
        output_fq = _find_output_fake_quant(model, node)
        if input_fq is None or weight_fq is None or output_fq is None:
            continue

        ratio = input_fq.scale * weight_fq.scale / output_fq.scale
        shifts = -torch.ceil(torch.log2(ratio))
        normalized = ratio * torch.pow(2.0, shifts)
        assert torch.all(normalized <= 1.0)
        assert torch.all(normalized > 0.99999)

        weight_node = node.args[1].args[0]
        weight = getattr(model, weight_node.target.split(".")[0]).weight.detach()
        scale_shape = (len(weight_fq.scale),) + (1,) * (weight.ndim - 1)
        weight_codes = torch.round(weight / weight_fq.scale.reshape(scale_shape)).clamp(-127, 127)
        folded_codes = torch.round(weight_codes * normalized.reshape(scale_shape))
        torch.testing.assert_close(folded_codes, weight_codes, rtol=0, atol=0)
        checked += 1

    assert checked == 2

    # PyTorch's fake-quant kernel and ONNX Runtime's QuantizeLinear are both
    # valid INT8 realizations, but CPU builds can resolve an exact half-way
    # value to adjacent codes.  Capture the final activation quantum before
    # stripping the scaffold so the parity check below permits that one-code
    # ambiguity without hiding a larger export mismatch.
    final_output_fake_quant = next(
        _find_output_fake_quant(model, node)
        for node in reversed(tuple(model.graph.nodes))
        if node.op == "call_function"
        and node.target in _SHIFT_AWARE_OPS
        and _find_output_fake_quant(model, node) is not None
    )
    final_output_scale = float(final_output_fake_quant.scale.detach().abs().max())

    finalized = sima_finalize_qat_model(model)
    finalized_state = finalized.state_dict()
    assert "shift_aware_qat" not in finalized_state
    assert "qat_frozen" not in finalized_state
    output_path = tmp_path / "tiny_shift_aware.onnx"
    sima_export_onnx(finalized, (inputs[:2],), str(output_path), device="cpu")

    with torch.no_grad():
        pytorch_output = finalized(inputs[:2]).cpu()
    session = onnxruntime.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    onnx_output = session.run(None, {session.get_inputs()[0].name: inputs[:2].numpy()})[0]
    torch.testing.assert_close(
        torch.from_numpy(onnx_output),
        pytorch_output,
        rtol=0,
        atol=final_output_scale * (1 + 1e-5),
    )


@pytest.mark.regression
def test_checkpoint_resume_and_legacy_compatibility():
    inputs = torch.randn(2, 3, 8, 8)

    shift_aware_model = sima_prepare_qat_model(TinyClassifier(), (inputs,), "cpu")
    shift_aware_model(inputs)
    sima_freeze_qat(shift_aware_model)
    shift_aware_state = shift_aware_model.state_dict()
    assert not any(key.endswith("sima_shift") for key in shift_aware_state)

    resumed_shift_aware = sima_prepare_qat_model(TinyClassifier(), (inputs,), "cpu")
    resumed_shift_aware.load_state_dict(shift_aware_state)
    assert bool(resumed_shift_aware.shift_aware_qat.item())
    assert bool(resumed_shift_aware.qat_frozen.item())

    legacy_model = sima_prepare_qat_model(
        TinyClassifier(),
        (inputs,),
        "cpu",
        shift_aware=False,
    )
    legacy_model(inputs)
    old_checkpoint = legacy_model.state_dict()
    del old_checkpoint["shift_aware_qat"]
    del old_checkpoint["qat_frozen"]

    resumed_legacy = sima_prepare_qat_model(
        TinyClassifier(),
        (inputs,),
        "cpu",
        shift_aware=False,
    )
    resumed_legacy.load_state_dict(old_checkpoint)
    assert not bool(resumed_legacy.shift_aware_qat.item())
    assert not bool(resumed_legacy.qat_frozen.item())

    default_model = sima_prepare_qat_model(TinyClassifier(), (inputs,), "cpu")
    with pytest.raises(RuntimeError, match="predates shift-aware QAT"):
        default_model.load_state_dict(old_checkpoint)

    with pytest.raises(RuntimeError, match="mode does not match"):
        resumed_legacy.load_state_dict(shift_aware_state)


@pytest.mark.regression
def test_freeze_uses_exact_observer_qparams_for_learned_activation_scales():
    inputs = torch.randn(2, 3, 8, 8)
    model = sima_prepare_qat_model(
        TinyClassifier(), (inputs,), "cpu", full_range_ste=True, learn_scales=True
    )
    model(inputs)
    activation_fake_quantizers = _activation_fake_quantizers(model)
    assert activation_fake_quantizers
    learned_fake_quantizers = [
        fake_quant
        for fake_quant in activation_fake_quantizers
        if hasattr(fake_quant, "log_scale")
    ]
    assert learned_fake_quantizers

    # Simulate learned-scale training drifting away from observer/export
    # qparams while keeping all scale ratios unchanged and representable.
    with torch.no_grad():
        for fake_quant in learned_fake_quantizers:
            fake_quant.log_scale.add_(0.25)
            fake_quant.sync_learned_scale()
    assert any(
        not torch.equal(
            fake_quant.scale,
            fake_quant.activation_post_process.calculate_qparams()[0],
        )
        for fake_quant in learned_fake_quantizers
    )

    sima_freeze_qat(model)
    locked = []
    for fake_quant in activation_fake_quantizers:
        scale, zero_point = fake_quant.activation_post_process.calculate_qparams()
        torch.testing.assert_close(fake_quant.scale, scale, rtol=0, atol=0)
        torch.testing.assert_close(fake_quant.zero_point, zero_point, rtol=0, atol=0)
        if hasattr(fake_quant, "learn_scale"):
            assert not fake_quant.learn_scale
        locked.append(fake_quant.scale.detach().clone())
    model(inputs)
    for fake_quant, expected in zip(activation_fake_quantizers, locked):
        torch.testing.assert_close(fake_quant.scale, expected, rtol=0, atol=0)

    resumed = sima_prepare_qat_model(
        TinyClassifier(), (inputs,), "cpu", full_range_ste=True, learn_scales=True
    )
    resumed.load_state_dict(model.state_dict())
    assert all(
        not fake_quant.learn_scale
        for fake_quant in _activation_fake_quantizers(resumed)
    )


@pytest.mark.regression
def test_freeze_is_atomic_when_a_layer_cannot_be_locked():
    inputs = torch.randn(2, 3, 8, 8)
    model = sima_prepare_qat_model(TinyClassifier(), (inputs,), "cpu")
    model(inputs)

    first_conv = next(
        node for node in model.graph.nodes
        if node.op == "call_function" and node.target in _SHIFT_AWARE_OPS
    )
    output_fq = _find_output_fake_quant(model, first_conv)
    assert output_fq is not None
    output_fq_node = next(
        node for node in model.graph.nodes
        if _fake_quant_module(model, node) is output_fq
    )
    model.set_submodule(output_fq_node.target, torch.nn.Identity())

    input_fq = _fake_quant_module(model, first_conv.args[0])
    weight_fq = _fake_quant_module(model, first_conv.args[1])
    assert input_fq is not None and weight_fq is not None
    original_weight_scale = weight_fq.scale.detach().clone()

    with pytest.raises(RuntimeError, match="could not determine complete"):
        sima_freeze_qat(model)

    assert not bool(model.qat_frozen.item())
    assert bool(input_fq.observer_enabled.item())
    torch.testing.assert_close(weight_fq.scale, original_weight_scale, rtol=0, atol=0)


@pytest.mark.regression
def test_shift_aware_scale_rejects_unrepresentable_or_nonfinite_values():
    with pytest.raises(RuntimeError, match="cannot fit"):
        _safe_power_of_two_weight_scale(
            torch.tensor(1.0),
            torch.tensor(1.0),
            torch.tensor([[128.0]]),
        )

    with pytest.raises(RuntimeError, match="must be finite"):
        _safe_power_of_two_weight_scale(
            torch.tensor(1.0),
            torch.tensor(1.0),
            torch.tensor([[float("inf")]]),
        )

    with pytest.raises(RuntimeError, match="must be positive"):
        _safe_power_of_two_weight_scale(
            torch.tensor(float("nan")),
            torch.tensor(1.0),
            torch.tensor([[1.0]]),
        )
