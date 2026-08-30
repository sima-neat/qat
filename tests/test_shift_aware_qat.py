#**************************************************************************
#||                        SiMa.ai CONFIDENTIAL                          ||
#||   Unpublished Copyright (c) 2024 SiMa.ai, All Rights Reserved.       ||
#**************************************************************************
import inspect

import onnxruntime
import torch
import pytest

from torch.ao.quantization.fake_quantize import FakeQuantizeBase

from sima_qat.qat_api import (
    _SHIFT_AWARE_OPS,
    _fake_quant_module,
    _find_output_fake_quant,
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


@pytest.mark.regression
def test_qat_always_fake_quantizes_weights():
    config = get_sima_quantization_config(is_qat=True)
    weight_module = config.weight.observer_or_fake_quant_ctr()
    assert isinstance(weight_module, FakeQuantizeBase)
    assert "shift_aware" not in inspect.signature(sima_prepare_qat_model).parameters

    model = sima_prepare_qat_model(
        TinyClassifier(),
        (torch.randn(2, 3, 8, 8),),
        "cpu",
    )
    assert len(_weight_fake_quantizers(model)) == 2


@pytest.mark.regression
def test_two_epoch_cpu_qat_locks_afe_compatible_scales(tmp_path):
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
    assert bool(model.qat_frozen.item())
    for module, locked_scale in zip(_weight_fake_quantizers(model), locked_scales):
        torch.testing.assert_close(module.scale, locked_scale, rtol=0, atol=0)

    # Reproduce AFE's normalization boundary calculation. The normalized
    # correction must be effectively 1, so folding it preserves every INT8 code.
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

    finalized = sima_finalize_qat_model(model)
    output_path = tmp_path / "tiny_shift_aware.onnx"
    sima_export_onnx(finalized, (inputs[:2],), str(output_path), device="cpu")

    with torch.no_grad():
        pytorch_output = finalized(inputs[:2]).cpu()
    session = onnxruntime.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    onnx_output = session.run(None, {session.get_inputs()[0].name: inputs[:2].numpy()})[0]
    torch.testing.assert_close(torch.from_numpy(onnx_output), pytorch_output, rtol=1e-5, atol=1e-6)


@pytest.mark.regression
def test_shift_aware_checkpoint_resume():
    inputs = torch.randn(2, 3, 8, 8)

    model = sima_prepare_qat_model(TinyClassifier(), (inputs,), "cpu")
    model(inputs)
    sima_freeze_qat(model)
    state = model.state_dict()
    assert "shift_aware_qat" not in state
    assert not any(key.endswith("sima_shift") for key in state)

    resumed = sima_prepare_qat_model(TinyClassifier(), (inputs,), "cpu")
    resumed.load_state_dict(state)
    assert bool(resumed.qat_frozen.item())

    # Checkpoints from the first shift-aware release included an always-true
    # mode marker. It is redundant now but remains load-compatible.
    transitional_state = state.copy()
    transitional_state["shift_aware_qat"] = torch.tensor([True])
    resumed.load_state_dict(transitional_state)

    unsupported_state = state.copy()
    unsupported_state["shift_aware_qat"] = torch.tensor([False])
    with pytest.raises(RuntimeError, match="Only shift-aware"):
        resumed.load_state_dict(unsupported_state)


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
