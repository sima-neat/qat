"""Recovery-epoch and checkpoint behavior for the single QAT mode."""

import pytest
import torch

from sima_qat import sima_freeze_qat, sima_prepare_qat_model
from sima_qat.qat_api import _SHIFT_AWARE_OPS, _fake_quant_module, _find_output_fake_quant


pytestmark = pytest.mark.regression


class TinyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(4, 2, 1)

    def forward(self, inputs):
        return self.conv2(self.relu(self.conv1(inputs))).mean(dim=(-2, -1))


def test_recovery_epoch_keeps_grids_locked_and_updates_weights() -> None:
    torch.manual_seed(7)
    inputs = torch.randn(8, 3, 8, 8)
    targets = torch.randn(8, 2)
    model = sima_prepare_qat_model(TinyClassifier(), (inputs[:2],), "cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    for batch in range(0, len(inputs), 2):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(
            model(inputs[batch : batch + 2]),
            targets[batch : batch + 2],
        )
        loss.backward()
        optimizer.step()

    sima_freeze_qat(model)
    locked_scales = {
        name: module.scale.detach().clone()
        for name, module in model.named_modules()
        if hasattr(module, "observer_enabled")
    }
    weights_before = model.conv1.weight.detach().clone()

    for batch in range(0, len(inputs), 2):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(
            model(inputs[batch : batch + 2]),
            targets[batch : batch + 2],
        )
        loss.backward()
        optimizer.step()

    assert torch.isfinite(loss)
    assert not torch.equal(model.conv1.weight, weights_before)
    for name, locked_scale in locked_scales.items():
        torch.testing.assert_close(model.get_submodule(name).scale, locked_scale, rtol=0, atol=0)

    checked = 0
    for node in model.graph.nodes:
        if node.op != "call_function" or node.target not in _SHIFT_AWARE_OPS:
            continue
        input_fq = _fake_quant_module(model, node.args[0])
        weight_fq = _fake_quant_module(model, node.args[1])
        output_fq = _find_output_fake_quant(model, node)
        assert input_fq is not None and weight_fq is not None and output_fq is not None
        ratio = input_fq.scale * weight_fq.scale / output_fq.scale
        shifts = -torch.ceil(torch.log2(ratio))
        normalized = ratio * torch.pow(2.0, shifts)
        assert torch.all(normalized <= 1.0)
        assert torch.all(normalized > 0.99999)
        checked += 1
    assert checked == 2


def test_frozen_checkpoint_resumes_without_legacy_mode_state() -> None:
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

    transitional_state = state.copy()
    transitional_state["shift_aware_qat"] = torch.tensor([True])
    resumed.load_state_dict(transitional_state)

    unsupported_state = state.copy()
    unsupported_state["shift_aware_qat"] = torch.tensor([False])
    with pytest.raises(RuntimeError, match="Only shift-aware"):
        resumed.load_state_dict(unsupported_state)
