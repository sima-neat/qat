"""Device-kwarg rewriting across the QAT lifecycle."""

import pytest
import torch

from sima_qat import (
    sima_export_onnx,
    sima_finalize_qat_model,
    sima_freeze_qat,
    sima_prepare_qat_model,
)
from sima_qat.qat_api import check_graph_nodes, device_modifier_ops


pytestmark = pytest.mark.regression


class RandomMaskModel(torch.nn.Module):
    def forward(self, inputs):
        mask = torch.empty(
            [inputs.shape[0], 1, 1, 1],
            dtype=inputs.dtype,
            device=inputs.device,
        ).bernoulli_(0.95)
        return inputs * mask


def _device_kwargs(model):
    return [
        node.kwargs["device"]
        for node in model.graph.nodes
        if node.target in device_modifier_ops
    ]


def test_finalization_moves_the_model_and_device_kwargs_to_cpu() -> None:
    inputs = torch.randn(2, 3, 8, 8)
    prepared = sima_prepare_qat_model(RandomMaskModel(), (inputs,), "cpu")
    assert _device_kwargs(prepared)
    assert all(device == "cpu" for device in _device_kwargs(prepared))

    check_graph_nodes(prepared, "cuda")
    assert all(device == "cuda" for device in _device_kwargs(prepared))

    check_graph_nodes(prepared, "cpu")
    prepared(inputs)
    sima_freeze_qat(prepared)
    check_graph_nodes(prepared, "cuda")
    finalized = sima_finalize_qat_model(prepared)

    assert all(tensor.device.type == "cpu" for tensor in finalized.parameters())
    assert all(tensor.device.type == "cpu" for tensor in finalized.buffers())
    assert all(device == "cpu" for device in _device_kwargs(finalized))
    assert torch.isfinite(finalized(inputs)).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_training_model_finalizes_and_exports_on_cpu(
    tmp_path, monkeypatch
) -> None:
    inputs = torch.randn(2, 3, 8, 8, device="cuda")
    prepared = sima_prepare_qat_model(RandomMaskModel(), (inputs,), "cuda")
    prepared(inputs)
    sima_freeze_qat(prepared)

    finalized = sima_finalize_qat_model(prepared)
    assert all(tensor.device.type == "cpu" for tensor in finalized.buffers())
    assert all(device == "cpu" for device in _device_kwargs(finalized))

    observed = {}

    def record_export(model, export_inputs, *_args, **_kwargs):
        observed["model_devices"] = {
            tensor.device.type for tensor in model.buffers()
        }
        observed["input_devices"] = {
            tensor.device.type for tensor in export_inputs
        }

    monkeypatch.setattr(torch.onnx, "export", record_export)
    exported_model = sima_export_onnx(
        finalized, (inputs,), str(tmp_path / "random_mask.onnx")
    )

    assert observed == {
        "model_devices": {"cpu"},
        "input_devices": {"cpu"},
    }
    assert inputs.device.type == "cuda"
    assert all(tensor.device.type == "cpu" for tensor in exported_model.buffers())
