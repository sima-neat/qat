"""Device-kwarg rewriting across the QAT lifecycle."""

import pytest
import torch

from sima_qat import sima_finalize_qat_model, sima_freeze_qat, sima_prepare_qat_model
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


def test_device_kwargs_can_be_rewritten_before_and_after_freeze() -> None:
    inputs = torch.randn(2, 3, 8, 8)
    prepared = sima_prepare_qat_model(RandomMaskModel(), (inputs,), "cpu")
    assert _device_kwargs(prepared)
    assert all(device == "cpu" for device in _device_kwargs(prepared))

    check_graph_nodes(prepared, "cuda")
    assert all(device == "cuda" for device in _device_kwargs(prepared))

    check_graph_nodes(prepared, "cpu")
    prepared(inputs)
    sima_freeze_qat(prepared)
    finalized = sima_finalize_qat_model(prepared)

    assert torch.isfinite(finalized(inputs)).all()
