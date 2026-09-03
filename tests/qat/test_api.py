"""Public API contract for the single shift-aware QAT implementation."""

import inspect

import onnx
import pytest
import torch
from torch import nn
from torch.ao.quantization.fake_quantize import FakeQuantizeBase

import sima_qat
from sima_qat import (
    sima_export_onnx,
    sima_finalize_qat_model,
    sima_freeze_qat,
    sima_prepare_qat_model,
)
from sima_qat.sima_quantizer import get_sima_quantization_config

from tests.operators.cases import Conv2dModel


pytestmark = pytest.mark.regression


class BatchTokenLinear(nn.Module):
    """Minimal transformer-style class-token expansion."""

    def __init__(self) -> None:
        super().__init__()
        self.token = nn.Parameter(torch.randn(1, 1, 4))
        self.linear = nn.Linear(4, 4)

    def forward(self, inputs):
        token = self.token.expand(inputs.shape[0], -1, -1)
        return self.linear(torch.cat((token, inputs), dim=1))


class FoldedBatchDirection(nn.Module):
    """A recurrence-like graph whose internal geometry requires batch one."""

    def forward(self, inputs):
        torch._assert(
            inputs.shape[0] == 1,
            "batch is folded into the scan-direction dimension",
        )
        folded = inputs[:, None, :].expand(-1, 2, -1).reshape(2, 4)
        return folded.reshape(1, 2, 4).sum(dim=1)


class DynamicBatchNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(3)

    def forward(self, inputs):
        return self.bn(inputs)


def test_public_api_exposes_only_the_single_qat_mode() -> None:
    assert sima_qat.__all__ == [
        "sima_prepare_qat_model",
        "sima_freeze_qat",
        "sima_finalize_qat_model",
        "sima_export_onnx",
    ]
    assert "shift_aware" not in inspect.signature(sima_prepare_qat_model).parameters
    assert "dynamic_shapes" not in inspect.signature(sima_prepare_qat_model).parameters
    dynamic_batch = inspect.signature(sima_prepare_qat_model).parameters[
        "dynamic_batch"
    ]
    assert dynamic_batch.kind is inspect.Parameter.KEYWORD_ONLY
    assert dynamic_batch.default is False

    config = get_sima_quantization_config(is_qat=True)
    assert isinstance(config.weight.observer_or_fake_quant_ctr(), FakeQuantizeBase)


def test_finalize_retains_auto_freeze_compatibility() -> None:
    inputs = torch.randn(2, 3, 8, 8)
    prepared = sima_prepare_qat_model(Conv2dModel(), (inputs,), "cpu")
    prepared(inputs)

    with pytest.warns(UserWarning, match="scales will be locked now"):
        finalized = sima_finalize_qat_model(prepared)

    assert bool(finalized.qat_frozen.item())
    assert torch.isfinite(finalized(inputs)).all()


def test_prepare_accepts_dynamic_training_batch() -> None:
    example = torch.randn(1, 2, 4)
    prepared = sima_prepare_qat_model(
        BatchTokenLinear(),
        (example,),
        "cpu",
        dynamic_batch=True,
    )

    for batch_size in (1, 2, 4):
        output = prepared(torch.randn(batch_size, 2, 4))
        assert output.shape == (batch_size, 3, 4)


def test_dynamic_training_model_exports_static_batch_one(tmp_path) -> None:
    example = torch.randn(1, 2, 4)
    prepared = sima_prepare_qat_model(
        BatchTokenLinear(),
        (example,),
        "cpu",
        dynamic_batch=True,
    )
    prepared(example)
    sima_freeze_qat(prepared)
    finalized = sima_finalize_qat_model(prepared)

    output_path = tmp_path / "batch_token_linear.onnx"
    sima_export_onnx(
        finalized,
        (example,),
        str(output_path),
        input_names=["tokens"],
        output_names=["output"],
        device="cpu",
    )

    exported = onnx.load(output_path)
    batch_dimension = exported.graph.input[0].type.tensor_type.shape.dim[0]
    assert batch_dimension.dim_value == 1


def test_static_capture_preserves_folded_batch_direction_geometry() -> None:
    example = torch.randn(1, 4)
    prepared = sima_prepare_qat_model(
        FoldedBatchDirection(),
        (example,),
        "cpu",
    )

    output = prepared(example)
    assert output.shape == (1, 4)
    assert torch.isfinite(output).all()


def test_dynamic_batch_opt_in_fails_closed_for_folded_geometry() -> None:
    example = torch.randn(1, 4)
    with pytest.raises(RuntimeError, match="Dynamic-batch QAT capture"):
        sima_prepare_qat_model(
            FoldedBatchDirection(),
            (example,),
            "cpu",
            dynamic_batch=True,
        )


def test_dynamic_validation_does_not_mutate_returned_batchnorm_state() -> None:
    example = torch.randn(1, 3, 4, 4)
    prepared = sima_prepare_qat_model(
        DynamicBatchNorm(),
        (example,),
        "cpu",
        dynamic_batch=True,
    )

    running_means = [
        value
        for name, value in prepared.named_buffers()
        if "running_mean" in name
    ]
    running_vars = [
        value
        for name, value in prepared.named_buffers()
        if "running_var" in name
    ]
    assert running_means and running_vars
    assert all(bool((value == 0).all()) for value in running_means)
    assert all(bool((value == 1).all()) for value in running_vars)
    assert prepared.training


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_finalize_keeps_wrapper_state_on_cuda_graph_device() -> None:
    inputs = torch.randn(2, 3, 8, 8, device="cuda")
    prepared = sima_prepare_qat_model(Conv2dModel(), (inputs,), "cuda")
    prepared(inputs)
    sima_freeze_qat(prepared)

    finalized = sima_finalize_qat_model(prepared)

    assert finalized.qat_state.device.type == "cuda"
    assert finalized.qat_frozen.device.type == "cuda"
    assert torch.isfinite(finalized(inputs)).all()
