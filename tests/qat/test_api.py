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


def test_public_api_exposes_only_the_single_qat_mode() -> None:
    assert sima_qat.__all__ == [
        "sima_prepare_qat_model",
        "sima_freeze_qat",
        "sima_finalize_qat_model",
        "sima_export_onnx",
    ]
    assert "shift_aware" not in inspect.signature(sima_prepare_qat_model).parameters
    assert "dynamic_shapes" not in inspect.signature(sima_prepare_qat_model).parameters

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
    )

    for batch_size in (1, 2, 4):
        output = prepared(torch.randn(batch_size, 2, 4))
        assert output.shape == (batch_size, 3, 4)


def test_dynamic_training_model_exports_static_batch_one(tmp_path) -> None:
    example = torch.randn(1, 2, 4)
    prepared = sima_prepare_qat_model(BatchTokenLinear(), (example,), "cpu")
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
