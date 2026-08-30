"""Public API contract for the single shift-aware QAT implementation."""

import inspect

import pytest
import torch
from torch.ao.quantization.fake_quantize import FakeQuantizeBase

import sima_qat
from sima_qat import sima_finalize_qat_model, sima_prepare_qat_model
from sima_qat.sima_quantizer import get_sima_quantization_config

from tests.operators.cases import Conv2dModel


pytestmark = pytest.mark.regression


def test_public_api_exposes_only_the_single_qat_mode() -> None:
    assert sima_qat.__all__ == [
        "sima_prepare_qat_model",
        "sima_freeze_qat",
        "sima_finalize_qat_model",
        "sima_export_onnx",
    ]
    assert "shift_aware" not in inspect.signature(sima_prepare_qat_model).parameters

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
