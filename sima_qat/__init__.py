"""SiMa.ai quantization-aware training for strict INT8 deployment."""

from sima_qat.qat_api import (
    sima_export_onnx,
    sima_finalize_qat_model,
    sima_freeze_qat,
    sima_project_qat_to_target_grids,
    sima_thaw_qat_scales,
    sima_qat_activation_diagnostics,
    sima_prepare_qat_model,
)
from sima_qat.session import (
    QATBundle,
    QATCalibrationReport,
    QATRecipe,
    QATReport,
    QATSession,
    load_recipe,
    prepare,
)

__all__ = [
    "QATBundle",
    "QATCalibrationReport",
    "QATRecipe",
    "QATReport",
    "QATSession",
    "load_recipe",
    "prepare",
    "sima_export_onnx",
    "sima_finalize_qat_model",
    "sima_freeze_qat",
    "sima_project_qat_to_target_grids",
    "sima_thaw_qat_scales",
    "sima_qat_activation_diagnostics",
    "sima_prepare_qat_model",
]
