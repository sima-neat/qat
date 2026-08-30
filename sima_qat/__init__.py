"""SiMa.ai quantization-aware training for strict INT8 deployment."""

from sima_qat.qat_api import (
    sima_export_onnx,
    sima_finalize_qat_model,
    sima_freeze_qat,
    sima_prepare_qat_model,
    sima_project_qat_to_target_grids,
    sima_qat_activation_diagnostics,
    sima_thaw_qat_scales,
)
from sima_qat.range_refinement import refine_activation_ranges
from sima_qat.session import (
    ActivationRangeRefinementReport,
    ActivationRangeSelector,
    QATBundle,
    QATCalibrationReport,
    QATRecipe,
    QATReport,
    QATSession,
    load_recipe,
    prepare,
)

__all__ = [
    "ActivationRangeRefinementReport",
    "ActivationRangeSelector",
    "QATBundle",
    "QATCalibrationReport",
    "QATRecipe",
    "QATReport",
    "QATSession",
    "load_recipe",
    "prepare",
    "refine_activation_ranges",
    "sima_export_onnx",
    "sima_finalize_qat_model",
    "sima_freeze_qat",
    "sima_prepare_qat_model",
    "sima_project_qat_to_target_grids",
    "sima_qat_activation_diagnostics",
    "sima_thaw_qat_scales",
]
