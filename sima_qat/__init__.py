"""Public SiMa QAT package interface."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from sima_qat.qat_api import (
    sima_export_onnx,
    sima_finalize_qat_model,
    sima_prepare_qat_model,
)


def _resolve_version() -> str:
    source_root = Path(__file__).resolve().parents[1]
    source_version = source_root / "VERSION.in"
    if source_version.is_file() and (source_root / "pyproject.toml").is_file():
        return source_version.read_text(encoding="utf-8").strip()
    try:
        return version("sima-qat")
    except PackageNotFoundError:
        return "0+unknown"


__version__ = _resolve_version()

__all__ = [
    "sima_prepare_qat_model",
    "sima_finalize_qat_model",
    "sima_export_onnx",
    "__version__",
]
