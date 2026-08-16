"""Compile a float ONNX model after Model Compiler PTQ."""

from pathlib import Path

from ._support import (
    COMPILER_TEST_MARKS,
    _CompilerApi,
    _lower_for_compile,
    _save_and_compile,
)


pytestmark = COMPILER_TEST_MARKS


def test_float_onnx_compiles_after_ptq(
    float_onnx: Path,
    compiler_api: _CompilerApi,
    compiler_artifact_root: Path,
) -> None:
    model_name = "pre_qat"
    model = _lower_for_compile(
        compiler_api,
        float_onnx,
        model_name,
    )
    _save_and_compile(
        model,
        compiler_artifact_root / "pre_qat" / "compile",
        model_name,
    )
