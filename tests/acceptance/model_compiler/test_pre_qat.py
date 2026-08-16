"""Compile a float ONNX model after Model Compiler PTQ."""

from pathlib import Path

from ._support import (
    COMPILER_TEST_MARKS,
    _CompilerApi,
    _OnnxPair,
    _lower_for_compile,
    _save_and_compile,
)


pytestmark = COMPILER_TEST_MARKS


def test_pre_qat_float_onnx_compiles(
    onnx_pair: _OnnxPair,
    compiler_api: _CompilerApi,
    tmp_path: Path,
) -> None:
    model_name = "pre_qat"
    model = _lower_for_compile(
        compiler_api,
        onnx_pair.pre_qat,
        model_name,
    )
    _save_and_compile(model, tmp_path / "pre-qat-compile", model_name)
