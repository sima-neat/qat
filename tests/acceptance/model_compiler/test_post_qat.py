"""Compile Q/DQ ONNX while preserving the learned QAT contract."""

from pathlib import Path

from ._support import (
    COMPILER_TEST_MARKS,
    _CompilerApi,
    _assert_qat_execution_matches,
    _assert_qat_lowering_preserved,
    _lower_for_compile,
    _qat_contract,
    _save_and_compile,
)


pytestmark = COMPILER_TEST_MARKS


def test_qat_onnx_compiles_and_preserves_quantization(
    qat_onnx: Path,
    compiler_api: _CompilerApi,
    compiler_artifact_root: Path,
) -> None:
    model_name = "post_qat"
    contract = _qat_contract(qat_onnx)
    model = _lower_for_compile(
        compiler_api,
        qat_onnx,
        model_name,
    )
    _assert_qat_execution_matches(
        model,
        compiler_api,
        qat_onnx,
        contract.output_scale,
    )
    output_dir = compiler_artifact_root / "post_qat" / "compile"
    _save_and_compile(model, output_dir, model_name)
    _assert_qat_lowering_preserved(output_dir, model_name, contract)
