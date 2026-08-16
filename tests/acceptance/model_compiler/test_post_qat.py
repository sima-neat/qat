"""Compile Q/DQ ONNX while preserving the learned QAT contract."""

from pathlib import Path

from ._support import (
    COMPILER_TEST_MARKS,
    _CompilerApi,
    _OnnxPair,
    _assert_qat_execution_matches,
    _assert_qat_lowering_preserved,
    _lower_for_compile,
    _qat_contract,
    _save_and_compile,
)


pytestmark = COMPILER_TEST_MARKS


def test_post_qat_qdq_onnx_compiles(
    onnx_pair: _OnnxPair,
    compiler_api: _CompilerApi,
    tmp_path: Path,
) -> None:
    model_name = "post_qat"
    contract = _qat_contract(onnx_pair.post_qat)
    model = _lower_for_compile(
        compiler_api,
        onnx_pair.post_qat,
        model_name,
    )
    _assert_qat_execution_matches(
        model,
        compiler_api,
        onnx_pair.post_qat,
        contract.output_scale,
    )
    output_dir = tmp_path / "post-qat-compile"
    _save_and_compile(model, output_dir, model_name)
    _assert_qat_lowering_preserved(output_dir, model_name, contract)
