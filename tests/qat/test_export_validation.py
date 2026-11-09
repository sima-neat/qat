"""An executable diagnostic reference must never become a compiler input by accident."""

from __future__ import annotations

import importlib.util
from importlib.machinery import PathFinder

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

pytestmark = pytest.mark.regression
# Locate the actual package on sys.path, including the installed-wheel CI path,
# without importing the training API just to inspect a standalone CPU helper.
PACKAGE = PathFinder.find_spec("sima_qat")
assert PACKAGE is not None
SPEC = PathFinder.find_spec(
    "sima_qat.export_validation", PACKAGE.submodule_search_locations
)
assert SPEC is not None
checks = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checks)


def model(
    path, *, eligible="false", purpose="NONDEPLOYABLE_DIAGNOSTIC_REFERENCE", guard=False
):
    value = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1])
    nodes = [helper.make_node("Identity", ["x"], ["y"])]
    outputs = [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1])]
    if guard:
        nodes.append(helper.make_node("Equal", ["x", "x"], ["domain_valid"]))
        outputs.append(
            helper.make_tensor_value_info("domain_valid", TensorProto.BOOL, [1, 1])
        )
    result = helper.make_model(
        helper.make_graph(nodes, "fixture", [value], outputs),
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=9,
    )
    helper.set_model_props(
        result,
        {
            "sima_qat.compiler_eligible": eligible,
            "sima_qat.artifact_purpose": purpose,
            "sima_qat.final_checkpoint_sha256": "1" * 64,
        },
    )
    onnx.save(result, path)


def test_diagnostic_inspection_is_nonmutating_and_fails_before_compilation(tmp_path):
    path = tmp_path / "diagnostic.onnx"
    model(path, guard=True)
    before = path.read_bytes()
    result = checks.inspect_export(path, checkpoint_sha256="1" * 64)
    assert len(result["blockers"]) == 3
    assert result["target_qualification"] == "NOT_RUN"
    with pytest.raises(ValueError, match="not a direct compiler input"):
        checks.require_compiler_input(result)
    assert path.read_bytes() == before


def test_positive_declaration_is_only_preflight_not_target_qualification(tmp_path):
    path = tmp_path / "candidate.onnx"
    model(path, eligible="true", purpose="DEPLOYABLE_QAT_EXPORT")
    result = checks.inspect_export(path, checkpoint_sha256="1" * 64)
    checks.require_compiler_input(result)
    assert result["compiler_input_preflight"]
    assert result["target_qualification"] == "NOT_RUN"
    result = checks.inspect_export(path, checkpoint_sha256="2" * 64)
    with pytest.raises(ValueError, match="checkpoint identity"):
        checks.require_compiler_input(result)


def test_output_comparison_preserves_signed_zero_and_checks_guards(tmp_path):
    left, right = tmp_path / "left.npz", tmp_path / "right.npz"
    np.savez(left, y=np.array([0.0], dtype=np.float32), domain=np.array(True))
    np.savez(right, y=np.array([-0.0], dtype=np.float32), domain=np.array(True))
    result = checks.compare_output_files(left, right)
    assert not result["bitwise_exact"]
    assert result["outputs"]["y"]["different_bit_patterns"] == 1
    assert result["execution_scope"] == "existing_arrays_only_no_new_model_execution"
    np.savez(right, y=np.array([0.0], dtype=np.float32), domain=np.array(False))
    with pytest.raises(ValueError, match="validity guard"):
        checks.compare_output_files(left, right)


@pytest.mark.parametrize("problem", ["names", "shape", "dtype", "nan", "empty"])
def test_incomplete_comparisons_fail_instead_of_reporting_parity(tmp_path, problem):
    left, right = tmp_path / "a.npz", tmp_path / "b.npz"
    np.savez(left, y=np.array([1], dtype=np.float32))
    options = {
        "names": {"z": np.array([1], dtype=np.float32)},
        "shape": {"y": np.array([1, 2], dtype=np.float32)},
        "dtype": {"y": np.array([1], dtype=np.float64)},
        "nan": {"y": np.array([np.nan], dtype=np.float32)},
        "empty": {},
    }
    np.savez(right, **options[problem])
    with pytest.raises(ValueError):
        checks.compare_output_files(left, right)
