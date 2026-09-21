"""Drift guards for the versioned QAT operator contract."""

from __future__ import annotations

import re

import pytest

from sima_qat.operator_manifest import (
    COMPILER_INT8_ONNX_OPS_OPSET17,
    MANIFEST_BY_FAMILY,
    ONNX_OPSET,
    ONNX_TEST_CASE_IDS,
    OPERATOR_MANIFEST,
    OPERATOR_MANIFEST_VERSION,
    QuantizationBehavior,
    SupportStatus,
)
from sima_qat.sima_quantizer import SimaQuantizer

from .cases import ALL_OPERATOR_CASES


pytestmark = pytest.mark.regression


EXPECTED_COMPILER_INT8_OPS_OPSET17 = {
    "Abs", "Add", "ArgMax", "AveragePool", "Clip", "Concat", "Conv",
    "ConvTranspose", "DepthToSpace", "Div", "Einsum", "Elu", "Erf", "Exp",
    "Expand", "Gemm", "GlobalAveragePool", "GlobalMaxPool", "HardSigmoid",
    "HardSwish", "InstanceNormalization", "LRN", "LayerNormalization",
    "LeakyRelu", "Log", "LogSoftmax", "MatMul", "MaxPool", "Mean",
    "MeanVarianceNormalization", "Mul", "Neg", "PRelu", "Pad", "Pow",
    "Reciprocal", "ReduceL1", "ReduceLogSum", "ReduceLogSumExp", "ReduceMax",
    "ReduceMean", "ReduceSum", "ReduceSumSquare", "Relu", "Reshape", "Resize",
    "Sigmoid", "Slice", "Softmax", "Softplus", "SpaceToDepth", "Split", "Sqrt",
    "Sub", "Sum", "Tanh", "Tile", "TopK", "Transpose",
}


def test_manifest_is_versioned_and_has_unique_families() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", OPERATOR_MANIFEST_VERSION)
    assert ONNX_OPSET == 17
    assert len(MANIFEST_BY_FAMILY) == len(OPERATOR_MANIFEST)


def test_manifest_matches_pr111_opset17_int8_snapshot() -> None:
    assert COMPILER_INT8_ONNX_OPS_OPSET17 == EXPECTED_COMPILER_INT8_OPS_OPSET17


def test_every_operator_case_is_owned_by_a_manifest_entry() -> None:
    case_ids = {case.name for case in ALL_OPERATOR_CASES}
    manifested_ids = {
        case_id for entry in OPERATOR_MANIFEST for case_id in entry.test_case_ids
    }

    assert manifested_ids == case_ids
    assert ONNX_TEST_CASE_IDS <= case_ids


def test_every_active_annotator_is_declared_in_the_manifest() -> None:
    active = set(SimaQuantizer.STATIC_QAT_ONLY_OPS + SimaQuantizer.STATIC_OPS)
    manifested = {
        annotator
        for entry in OPERATOR_MANIFEST
        for annotator in (
            (() if entry.annotator is None else (entry.annotator,))
            + entry.additional_annotators
        )
        if annotator != "propagate_annotation"
    }

    assert manifested == active


def test_supported_entries_have_executable_positive_cases() -> None:
    for entry in OPERATOR_MANIFEST:
        assert entry.supported_dtypes
        assert entry.expected_onnx_operators
        assert entry.negative_test_requirements
        if entry.status is SupportStatus.SUPPORTED:
            assert entry.test_case_ids, entry.family
            assert entry.positive_test_requirements, entry.family
        if entry.behavior in {
            QuantizationBehavior.ANNOTATION,
            QuantizationBehavior.MIXED_OUTPUT,
        } and entry.status is SupportStatus.SUPPORTED:
            assert entry.annotator, entry.family


def test_rejected_entries_do_not_claim_positive_lifecycle_coverage() -> None:
    for entry in OPERATOR_MANIFEST:
        if entry.status is SupportStatus.REJECTED:
            assert entry.behavior is QuantizationBehavior.UNSUPPORTED
            assert not entry.positive_test_requirements
            assert not entry.test_case_ids
