"""Drift guards for the versioned QAT operator contract."""

from __future__ import annotations

import re

import pytest

from sima_qat.operator_manifest import (
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


def test_manifest_is_versioned_and_has_unique_families() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", OPERATOR_MANIFEST_VERSION)
    assert ONNX_OPSET == 17
    assert len(MANIFEST_BY_FAMILY) == len(OPERATOR_MANIFEST)


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


def test_passthrough_entries_do_not_claim_annotation_coverage() -> None:
    for entry in OPERATOR_MANIFEST:
        if entry.behavior is QuantizationBehavior.PASSTHROUGH:
            assert entry.status is SupportStatus.DEFERRED
            assert entry.annotator is None
            assert not entry.positive_test_requirements
            assert not entry.test_case_ids
