"""Coverage and annotation tests for the supported QAT operator matrix."""

import pytest
from torch.ao.quantization.fake_quantize import FakeQuantizeBase
from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import OP_TO_ANNOTATOR

from sima_qat.sima_quantizer import SimaQuantizer

from .cases import ALL_OPERATOR_CASES, OPERATOR_CASES, case_ids
from .helpers import fake_quantizers, prepare_case


pytestmark = pytest.mark.regression


def test_every_active_pattern_has_an_operator_case() -> None:
    active_patterns = set(SimaQuantizer.STATIC_QAT_ONLY_OPS + SimaQuantizer.STATIC_OPS)
    covered_patterns = {case.pattern for case in OPERATOR_CASES}

    assert covered_patterns == active_patterns


def test_every_active_pattern_has_a_registered_annotator() -> None:
    active_patterns = SimaQuantizer.STATIC_QAT_ONLY_OPS + SimaQuantizer.STATIC_OPS

    assert all(pattern in OP_TO_ANNOTATOR for pattern in active_patterns)


@pytest.mark.parametrize("case", ALL_OPERATOR_CASES, ids=case_ids)
def test_supported_operator_is_annotated(case) -> None:
    prepared, _ = prepare_case(case)
    matching_nodes = [
        node
        for node in prepared.graph.nodes
        if node.op == "call_function" and node.target in case.annotation_targets
    ]

    assert matching_nodes, f"{case.name} was not captured as its expected ATen operation"
    annotation_present = any(
        getattr(node.meta.get("quantization_annotation"), "_annotated", False)
        for node in matching_nodes
    )
    qdq_boundary_present = any(
        any(
            getattr(argument, "op", None) == "call_module"
            and isinstance(prepared.get_submodule(argument.target), FakeQuantizeBase)
            for argument in node.all_input_nodes
        )
        and any(
            user.op == "call_module"
            and isinstance(prepared.get_submodule(user.target), FakeQuantizeBase)
            for user in node.users
        )
        for node in matching_nodes
    )
    assert annotation_present or qdq_boundary_present, (
        f"{case.name} did not retain an annotation or receive a fake-quant boundary"
    )
    assert fake_quantizers(prepared), f"{case.name} did not receive fake quantization"
