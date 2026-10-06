"""Prepared-to-finalized graph behavior for all supported operators."""

import pytest
import torch
from torch.ao.quantization.fake_quantize import FakeQuantizeBase

from sima_qat import sima_finalize_qat_model, sima_freeze_qat

from .cases import ALL_OPERATOR_CASES, case_ids
from .helpers import prepare_case


pytestmark = pytest.mark.regression


@pytest.mark.parametrize("case", ALL_OPERATOR_CASES, ids=case_ids)
def test_supported_operator_finalizes_and_executes(case) -> None:
    prepared, inputs = prepare_case(case)
    sima_freeze_qat(prepared)

    finalized = sima_finalize_qat_model(prepared)
    with torch.no_grad():
        output = finalized(*inputs)

    assert torch.isfinite(output).all()
    assert not any(isinstance(module, FakeQuantizeBase) for module in finalized.modules())
    assert any(
        node.op == "call_function"
        and "quantized_decomposed.dequantize" in str(node.target)
        for node in finalized.graph.nodes
    )
