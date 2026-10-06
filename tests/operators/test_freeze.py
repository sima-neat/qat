"""Shift-aware freeze contract across every weighted operator pattern."""

import pytest
import torch

from sima_qat import sima_freeze_qat
from sima_qat.qat_api import _fake_quant_module, _minimum_weight_scale

from .cases import WEIGHTED_CASES, case_ids
from .helpers import assert_shift_realizable, fake_quantizers, prepare_case, weighted_nodes


pytestmark = pytest.mark.regression


@pytest.mark.parametrize("case", WEIGHTED_CASES, ids=case_ids)
def test_weighted_operator_freezes_to_non_clipping_shift_grid(case) -> None:
    prepared, _ = prepare_case(case)

    sima_freeze_qat(prepared)

    assert bool(prepared.qat_frozen.item())
    assert_shift_realizable(prepared)
    for fake_quant in fake_quantizers(prepared):
        assert not bool(fake_quant.observer_enabled.item())

    for node in weighted_nodes(prepared):
        weight_fq = _fake_quant_module(prepared, node.args[1])
        assert weight_fq is not None
        minimum_scale = _minimum_weight_scale(prepared, node.args[1]).reshape(-1)
        assert torch.all(weight_fq.scale.reshape(-1) >= minimum_scale)
