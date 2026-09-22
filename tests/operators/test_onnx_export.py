"""Representative ONNX QDQ and ONNX Runtime parity tests by operator family."""

import math
import warnings

import numpy as np
import onnx
import onnxruntime
import pytest
import torch

from sima_qat import sima_export_onnx, sima_finalize_qat_model, sima_freeze_qat

from .cases import ONNX_CASES, case_ids
from .helpers import fake_quantizers, prepare_case


pytestmark = pytest.mark.regression

_ACCUMULATION_OUTLIER_FRACTION = 0.001
_ACCUMULATION_OUTLIER_QUANTA = 16


def _largest_activation_quantum(model) -> float:
    scales = [
        float(module.scale.detach().abs().max())
        for module in fake_quantizers(model)
        if module.qscheme in (torch.per_tensor_affine, torch.per_tensor_symmetric)
    ]
    return max(scales, default=0.0)


def _assert_runtime_parity(
    actual: np.ndarray,
    expected: np.ndarray,
    output_quantum: float,
    *,
    weighted: bool,
) -> None:
    base_atol = 2 * output_quantum + 1e-6
    if not weighted:
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1e-5,
            atol=base_atol,
        )
        return

    # A value exactly on a QDQ boundary can round by one code in PyTorch and
    # the other direction in ONNX Runtime. Weighted accumulation can amplify
    # that isolated operand difference into several output code points. Keep
    # the ordinary two-quantum bound for virtually the whole tensor, then cap
    # both the count and magnitude of those isolated accumulation outliers.
    close = np.isclose(actual, expected, rtol=1e-5, atol=base_atol)
    outlier_count = int(np.count_nonzero(~close))
    max_outliers = max(
        1,
        math.ceil(actual.size * _ACCUMULATION_OUTLIER_FRACTION),
    )
    assert outlier_count <= max_outliers, (
        f"{outlier_count}/{actual.size} weighted outputs exceed the "
        f"two-quantum tolerance; allowed {max_outliers}"
    )
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-5,
        atol=_ACCUMULATION_OUTLIER_QUANTA * output_quantum + 1e-6,
    )


@pytest.mark.parametrize("case", ONNX_CASES, ids=case_ids)
def test_operator_family_exports_standard_qdq_and_matches_onnxruntime(case, tmp_path) -> None:
    prepared, inputs = prepare_case(case)
    sima_freeze_qat(prepared)
    output_quantum = _largest_activation_quantum(prepared)
    finalized = sima_finalize_qat_model(prepared)

    output_path = tmp_path / f"{case.name}.onnx"
    if case.name == "instance_norm":
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sima_export_onnx(finalized, inputs, str(output_path), device="cpu")
        assert not any("instance_norm' is set to train=True" in str(item.message) for item in caught)
    else:
        sima_export_onnx(finalized, inputs, str(output_path), device="cpu")

    exported = onnx.load(output_path)
    onnx.checker.check_model(exported)
    operator_types = {node.op_type for node in exported.graph.node}
    assert "QuantizeLinear" in operator_types
    assert "DequantizeLinear" in operator_types

    with torch.no_grad():
        torch_output = finalized(*inputs).detach().cpu().numpy()
    session = onnxruntime.InferenceSession(
        str(output_path),
        providers=["CPUExecutionProvider"],
    )
    feed = {
        onnx_input.name: value.detach().cpu().numpy()
        for onnx_input, value in zip(session.get_inputs(), inputs)
    }
    onnx_output = session.run(None, feed)[0]

    assert np.isfinite(onnx_output).all()
    _assert_runtime_parity(
        onnx_output,
        torch_output,
        output_quantum,
        weighted=case.weighted,
    )
