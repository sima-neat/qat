"""Representative ONNX QDQ and ONNX Runtime parity tests by operator family."""

import numpy as np
import onnx
import onnxruntime
import pytest
import torch

from sima_qat import sima_export_onnx, sima_finalize_qat_model, sima_freeze_qat

from .cases import ONNX_CASES, case_ids
from .helpers import fake_quantizers, prepare_case


pytestmark = pytest.mark.regression


def _largest_activation_quantum(model) -> float:
    scales = [
        float(module.scale.detach().abs().max())
        for module in fake_quantizers(model)
        if module.qscheme in (torch.per_tensor_affine, torch.per_tensor_symmetric)
    ]
    return max(scales, default=0.0)


@pytest.mark.parametrize("case", ONNX_CASES, ids=case_ids)
def test_operator_family_exports_standard_qdq_and_matches_onnxruntime(case, tmp_path) -> None:
    prepared, inputs = prepare_case(case)
    sima_freeze_qat(prepared)
    output_quantum = _largest_activation_quantum(prepared)
    finalized = sima_finalize_qat_model(prepared)

    output_path = tmp_path / f"{case.name}.onnx"
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
    np.testing.assert_allclose(
        onnx_output,
        torch_output,
        rtol=1e-5,
        # PyTorch and ONNX Runtime may round an intermediate QDQ edge in
        # opposite directions. Two output quanta bounds that legitimate
        # difference while remaining tied to the exported quantization grid.
        atol=2 * output_quantum + 1e-6,
    )
