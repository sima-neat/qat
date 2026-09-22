"""Representative ONNX QDQ and ONNX Runtime parity tests by operator family."""

import warnings

import numpy as np
import onnx
import onnxruntime
import pytest
import torch
from torch.ao.quantization.fake_quantize import FakeQuantizeBase
from torch.utils._pytree import tree_leaves

from sima_qat import (
    sima_export_onnx,
    sima_finalize_qat_model,
    sima_freeze_qat,
    sima_prepare_qat_model,
)

from .cases import ONNX_CASES, case_ids
from .helpers import prepare_case


pytestmark = pytest.mark.regression


def _output_quantum(model) -> float:
    output = next(node for node in model.graph.nodes if node.op == "output")
    leaves = tree_leaves(output.args[0])
    assert len(leaves) == 1, "Each output needs its own parity budget"
    node = leaves[0]
    # Selection preserves the grid, unlike arbitrary arithmetic upstream.
    while node.op == "call_function" and node.target == torch.ops.aten.select.int:
        node = node.args[0]
    if node.op == "call_module":
        module = model.get_submodule(node.target)
        assert isinstance(module, FakeQuantizeBase)
        assert module.scale.numel() == 1
        return float(module.scale.detach().item())
    assert not node.meta["val"].is_floating_point(), "Output has no known quantization grid"
    return 0.0


def _assert_runtime_parity(
    actual: np.ndarray,
    expected: np.ndarray,
    output_quantum: float,
    *,
    output_quanta: int = 1,
) -> None:
    if np.issubdtype(expected.dtype, np.integer):
        np.testing.assert_array_equal(actual, expected)
        return
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-5,
        atol=output_quanta * output_quantum + 1e-6,
    )


@pytest.mark.parametrize("case", ONNX_CASES, ids=case_ids)
def test_operator_family_exports_standard_qdq_and_matches_onnxruntime(case, tmp_path) -> None:
    prepared, inputs = prepare_case(case)
    sima_freeze_qat(prepared)
    output_quantum = _output_quantum(prepared)
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
    # Test exported QDQ semantics, not ORT's fused integer kernels, whose
    # operand rounding can accumulate differently from the explicit graph.
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = onnxruntime.InferenceSession(
        str(output_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    feed = {
        onnx_input.name: value.detach().cpu().numpy()
        for onnx_input, value in zip(session.get_inputs(), inputs)
    }
    onnx_output = session.run(None, feed)[0]

    assert np.isfinite(onnx_output).all()
    # Both average-pool fixtures contain input-grid rounding ties. Float
    # reduction order can cross those ties; the following convolution amplifies
    # a one-code operand difference to two output codes. Keep this budget local
    # to these fixtures, using their output scale, not an intermediate scale.
    _assert_runtime_parity(
        onnx_output,
        torch_output,
        output_quantum,
        output_quanta=2 if case.name in {"adaptive_avg_pool2d", "global_average_pool"} else 1,
    )


def test_parity_rejects_inverted_probabilities_despite_large_input_scale() -> None:
    inputs = (torch.tensor([[-100.0, 100.0]]),)
    prepared = sima_prepare_qat_model(torch.nn.Softmax(dim=-1), inputs, "cpu")
    prepared(*inputs)
    sima_freeze_qat(prepared)
    quantum = _output_quantum(prepared)
    assert quantum == pytest.approx(1 / 255)
    for quanta in (1, 2):
        with pytest.raises(AssertionError):
            _assert_runtime_parity(
                np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]]), quantum,
                output_quanta=quanta,
            )


def test_parity_bounds_pooling_roundoff_to_two_output_codes() -> None:
    quantum = 0.005
    expected = np.zeros(4, dtype=np.float32)
    _assert_runtime_parity(
        expected + 2 * quantum, expected, quantum, output_quanta=2,
    )
    with pytest.raises(AssertionError):
        _assert_runtime_parity(
            expected + 3 * quantum, expected, quantum, output_quanta=2,
        )
