#!/usr/bin/env python3
"""Functional acceptance test for QAT installed into Model Compiler."""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import inspect
from pathlib import Path
import sys
import tempfile

import numpy as np
import onnx
import onnxruntime
import torch
from onnx import numpy_helper

from sima_qat import qat_api
from sima_qat import (
    __version__,
    sima_export_onnx,
    sima_finalize_qat_model,
    sima_prepare_qat_model,
)


class SmokeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU(),
        )
        self.left = torch.nn.Conv2d(4, 4, kernel_size=1)
        self.right = torch.nn.Conv2d(4, 4, kernel_size=1)
        self.dropout = torch.nn.Dropout(0.2)
        self.classifier = torch.nn.Linear(8 * 8 * 8, 5)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        stem = self.stem(inputs)
        left = self.left(stem)
        residual = left + self.right(stem)
        merged = torch.cat((left, residual), dim=1)
        merged = self.dropout(merged)
        return self.classifier(torch.flatten(merged, 1))


def validate_prefix(expected_prefix: str) -> None:
    if not expected_prefix:
        return
    actual = Path(sys.prefix).resolve()
    expected = Path(expected_prefix).resolve()
    if actual != expected:
        raise RuntimeError(
            f"QAT smoke test uses Python prefix {actual}; expected {expected}."
        )


def validate_no_forbidden_backend_calls() -> None:
    source = inspect.getsource(qat_api)
    forbidden = (
        "torch._dynamo",
        "capture_pre_autograd_graph",
        "export_for_training",
        "prepare_qat_pt2e",
        "convert_pt2e",
    )
    present = [name for name in forbidden if name in source]
    if present:
        raise RuntimeError(f"Forbidden Dynamo/PT2E API references found: {present}")


def validate_onnx(model: onnx.ModelProto) -> None:
    onnx.checker.check_model(model)
    node_types = [node.op_type for node in model.graph.node]
    if "QuantizeLinear" not in node_types or "DequantizeLinear" not in node_types:
        raise RuntimeError("Exported ONNX model has no standard Q/DQ topology.")
    if "Dropout" in node_types or "BatchNormalization" in node_types:
        raise RuntimeError("Dropout or fused BatchNormalization escaped finalization.")
    if any(node.domain not in ("", "ai.onnx") for node in model.graph.node):
        raise RuntimeError("Exported ONNX model contains a custom operator domain.")

    initializers = {value.name: value for value in model.graph.initializer}
    weight_dq = 0
    for node in model.graph.node:
        if node.op_type not in ("QuantizeLinear", "DequantizeLinear"):
            continue
        axis = next(
            (attribute.i for attribute in node.attribute if attribute.name == "axis"),
            None,
        )
        if axis is None:
            for name in node.input[1:3]:
                value = initializers.get(name)
                if value is not None and list(value.dims) != []:
                    raise RuntimeError("Per-tensor ONNX qparams are not scalar.")
        elif node.op_type == "QuantizeLinear":
            raise RuntimeError("Per-channel weight QuantizeLinear was not folded.")
        elif axis == 0:
            value = initializers.get(node.input[0])
            if value is None or value.data_type != onnx.TensorProto.INT8:
                raise RuntimeError("Weight DQ does not consume an INT8 initializer.")
            codes = numpy_helper.to_array(value)
            if codes.min() < -127 or codes.max() > 127:
                raise RuntimeError("Frozen weight is outside the SiMa INT8 range.")
            weight_dq += 1
    if weight_dq == 0:
        raise RuntimeError("No frozen per-channel weight edge was exported.")


def run_smoke_test(work_dir: Path) -> None:
    validate_no_forbidden_backend_calls()
    torch.manual_seed(7)
    inputs = (torch.randn(2, 3, 8, 8),)
    source = SmokeModel().train()
    source_parameter_ids = {id(parameter) for parameter in source.parameters()}
    optimizer = torch.optim.SGD(source.parameters(), lr=0.01)

    prepared = sima_prepare_qat_model(source, inputs, "cpu")
    if {id(parameter) for parameter in prepared.parameters()} != source_parameter_ids:
        raise RuntimeError("QAT preparation broke pre-existing optimizer parameters.")
    if not prepared.training or prepared.qat_stage != "scaffold":
        raise RuntimeError("Prepared model did not enter scaffold training state.")
    if any(
        isinstance(module, torch.nn.modules.dropout._DropoutNd)
        for module in prepared.modules()
    ):
        raise RuntimeError("Dropout was not removed before QAT observer insertion.")

    optimizer.zero_grad()
    loss = prepared(*inputs).square().mean()
    loss.backward()
    optimizer.step()

    checkpoint = copy.deepcopy(prepared.state_dict())
    reload_skeleton = sima_prepare_qat_model(SmokeModel(), inputs, "cpu")
    reload_skeleton.load_state_dict(checkpoint, strict=True)

    parent = torch.nn.Module()
    parent.add_module("qat_model", reload_skeleton)
    nested_checkpoint = copy.deepcopy(parent.state_dict())
    parent.load_state_dict(nested_checkpoint, strict=True)
    nested_wrong_schema = copy.deepcopy(nested_checkpoint)
    nested_wrong_schema["qat_model.qat_backend_version"].fill_(99)
    nested_legacy = copy.deepcopy(nested_checkpoint)
    nested_legacy.pop("qat_model.qat_backend_version")
    for state, strict, marker in (
        (nested_wrong_schema, True, "Unsupported QAT checkpoint schema"),
        (nested_legacy, False, "Legacy PT2E checkpoints"),
    ):
        try:
            parent.load_state_dict(state, strict=strict)
        except RuntimeError as error:
            if marker not in str(error):
                raise RuntimeError("Unexpected nested checkpoint error.") from error
        else:
            raise RuntimeError("Nested checkpoint bypassed QAT schema validation.")

    finalized = sima_finalize_qat_model(prepared)
    if finalized.training or finalized.qat_stage != "fq":
        raise RuntimeError("Finalized model is not in fake-quant inference state.")
    try:
        finalized.train(True)
    except RuntimeError:
        pass
    else:
        raise RuntimeError("Finalized model incorrectly accepted training mode.")

    with torch.no_grad():
        torch_output = finalized(*inputs).cpu().numpy()

    onnx_path = work_dir / "qat-smoke.onnx"
    sima_export_onnx(
        finalized,
        inputs,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        device="cpu",
    )
    onnx_model = onnx.load(onnx_path)
    validate_onnx(onnx_model)

    strict_options = onnxruntime.SessionOptions()
    strict_options.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    strict_session = onnxruntime.InferenceSession(
        str(onnx_path),
        sess_options=strict_options,
        providers=["CPUExecutionProvider"],
    )
    strict_output = strict_session.run(None, {"input": inputs[0].numpy()})[0]
    np.testing.assert_allclose(strict_output, torch_output, rtol=1e-6, atol=1e-6)

    optimized_session = onnxruntime.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    optimized_output = optimized_session.run(None, {"input": inputs[0].numpy()})[0]
    if optimized_output.shape != torch_output.shape:
        raise RuntimeError("Optimized ONNX Runtime output shape is incorrect.")
    if not np.isfinite(optimized_output).all():
        raise RuntimeError("Optimized ONNX Runtime output contains non-finite values.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run QAT inside an installed Model Compiler environment."
    )
    parser.add_argument(
        "--expected-prefix",
        default="",
        help="Require this virtual-environment prefix.",
    )
    args = parser.parse_args()

    validate_prefix(args.expected_prefix)
    installed_version = importlib.metadata.version("sima-qat")
    if __version__ != installed_version:
        raise RuntimeError(
            f"sima_qat.__version__ is {__version__}, "
            f"but wheel metadata is {installed_version}."
        )
    with tempfile.TemporaryDirectory(prefix="qat-smoke-") as temporary:
        run_smoke_test(Path(temporary))

    print(
        "QAT functional smoke test passed "
        f"(python={sys.version.split()[0]}, "
        f"torch={importlib.metadata.version('torch')}, "
        f"sima-qat={installed_version})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
