"""Extract stable YOLO26 one-to-one raw heads from a finalized QDQ ONNX graph."""

from __future__ import annotations

import tempfile
from collections import defaultdict
from pathlib import Path

import onnx
from onnx import helper, shape_inference, utils


def _head_endpoint(
    model: onnx.ModelProto,
    consumers: dict[str, list[onnx.NodeProto]],
    bias_name: str,
) -> str:
    matching = [
        node
        for node in model.graph.node
        if node.op_type == "Conv" and bias_name in node.input
    ]
    if len(matching) != 1:
        raise RuntimeError(f"Expected one Conv consuming {bias_name}, found {len(matching)}")
    convolution = matching[0]
    quantizers = [
        node
        for node in consumers[convolution.output[0]]
        if node.op_type == "QuantizeLinear"
    ]
    if len(quantizers) != 1:
        raise RuntimeError(f"Expected one output QuantizeLinear after {convolution.name}")
    dequantizers = [
        node
        for node in consumers[quantizers[0].output[0]]
        if node.op_type == "DequantizeLinear"
    ]
    if len(dequantizers) != 1:
        raise RuntimeError(f"Expected one output DequantizeLinear after {convolution.name}")
    return dequantizers[0].output[0]


def extract_raw_one2one_heads(source: str | Path, destination: str | Path) -> None:
    """Prune a full training-output ONNX to six BoxDecode-compatible raw heads."""

    source = Path(source)
    destination = Path(destination)
    model = onnx.load(source)
    consumers = defaultdict(list)
    for node in model.graph.node:
        for value in node.input:
            consumers[value].append(node)

    output_names = []
    for role, module_name in (("bbox", "one2one_cv2"), ("class_logit", "one2one_cv3")):
        for level in range(3):
            endpoint = _head_endpoint(
                model,
                consumers,
                f"model.23.{module_name}.{level}.2.bias",
            )
            output_name = f"{role}_{level}"
            model.graph.node.append(
                helper.make_node(
                    "Identity",
                    [endpoint],
                    [output_name],
                    name=f"/sima_yolo26_heads/{output_name}",
                )
            )
            output_names.append(output_name)

    model = shape_inference.infer_shapes(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="yolo26-head-export-") as directory:
        intermediate = Path(directory) / "named.onnx"
        onnx.save(model, intermediate)
        utils.extract_model(
            str(intermediate),
            str(destination),
            [value.name for value in model.graph.input],
            output_names,
        )
    extracted = onnx.load(destination)
    onnx.checker.check_model(extracted)
    expected_channels = (4, 4, 4, 80, 80, 80)
    actual_channels = tuple(
        output.type.tensor_type.shape.dim[1].dim_value for output in extracted.graph.output
    )
    if actual_channels != expected_channels:
        raise RuntimeError(
            f"Unexpected raw-head channels: expected {expected_channels}, found {actual_channels}"
        )
