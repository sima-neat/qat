#**************************************************************************
#||                        SiMa.ai CONFIDENTIAL                          ||
#||   Unpublished Copyright (c) 2024 SiMa.ai, All Rights Reserved.       ||
#**************************************************************************
# NOTICE: All information contained herein remains the property of SiMa.ai.
# The source code is confidential and proprietary.
#**************************************************************************
"""ONNX export helpers for the Dynamo-free FX QAT backend."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import onnx
import torch
from onnx import ModelProto, NodeProto, numpy_helper
from torch import nn
from torch.ao.quantization import FakeQuantize


_QDQ_OPS = {"QuantizeLinear", "DequantizeLinear"}
_WEIGHT_OPS = {"Conv", "Gemm", "MatMul"}


def prepare_export_copy(model: nn.Module) -> Tuple[nn.Module, int]:
    """Create an ONNX-only model copy accepted by PyTorch's legacy exporter.

    SiMa trains and finalizes weights with the symmetric range [-127, 127].
    PyTorch's standard ONNX symbolic rejects that range even though ONNX can
    represent its values. The disposable copy traces those weight fake-quant
    calls with -128; normalize_qdq_model then folds the real [-127, 127]
    values into INT8 initializers before the artifact is allowed to escape.
    """

    shadow = copy.deepcopy(model).cpu().eval()
    shadowed = 0
    for fake_quant in shadow.modules():
        if not isinstance(fake_quant, FakeQuantize):
            continue
        fake_quant.disable_observer()
        if not fake_quant.is_per_channel:
            continue

        observer = fake_quant.activation_post_process
        bounds = (observer.quant_min, observer.quant_max, fake_quant.ch_axis)
        if bounds != (-127, 127, 0):
            raise RuntimeError(
                "Unsupported per-channel weight fake-quant configuration: "
                f"expected (-127, 127, axis=0), found {bounds}."
            )
        observer.quant_min = -128
        shadowed += 1
    return shadow, shadowed


def _axis_of(node: NodeProto) -> Optional[int]:
    return next(
        (attribute.i for attribute in node.attribute if attribute.name == "axis"),
        None,
    )


def _value_maps(model: ModelProto):
    producers = {
        output: node
        for node in model.graph.node
        for output in node.output
    }
    consumers: Dict[str, list[NodeProto]] = {}
    for node in model.graph.node:
        for value in node.input:
            consumers.setdefault(value, []).append(node)
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    constants = {}
    for node in model.graph.node:
        if node.op_type != "Constant" or len(node.output) != 1:
            continue
        value = next(
            (attribute.t for attribute in node.attribute if attribute.name == "value"),
            None,
        )
        if value is not None:
            constants[node.output[0]] = value
    return producers, consumers, initializers, constants


def _identity_root(name: str, producers: Dict[str, NodeProto]) -> str:
    seen = set()
    while name not in seen:
        seen.add(name)
        node = producers.get(name)
        if node is None or node.op_type != "Identity" or len(node.input) != 1:
            break
        name = node.input[0]
    return name


def _feeds_weight_input(value: str, consumers: Dict[str, list[NodeProto]]) -> bool:
    pending = [value]
    seen = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        for node in consumers.get(current, []):
            if (
                node.op_type in _WEIGHT_OPS
                and len(node.input) > 1
                and node.input[1] == current
            ):
                return True
            if node.op_type in {"Identity", "Transpose"}:
                pending.extend(node.output)
    return False


def _fold_per_channel_weights(model: ModelProto, expected_minimum: int) -> int:
    producers, consumers, initializers, _ = _value_maps(model)
    remove = []
    add = []

    for quantize in list(model.graph.node):
        if quantize.op_type != "QuantizeLinear" or _axis_of(quantize) != 0:
            continue

        uses = consumers.get(quantize.output[0], [])
        if len(uses) != 1 or uses[0].op_type != "DequantizeLinear":
            raise RuntimeError(
                "Per-channel QuantizeLinear is not paired with one DequantizeLinear."
            )
        dequantize = uses[0]
        if _axis_of(dequantize) != 0 or not _feeds_weight_input(
            dequantize.output[0], consumers
        ):
            raise RuntimeError("Per-channel Q/DQ pair is not a supported weight edge.")

        qparams = [
            _identity_root(value, producers)
            for value in quantize.input[1:3]
        ]
        dq_qparams = [
            _identity_root(value, producers)
            for value in dequantize.input[1:3]
        ]
        if qparams != dq_qparams:
            raise RuntimeError(
                "Weight QuantizeLinear and DequantizeLinear use different qparams."
            )

        weight_name = _identity_root(quantize.input[0], producers)
        scale_name, zero_point_name = qparams
        names = (weight_name, scale_name, zero_point_name)
        if not all(name in initializers for name in names):
            raise RuntimeError("Weight Q/DQ inputs must resolve to ONNX initializers.")

        weight = numpy_helper.to_array(initializers[weight_name])
        scale = numpy_helper.to_array(initializers[scale_name])
        zero_point = numpy_helper.to_array(initializers[zero_point_name])
        if (
            scale.ndim != 1
            or zero_point.ndim != 1
            or len(scale) != len(zero_point)
            or len(scale) != weight.shape[0]
        ):
            raise RuntimeError("Invalid per-channel weight qparam shape.")
        if not np.isfinite(weight).all():
            raise RuntimeError("Non-finite value found in an ONNX weight initializer.")
        if not np.isfinite(scale).all() or np.any(scale <= 0):
            raise RuntimeError("Per-channel weight scales must be finite and positive.")
        if (
            not np.isfinite(zero_point).all()
            or not np.equal(zero_point, np.rint(zero_point)).all()
        ):
            raise RuntimeError("Per-channel zero points must be finite integers.")

        broadcast_shape = [1] * weight.ndim
        broadcast_shape[0] = len(scale)
        codes = np.clip(
            np.rint(
                weight / scale.reshape(broadcast_shape)
                + zero_point.reshape(broadcast_shape)
            ),
            -127,
            127,
        ).astype(np.int8)
        add.append(numpy_helper.from_array(codes, name=quantize.output[0]))
        remove.append(quantize)

    for quantize in remove:
        model.graph.node.remove(quantize)
    model.graph.initializer.extend(add)

    if len(remove) < expected_minimum:
        raise RuntimeError(
            "Not every temporarily widened per-channel fake quant was folded: "
            f"expected at least {expected_minimum}, folded {len(remove)}."
        )
    if any(
        node.op_type == "QuantizeLinear" and _axis_of(node) is not None
        for node in model.graph.node
    ):
        raise RuntimeError("An unfrozen per-channel QuantizeLinear remains in ONNX.")
    return len(remove)


def _scalarize_per_tensor_qparams(model: ModelProto) -> None:
    producers, _, initializers, constants = _value_maps(model)
    scalar_roots = set()
    axis_roots = set()
    for node in model.graph.node:
        if node.op_type not in _QDQ_OPS:
            continue
        roots = {
            _identity_root(value, producers)
            for value in node.input[1:3]
        }
        if _axis_of(node) is None:
            scalar_roots.update(roots)
        else:
            axis_roots.update(roots)

    overlap = scalar_roots & axis_roots
    if overlap:
        raise RuntimeError(
            f"Qparams are shared by per-tensor and per-channel edges: {overlap}."
        )

    tensors = {**constants, **initializers}
    for name in scalar_roots:
        tensor = tensors.get(name)
        if tensor is None:
            raise RuntimeError(f"Per-tensor qparam {name} is not constant.")
        element_count = int(np.prod(list(tensor.dims) or [1]))
        if list(tensor.dims) not in ([], [1]) or element_count != 1:
            raise RuntimeError(f"Per-tensor qparam {name} is not scalar-compatible.")
        tensor.ClearField("dims")


def _remove_redundant_requantization(model: ModelProto) -> int:
    """Collapse exact Q-DQ-Q round trips which use identical qparams."""

    removed_pairs = 0
    while True:
        producers, consumers, initializers, constants = _value_maps(model)
        tensors = {**constants, **initializers}
        changed = False

        for quantize in list(model.graph.node):
            if quantize.op_type != "QuantizeLinear":
                continue
            dequantize = producers.get(quantize.input[0])
            if dequantize is None or dequantize.op_type != "DequantizeLinear":
                continue
            if _axis_of(quantize) != _axis_of(dequantize):
                raise RuntimeError(
                    "Direct DQ-to-Q edge changes per-tensor/per-channel mode."
                )

            dq_qparams = [
                _identity_root(value, producers)
                for value in dequantize.input[1:3]
            ]
            q_qparams = [
                _identity_root(value, producers)
                for value in quantize.input[1:3]
            ]
            equivalent = True
            for left_name, right_name in zip(dq_qparams, q_qparams):
                left = tensors.get(left_name)
                right = tensors.get(right_name)
                if left is None or right is None:
                    equivalent = False
                    break
                if not np.array_equal(
                    numpy_helper.to_array(left),
                    numpy_helper.to_array(right),
                ):
                    equivalent = False
                    break
            if not equivalent:
                raise RuntimeError(
                    "Direct DQ-to-Q requantization uses different qparams."
                )

            if any(
                output.name == quantize.output[0]
                for output in model.graph.output
            ):
                raise RuntimeError("Cannot collapse a graph-output QuantizeLinear.")

            upstream_codes = dequantize.input[0]
            for consumer in consumers.get(quantize.output[0], []):
                for index, value in enumerate(consumer.input):
                    if value == quantize.output[0]:
                        consumer.input[index] = upstream_codes

            model.graph.node.remove(quantize)
            if consumers.get(dequantize.output[0], []) == [quantize]:
                model.graph.node.remove(dequantize)
            removed_pairs += 1
            changed = True
            break

        if not changed:
            return removed_pairs


def _remove_unused_initializers(model: ModelProto) -> None:
    used = {
        value
        for node in model.graph.node
        for value in node.input
    }
    used.update(value.name for value in model.graph.input)
    used.update(value.name for value in model.graph.output)
    unused = [
        initializer
        for initializer in model.graph.initializer
        if initializer.name not in used
    ]
    for initializer in unused:
        model.graph.initializer.remove(initializer)


def _validate_qdq_topology(model: ModelProto) -> None:
    producers, _, initializers, constants = _value_maps(model)
    tensors = {**constants, **initializers}

    for node in model.graph.node:
        if node.op_type == "QuantizeLinear" and _axis_of(node) is not None:
            raise RuntimeError("Per-channel QuantizeLinear must be folded into INT8.")
        if node.op_type not in _QDQ_OPS:
            continue

        if _axis_of(node) is None:
            for value in node.input[1:3]:
                tensor = tensors.get(_identity_root(value, producers))
                if tensor is None or list(tensor.dims) != []:
                    raise RuntimeError("Per-tensor Q/DQ qparams must be scalar.")
        elif node.op_type == "DequantizeLinear":
            weight_name = _identity_root(node.input[0], producers)
            weight = initializers.get(weight_name)
            if weight is None or weight.data_type != onnx.TensorProto.INT8:
                raise RuntimeError(
                    "Per-channel weight DQ must consume an INT8 initializer."
                )

        source = producers.get(_identity_root(node.input[0], producers))
        if node.op_type == "QuantizeLinear" and source is not None:
            if source.op_type == "DequantizeLinear":
                raise RuntimeError("Redundant direct DequantizeLinear to QuantizeLinear edge.")


def normalize_qdq_model(
    output_file: Union[str, Path],
    shadowed_weight_fake_quants: int,
) -> ModelProto:
    """Finalize standard Q/DQ topology and validate the serialized ONNX model."""

    output_path = str(output_file)
    model = onnx.load(output_path)
    _remove_redundant_requantization(model)
    _fold_per_channel_weights(model, shadowed_weight_fake_quants)
    _scalarize_per_tensor_qparams(model)
    _remove_unused_initializers(model)
    _validate_qdq_topology(model)
    onnx.checker.check_model(model)
    try:
        model = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    except Exception as error:
        raise RuntimeError("ONNX shape inference failed after Q/DQ normalization.") from error
    onnx.checker.check_model(model)
    onnx.save(model, output_path)

    reloaded = onnx.load(output_path)
    onnx.checker.check_model(reloaded)
    _validate_qdq_topology(reloaded)
    return reloaded
