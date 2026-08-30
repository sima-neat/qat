#**************************************************************************
#||                        SiMa.ai CONFIDENTIAL                          ||
#||   Unpublished Copyright (c) 2024 SiMa.ai, All Rights Reserved.       ||
#**************************************************************************
# NOTICE:  All information contained herein is, and remains the property of
# SiMa.ai. The intellectual and technical concepts contained herein are
# proprietary to SiMa and may be covered by U.S. and Foreign Patents,
# patents in process, and are protected by trade secret or copyright law.
#
# Dissemination of this information or reproduction of this material is
# strictly forbidden unless prior written permission is obtained from
# SiMa.ai.  Access to the source code contained herein is hereby forbidden
# to anyone except current SiMa.ai employees, managers or contractors who
# have executed Confidentiality and Non-disclosure agreements explicitly
# covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes information
# that is confidential and/or proprietary, and is a trade secret, of SiMa.ai.
#
# ANY REPRODUCTION, MODIFICATION, DISTRIBUTION, PUBLIC PERFORMANCE, OR PUBLIC
# DISPLAY OF OR THROUGH USE OF THIS SOURCE CODE WITHOUT THE EXPRESS WRITTEN
# CONSENT OF SiMa.ai IS STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE
# LAWS AND INTERNATIONAL TREATIES. THE RECEIPT OR POSSESSION OF THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS TO
# REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE, USE, OR
# SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
#**************************************************************************
import functools

import torch
import torch._C._onnx as _C_onnx
import torch.nn.modules.utils
import torch.onnx
from torch.onnx import (
    _type_utils,
    errors,
    symbolic_helper,
)


from torch.onnx._internal import jit_utils, registration
try:
    from torch.onnx._internal import _beartype
except ImportError:
    # torch 2.8 removed the optional runtime type-checking wrapper while
    # retaining the legacy symbolic registry used by this module.
    class _NoOpBearType:
        @staticmethod
        def beartype(function):
            return function

    _beartype = _NoOpBearType()


# Q/DQ operators in ONNX had a major revision at Opset 13. Opset 19 was the next revision,
# and those changes are not relevant for Sima.
_onnx_symbolic = functools.partial(registration.onnx_symbolic, opset=13)



@_onnx_symbolic("quantized_decomposed::quantize_per_tensor")
@symbolic_helper.parse_args("v", "v", "v", "i", "i", "v")
@_beartype.beartype
def fake_quantize_per_tensor_affine(
    g: jit_utils.GraphContext,
    inputs,
    scale,
    zero_point,
    quant_min=-128,
    quant_max=127,
    dtype=torch.dtype,
):
    # NOTE: (0, 127) is allowed as special case. PyTorch restricts activations to be in the range (0, 127).
    #   https://github.com/pytorch/pytorch/blob/b34b192d6b97325c9f78e5995c48c8498ede34bd/torch/ao/quantization/observer.py#L1422
    # ONNX carries the storage dtype, scale, and zero point but not an
    # explicit used-code subrange.  Symmetric SiMa weights intentionally leave
    # INT8 code -128 unused, so [-127, 127] is represented by the same signed
    # QuantizeLinear type and remains clipped by the trained PyTorch graph.
    if (quant_min, quant_max) not in [(0, 255), (-128, 127), (-127, 127), (0, 127)]:
        raise errors.SymbolicValueError(
            "Unsupported ONNX quantization code range. "
            f"Got ({quant_min}, {quant_max})",
            inputs,
        )
    if quant_min == 0:
        zero_point = g.op("Cast", zero_point, to_i=_C_onnx.TensorProtoDataType.UINT8)
    else:
        zero_point = g.op("Cast", zero_point, to_i=_C_onnx.TensorProtoDataType.INT8)
    # Assert-based for now; this will catch logical graph problems.
    input_type = _type_utils.JitScalarType.from_value(inputs, _type_utils.JitScalarType.UNDEFINED)
    assert input_type == _type_utils.JitScalarType.FLOAT
    # This is apparently important, because scale can come is as a double (!?) on this function call.
    if (
        _type_utils.JitScalarType.from_value(scale, _type_utils.JitScalarType.UNDEFINED)
        != _type_utils.JitScalarType.FLOAT
    ):
        scale = g.op("Cast", scale, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    quantized = g.op("QuantizeLinear", inputs, scale, zero_point)
    assert _type_utils.JitScalarType.from_value(quantized, _type_utils.JitScalarType.UNDEFINED) == _type_utils.JitScalarType.INT8
    return quantized


@_onnx_symbolic("quantized_decomposed::dequantize_per_tensor")
@symbolic_helper.parse_args("v", "v", "v", "i", "i", "v", "v")
@_beartype.beartype
def fake_dequantize_per_tensor_affine(
    g: jit_utils.GraphContext,
    inputs,
    scale,
    zero_point,
    quant_min=-128,
    quant_max=127,
    dtype=torch.dtype,
    out_dtype=None,
):
    del out_dtype
    if quant_min == 0:
        zero_point = g.op("Cast", zero_point, to_i=_C_onnx.TensorProtoDataType.UINT8)
    else:
        zero_point = g.op("Cast", zero_point, to_i=_C_onnx.TensorProtoDataType.INT8)
    input_type = _type_utils.JitScalarType.from_value(inputs, _type_utils.JitScalarType.UNDEFINED)
    assert input_type == _type_utils.JitScalarType.INT8
    quantized = inputs
    # This is apparently important, because scale can come is as a double (!?) on this function call.
    if (
        _type_utils.JitScalarType.from_value(scale, _type_utils.JitScalarType.UNDEFINED)
        != _type_utils.JitScalarType.FLOAT
    ):
        scale = g.op("Cast", scale, to_i=_C_onnx.TensorProtoDataType.FLOAT)    
    dq = g.op("DequantizeLinear", quantized, scale, zero_point)
    assert _type_utils.JitScalarType.from_value(dq, _type_utils.JitScalarType.UNDEFINED) == _type_utils.JitScalarType.FLOAT
    return dq


@_onnx_symbolic("quantized_decomposed::dequantize_per_channel")
@symbolic_helper.parse_args("v", "v", "v", "i", "i", "i", "v", "v")
@_beartype.beartype
def fake_quantize_per_channel_affine(
    g: jit_utils.GraphContext,
    inputs,
    scales,
    zero_points,
    axis,
    quant_min=-128,
    quant_max=127,
    dtype=torch.dtype,
    out_dtype=None,
):
    del out_dtype
    # NOTE: (0, 127) is allowed as special case. PyTorch restricts activations to be in the range (0, 127).
    #   https://github.com/pytorch/pytorch/blob/b34b192d6b97325c9f78e5995c48c8498ede34bd/torch/ao/quantization/observer.py#L1422
    # if (quant_min, quant_max) not in [(0, 255), (-128, 127), (0, 127)]:
    #     raise errors.SymbolicValueError(
    #         "For (quant_min, quant_max), ONNX allows only (0, 127), (0, 255) and (-128, 127). "
    #         f"Got ({quant_min}, {quant_max})",
    #         inputs,
    #     )
    # ONNX defines zero_point to be int8 or uint8
    if quant_min == 0:
        zero_points = g.op("Cast", zero_points, to_i=_C_onnx.TensorProtoDataType.UINT8)
    else:
        zero_points = g.op("Cast", zero_points, to_i=_C_onnx.TensorProtoDataType.INT8)
    input_type = _type_utils.JitScalarType.from_value(inputs, _type_utils.JitScalarType.UNDEFINED)
    assert input_type == _type_utils.JitScalarType.INT8
    quantized = inputs
    # axis_cast = g.op("Cast", axis, to_i=_C_onnx.TensorProtoDataType.INT32)
    # This is apparently important, because scale can come is as a double (!?) on this function call.
    if (
        _type_utils.JitScalarType.from_value(scales, _type_utils.JitScalarType.UNDEFINED)
        != _type_utils.JitScalarType.FLOAT
    ):
        scales = g.op("Cast", scales, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    dq = g.op("DequantizeLinear", quantized, scales, zero_points, axis_i=axis)
    assert _type_utils.JitScalarType.from_value(dq, _type_utils.JitScalarType.UNDEFINED) == _type_utils.JitScalarType.FLOAT
    return dq


def canonicalize_repeated_input_concat_qdq(output_file: str) -> int:
    """Lower an exact repeated-input Concat onto its shared integer grid.

    PT2E represents a quantized Concat as ``DQ -> Concat -> Q -> DQ``.  When
    every Concat entry aliases the same tensor and the input/output qparams are
    identical, the middle floating-point round trip is an identity on integer
    codes.  Publishing aligned tensors (for example C1 -> C16) should therefore
    use the target-realizable ``Q -> Concat -> DQ`` form.  This rewrite is
    intentionally fail-closed: non-aliasing inputs, differing qparams, axes, or
    fan-out are left untouched.
    """
    from collections import defaultdict

    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    model = onnx.load(output_file)
    graph = model.graph
    producers = {value: node for node in graph.node for value in node.output}
    consumers = defaultdict(list)
    for node in graph.node:
        for value in set(node.input):
            consumers[value].append(node)
    initializers = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in graph.initializer
    }

    def constant_value(value_name):
        if value_name in initializers:
            return initializers[value_name]
        node = producers.get(value_name)
        if node is None:
            return None
        if node.op_type == "Constant":
            for attribute in node.attribute:
                if attribute.name == "value":
                    return numpy_helper.to_array(attribute.t)
                if attribute.name.startswith("value_"):
                    return np.asarray(helper.get_attribute_value(attribute))
            return None
        if node.op_type == "Identity" and node.input:
            return constant_value(node.input[0])
        if node.op_type == "Cast" and node.input:
            value = constant_value(node.input[0])
            to = next(
                (helper.get_attribute_value(attr) for attr in node.attribute if attr.name == "to"),
                None,
            )
            if value is not None and to is not None:
                return value.astype(helper.tensor_dtype_to_np_dtype(to))
        return None

    def qparams_equal(left, right):
        if len(left.input) < 3 or len(right.input) < 3:
            return False
        for left_name, right_name in zip(left.input[1:3], right.input[1:3]):
            left_value = constant_value(left_name)
            right_value = constant_value(right_name)
            if (
                left_value is None
                or right_value is None
                or left_value.dtype != right_value.dtype
                or not np.array_equal(left_value, right_value)
            ):
                return False
        left_axis = [
            helper.get_attribute_value(attr)
            for attr in left.attribute if attr.name == "axis"
        ]
        right_axis = [
            helper.get_attribute_value(attr)
            for attr in right.attribute if attr.name == "axis"
        ]
        return left_axis == right_axis

    remove_ids = set()
    rewritten = 0
    for concat in list(graph.node):
        if (
            concat.op_type != "Concat"
            or len(concat.input) < 2
            or len(set(concat.input)) != 1
        ):
            continue
        input_dq = producers.get(concat.input[0])
        concat_users = consumers.get(concat.output[0], [])
        if (
            input_dq is None
            or input_dq.op_type != "DequantizeLinear"
            or len(concat_users) != 1
            or concat_users[0].op_type != "QuantizeLinear"
        ):
            continue
        output_q = concat_users[0]
        q_users = consumers.get(output_q.output[0], [])
        if len(q_users) != 1 or q_users[0].op_type != "DequantizeLinear":
            continue
        output_dq = q_users[0]
        if not (
            qparams_equal(input_dq, output_q)
            and qparams_equal(output_q, output_dq)
        ):
            continue

        concat.input[:] = [input_dq.input[0]] * len(concat.input)
        output_dq.input[0] = concat.output[0]
        remove_ids.add(id(output_q))
        if consumers.get(input_dq.output[0], []) == [concat]:
            remove_ids.add(id(input_dq))
        rewritten += 1

    if rewritten:
        kept = [node for node in graph.node if id(node) not in remove_ids]
        used_values = {
            value for node in kept for value in node.input
        } | {output.name for output in graph.output}
        kept = [
            node for node in kept
            if node.op_type != "Constant"
            or any(value in used_values for value in node.output)
        ]
        del graph.node[:]
        graph.node.extend(kept)
        used_values = {
            value for node in graph.node for value in node.input
        } | {output.name for output in graph.output}
        kept_initializers = [
            initializer for initializer in graph.initializer
            if initializer.name in used_values
        ]
        del graph.initializer[:]
        graph.initializer.extend(kept_initializers)
        onnx.checker.check_model(model)
        onnx.save(model, output_file)
    return rewritten
