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
    # torch >= 2.x removed the internal beartype shim; it was only runtime type
    # checking on the symbolic functions, so a no-op passthrough is fine.
    class _beartype:  # noqa: N801
        @staticmethod
        def beartype(fn):
            return fn


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
    if (quant_min, quant_max) not in [(0, 255), (-128, 127), (0, 127)]:
        raise errors.SymbolicValueError(
            "For (quant_min, quant_max), ONNX allows only (0, 127), (0, 255) and (-128, 127). "
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
    out_dtype=None,  # added in torch 2.4 (quantized_decomposed::dequantize_per_tensor); unused here
):
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
    out_dtype=None,  # added in torch 2.4 (quantized_decomposed::dequantize_per_channel); unused here
):
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
