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
import copy

import numpy as np
import onnx
import onnxruntime
import pytest
import torch
from onnx import TensorProto, numpy_helper
from torch import nn

from sima_qat.qat_api import (
    sima_export_onnx,
    sima_finalize_qat_model,
    sima_prepare_qat_model,
)


class FlattenDropoutModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 3, kernel_size=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.flatten = nn.Flatten(start_dim=1)
        self.linear = nn.Linear(3 * 4 * 4, 5)

    def forward(self, inputs):
        outputs = self.relu(self.conv(inputs))
        outputs = self.dropout(outputs)
        return self.linear(self.flatten(outputs))


def _axis(node):
    return next(
        (
            int(attribute.i)
            for attribute in node.attribute
            if attribute.name == "axis"
        ),
        None,
    )


def _value_maps(model):
    producers = {
        output: node
        for node in model.graph.node
        for output in node.output
    }
    initializers = {
        initializer.name: initializer
        for initializer in model.graph.initializer
    }
    constants = {}
    for node in model.graph.node:
        if node.op_type != "Constant" or len(node.output) != 1:
            continue
        value = next(
            (
                attribute.t
                for attribute in node.attribute
                if attribute.name == "value"
            ),
            None,
        )
        if value is not None:
            constants[node.output[0]] = value
    return producers, {**constants, **initializers}, initializers


def _identity_root(value, producers):
    seen = set()
    while value not in seen:
        seen.add(value)
        producer = producers.get(value)
        if (
            producer is None
            or producer.op_type != "Identity"
            or len(producer.input) != 1
        ):
            return value
        value = producer.input[0]
    raise AssertionError("Identity cycle found in ONNX graph.")


@pytest.mark.regression
def test_onnx_qdq_topology_checker_runtime_and_cpu_restoration(tmp_path):
    torch.manual_seed(23)
    example_inputs = (torch.randn(2, 2, 4, 4),)
    prepared = sima_prepare_qat_model(
        FlattenDropoutModel(),
        example_inputs,
        "cpu",
    )
    prepared(example_inputs[0])
    finalized = sima_finalize_qat_model(prepared)
    with torch.no_grad():
        torch_output = finalized(example_inputs[0]).numpy()
    state_before_export = copy.deepcopy(finalized.state_dict())

    output_file = tmp_path / "flatten_dropout.onnx"
    returned_model = sima_export_onnx(
        finalized,
        example_inputs,
        str(output_file),
        input_names=["input"],
        output_names=["output"],
        device="cpu",
    )

    assert returned_model is finalized
    assert all(
        tensor.device.type == "cpu"
        for tensor in (
            *finalized.parameters(),
            *finalized.buffers(),
        )
    )
    for name, value in finalized.state_dict().items():
        torch.testing.assert_close(
            value,
            state_before_export[name],
            rtol=0,
            atol=0,
        )

    model = onnx.load(str(output_file))
    onnx.checker.check_model(model, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(inferred)
    assert [(opset.domain, opset.version) for opset in model.opset_import] == [
        ("", 17)
    ]
    assert model.graph.input[0].name == "input"
    assert model.graph.output[0].name == "output"
    assert model.graph.input[0].type.tensor_type.elem_type == TensorProto.FLOAT
    assert model.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT
    assert all(node.domain in ("", "ai.onnx") for node in model.graph.node)
    assert not any(node.op_type == "Dropout" for node in model.graph.node)

    quantizers = [
        node for node in model.graph.node if node.op_type == "QuantizeLinear"
    ]
    dequantizers = [
        node for node in model.graph.node if node.op_type == "DequantizeLinear"
    ]
    learned_ops = [
        node for node in model.graph.node if node.op_type in {"Conv", "Gemm"}
    ]
    assert quantizers
    assert len(learned_ops) == 2
    assert len(dequantizers) - len(quantizers) == len(learned_ops)

    producers, tensors, initializers = _value_maps(model)
    for quantizer in quantizers:
        producer = producers.get(quantizer.input[0])
        assert producer is None or producer.op_type != "DequantizeLinear"
        assert _axis(quantizer) is None
        for qparam in quantizer.input[1:3]:
            qparam = _identity_root(qparam, producers)
            assert qparam in tensors
            assert list(tensors[qparam].dims) == []

    for dequantizer in dequantizers:
        if _axis(dequantizer) is None:
            for qparam in dequantizer.input[1:3]:
                qparam = _identity_root(qparam, producers)
                assert qparam in tensors
                assert list(tensors[qparam].dims) == []

    for learned_op in learned_ops:
        weight_dequantizer = producers.get(learned_op.input[1])
        assert weight_dequantizer is not None
        assert weight_dequantizer.op_type == "DequantizeLinear"
        assert _axis(weight_dequantizer) == 0

        weight_name = _identity_root(weight_dequantizer.input[0], producers)
        scale_name = _identity_root(weight_dequantizer.input[1], producers)
        zero_point_name = _identity_root(weight_dequantizer.input[2], producers)
        assert weight_name in initializers
        assert scale_name in tensors
        assert zero_point_name in tensors

        weight = initializers[weight_name]
        scale = tensors[scale_name]
        zero_point = tensors[zero_point_name]
        assert weight.data_type == TensorProto.INT8
        assert zero_point.data_type == TensorProto.INT8
        assert len(scale.dims) == 1
        assert list(scale.dims) == list(zero_point.dims)
        assert scale.dims[0] == weight.dims[0]
        weight_values = numpy_helper.to_array(weight)
        assert weight_values.min() >= -127
        assert weight_values.max() <= 127
        assert np.all(numpy_helper.to_array(scale) > 0)

    session_options = onnxruntime.SessionOptions()
    session_options.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    session = onnxruntime.InferenceSession(
        str(output_file),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    ort_output = session.run(
        None,
        {"input": example_inputs[0].numpy()},
    )[0]
    np.testing.assert_allclose(ort_output, torch_output, rtol=0, atol=1e-6)
