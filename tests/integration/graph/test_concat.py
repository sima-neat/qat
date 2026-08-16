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
import numpy as np
import onnx
import onnxruntime
import torch
from onnx import numpy_helper
from torch import nn

from sima_qat.qat_api import (
    sima_export_onnx,
    sima_finalize_qat_model,
    sima_prepare_qat_model,
)


class ConcatModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.low_range = nn.Conv2d(2, 2, kernel_size=1)
        self.high_range = nn.Conv2d(2, 2, kernel_size=1)
        self.output = nn.Conv2d(4, 2, kernel_size=1)
        with torch.no_grad():
            self.low_range.weight.fill_(0.05)
            self.low_range.bias.zero_()
            self.high_range.weight.fill_(2.0)
            self.high_range.bias.fill_(4.0)

    def forward(self, inputs):
        low = self.low_range(inputs)
        high = self.high_range(inputs)
        return self.output(torch.cat((low, high), dim=1))


def _onnx_value_maps(model):
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
    return producers, {**constants, **initializers}


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


def test_concat_keeps_independent_input_scales_and_runs_in_onnxruntime(tmp_path):
    torch.manual_seed(17)
    example_inputs = (torch.randn(2, 2, 6, 6),)
    prepared = sima_prepare_qat_model(ConcatModel(), example_inputs, "cpu")
    for multiplier in (1.0, 2.0, 3.0):
        prepared(example_inputs[0] * multiplier)
    finalized = sima_finalize_qat_model(prepared)

    output_file = tmp_path / "concat.onnx"
    sima_export_onnx(
        finalized,
        example_inputs,
        str(output_file),
        input_names=["input"],
        output_names=["output"],
    )

    model = onnx.load(str(output_file))
    onnx.checker.check_model(model)
    concat_nodes = [node for node in model.graph.node if node.op_type == "Concat"]
    assert len(concat_nodes) == 1
    concat = concat_nodes[0]
    assert len(concat.input) == 2

    producers, tensors = _onnx_value_maps(model)
    input_scales = []
    for concat_input in concat.input:
        dequantize = producers.get(concat_input)
        assert dequantize is not None
        assert dequantize.op_type == "DequantizeLinear"
        scale_name = _identity_root(dequantize.input[1], producers)
        assert scale_name in tensors
        scale = np.asarray(numpy_helper.to_array(tensors[scale_name])).reshape(-1)
        assert scale.size == 1
        input_scales.append(float(scale[0]))

    assert not np.isclose(input_scales[0], input_scales[1])
    assert any(
        node.op_type == "QuantizeLinear"
        and node.input[0] == concat.output[0]
        for node in model.graph.node
    )

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
    with torch.no_grad():
        torch_output = finalized(example_inputs[0]).numpy()
    np.testing.assert_allclose(ort_output, torch_output, rtol=0, atol=1e-6)
