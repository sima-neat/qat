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
import pytest
import torch
from torch import nn
from torch.ao.quantization import FakeQuantize

from sima_qat.qat_api import (
    sima_export_onnx,
    sima_finalize_qat_model,
    sima_prepare_qat_model,
)


class SliceSelectUnsqueezeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(
            3,
            4,
            bias=False,
            kernel_size=3,
            stride=1,
        )

    def forward(self, inputs):
        channels = [
            torch.unsqueeze(inputs[:, channel], 1)
            for channel in range(3)
        ]
        return self.conv(torch.cat(channels, dim=1))


@pytest.mark.regression
def test_slice_select_unsqueeze_boundaries_export_and_run(tmp_path):
    torch.manual_seed(19)
    example_inputs = (torch.randn(2, 3, 8, 8),)
    prepared = sima_prepare_qat_model(
        SliceSelectUnsqueezeModel(),
        example_inputs,
        "cpu",
    )
    prepared(example_inputs[0])
    finalized = sima_finalize_qat_model(prepared)
    torch_output = finalized(example_inputs[0]).detach().numpy()
    assert torch_output.shape == (2, 4, 6, 6)

    output_file = tmp_path / "slice_select_unsqueeze.onnx"
    sima_export_onnx(
        finalized,
        example_inputs,
        str(output_file),
        input_names=["input"],
        output_names=["output"],
    )
    model = onnx.load(str(output_file))
    onnx.checker.check_model(model)

    op_types = [node.op_type for node in model.graph.node]
    assert op_types.count("Gather") == 3
    assert op_types.count("Unsqueeze") == 3
    assert op_types.count("Concat") == 1
    assert op_types.count("Conv") == 1

    consumers = {}
    producers = {}
    for node in model.graph.node:
        for value in node.input:
            consumers.setdefault(value, []).append(node)
        for value in node.output:
            producers[value] = node

    concat = next(node for node in model.graph.node if node.op_type == "Concat")
    for unsqueeze in (
        node for node in model.graph.node if node.op_type == "Unsqueeze"
    ):
        quantizers = [
            node
            for node in consumers.get(unsqueeze.output[0], [])
            if node.op_type == "QuantizeLinear"
        ]
        assert len(quantizers) == 1
        dequantizers = [
            node
            for node in consumers.get(quantizers[0].output[0], [])
            if node.op_type == "DequantizeLinear"
        ]
        assert len(dequantizers) == 1
        assert dequantizers[0].output[0] in concat.input
        assert producers[unsqueeze.input[0]].op_type == "Gather"

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
    activation_steps = [
        float(module.scale.max())
        for module in finalized.modules()
        if isinstance(module, FakeQuantize) and not module.is_per_channel
    ]
    assert activation_steps
    np.testing.assert_allclose(
        ort_output,
        torch_output,
        rtol=0,
        atol=2 * max(activation_steps) + 1e-7,
    )
