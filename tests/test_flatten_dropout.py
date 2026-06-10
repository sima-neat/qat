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
import torch
import onnx

from sima_qat.qat_api import (sima_prepare_qat_model, 
                              sima_finalize_qat_model, 
                              sima_export_onnx)

import pytest


class Model(torch.nn.Module):
    # adapted from final calculations of MobileNetv2
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(10, 10, (1, 1))
        self.dropout = torch.nn.Dropout()
        self.linear = torch.nn.Linear(2560, 1000)
    
    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.linear(x)
        return x


def find_node_input(model, input_name):
    for node in list(model.graph.node):
        for i, input in enumerate(node.input):
            if input == input_name:
                return node, i

    raise RuntimeError(f"Node with input as {input_name} not found")


@pytest.mark.regression
@pytest.mark.parametrize("model", [Model()])
def test_fatten_dropout(model: torch.nn.Module):
    input_tensor = torch.randn(1, 10, 16, 16)
    model(input_tensor)
    example_inputs = (input_tensor, )

    prepared_model = sima_prepare_qat_model(model, example_inputs, 'cpu')

    prepared_model(example_inputs[0])

    prepared_model.cpu()

    converted_model = sima_finalize_qat_model(prepared_model)

    sima_export_onnx(converted_model, example_inputs, 'flatten_dropout.onnx')

    #check if onnx model has two consecutive QDQs because of dropout not being handled
    model = onnx.load('flatten_dropout.onnx')
    for node in model.graph.node:
        if node.op_type == 'DequantizeLinear':
            is_last_node = any(node.output[0] == x.name for x in list(model.graph.output))

            if not is_last_node:
                output = node.output[0]
                next_node = find_node_input(model, output)[0]
                assert next_node.op_type != 'QuantizeLinear'
