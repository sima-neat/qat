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
import torch

from sima_qat.qat_api import (sima_prepare_qat_model, 
                              sima_finalize_qat_model, 
                              sima_export_onnx)

import pytest


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(10, 10, (1, 1))
        self.conv2 = torch.nn.Conv2d(10, 10, (1, 1))
        self.conv2.weight = torch.nn.Parameter(self.conv1.weight + 1.)
        self.conv2.bias = torch.nn.Parameter(self.conv1.bias + 1.)
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x = torch.cat([x1, x2])
        return x


@pytest.mark.regression
@pytest.mark.parametrize("model", [Model()])
def test_concat(model: torch.nn.Module):
    input_tensor = torch.randn(1, 10, 16, 16)
    model(input_tensor)
    example_inputs = (input_tensor, )

    prepared_model = sima_prepare_qat_model(model, example_inputs, 'cpu')

    prepared_model(example_inputs[0])

    prepared_model.cpu()

    converted_model = sima_finalize_qat_model(prepared_model)

    scales = []
    for node in converted_model.graph.nodes:

        if 'dequantize_per_tensor' in node.name:

            if 'cat' in [n.name for n in node.users]:
                scales.append(node.args[1])

                cat_node = [n for n in node.users][0]
                q_node = [n for n in cat_node.users][0]
                dq_node = [n for n in q_node.users][0]

                if dq_node.args[1] not in scales:
                    scales.append(dq_node.args[1])

    assert not all(x == scales[0] for x in scales)
