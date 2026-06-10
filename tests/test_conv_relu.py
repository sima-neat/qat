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

from sima_qat.qat_api import sima_prepare_qat_model, sima_finalize_qat_model

import pytest

class ConvReluModel(torch.nn.Module):

    def __init__(self):
        super(ConvReluModel, self).__init__()

        self.conv1 = torch.nn.Conv2d(3, 200, 3)
        self.activation = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(200, 10, 3)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.softmax(x)
        return x


@pytest.mark.regression
@pytest.mark.parametrize("model", [ConvReluModel()])
def test_conv_relu_model(model: torch.nn.Module):
    example_inputs = (torch.randn(1, 3, 224, 224),)
    prepared_model = sima_prepare_qat_model(model, example_inputs, 'cpu')

    prepared_model(example_inputs[0][0])

    prepared_model.cpu()

    converted_model = sima_finalize_qat_model(prepared_model)

    for node in converted_model.graph.nodes:
        if node.name == 'relu':
            # check if prior node is conv and not a qdq node
            assert node.args[0].target is torch.ops.aten.conv2d.default

            # check if next node is the start of a qdq node
            assert node.next.target is torch.ops.quantized_decomposed.quantize_per_tensor.default
