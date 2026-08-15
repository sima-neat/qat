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
import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.ao.quantization import disable_observer

from sima_qat.qat_api import sima_finalize_qat_model, sima_prepare_qat_model


class DropoutModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=1)
        self.dropout = nn.Dropout(p=0.75)
        self.conv2 = nn.Conv2d(4, 4, kernel_size=1)

    def forward(self, inputs):
        outputs = self.dropout(self.conv1(inputs))
        outputs = F.dropout(outputs, p=0.5, training=self.training)
        return self.conv2(outputs)


@pytest.mark.regression
def test_module_and_functional_dropout_are_absent_from_prepared_model():
    model = DropoutModel()
    example_inputs = (torch.randn(2, 3, 6, 6),)
    prepared = sima_prepare_qat_model(model, example_inputs, "cpu")

    assert isinstance(model.dropout, nn.Dropout)
    assert not any(
        isinstance(module, nn.modules.dropout._DropoutNd)
        for module in prepared.modules()
    )
    assert not any(
        node.op == "call_function"
        and node.target
        in {
            F.dropout,
            F.dropout1d,
            F.dropout2d,
            F.dropout3d,
            F.alpha_dropout,
            F.feature_alpha_dropout,
        }
        for node in prepared.graph.nodes
    )

    prepared.train()
    prepared(example_inputs[0])
    prepared.apply(disable_observer)
    outputs_1 = prepared(example_inputs[0])
    outputs_2 = prepared(example_inputs[0])
    torch.testing.assert_close(outputs_1, outputs_2, rtol=0, atol=0)

    finalized = sima_finalize_qat_model(prepared)
    assert torch.isfinite(finalized(example_inputs[0])).all()
