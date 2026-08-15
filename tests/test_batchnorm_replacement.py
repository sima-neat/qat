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
from torch import nn
from torch.ao.nn.intrinsic import qat as intrinsic_qat

from sima_qat.qat_api import sima_finalize_qat_model, sima_prepare_qat_model


class ConvBatchNormReluModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(4)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        return self.relu(self.bn(self.conv(inputs)))


@pytest.mark.regression
def test_conv_batchnorm_relu_tracks_stats_then_folds_for_inference():
    torch.manual_seed(13)
    example_inputs = (torch.randn(4, 3, 8, 8) + 2.0,)
    prepared = sima_prepare_qat_model(
        ConvBatchNormReluModel(),
        example_inputs,
        "cpu",
    )

    fused = [
        module
        for module in prepared.modules()
        if isinstance(module, intrinsic_qat.ConvBnReLU2d)
    ]
    assert len(fused) == 1
    fused_module = fused[0]

    prepared.train()
    running_mean_before = fused_module.bn.running_mean.detach().clone()
    prepared(example_inputs[0])
    assert not torch.equal(running_mean_before, fused_module.bn.running_mean)

    prepared.eval()
    running_mean_eval = fused_module.bn.running_mean.detach().clone()
    prepared(example_inputs[0] * 3)
    torch.testing.assert_close(
        fused_module.bn.running_mean,
        running_mean_eval,
        rtol=0,
        atol=0,
    )

    finalized = sima_finalize_qat_model(prepared)
    assert finalized.meta["qat_weight_fake_quant_count"] >= 1
    assert not any(
        isinstance(module, nn.modules.batchnorm._BatchNorm)
        for module in finalized.modules()
    )
    assert not any(
        isinstance(module, intrinsic_qat.ConvBnReLU2d)
        for module in finalized.modules()
    )
    outputs = finalized(example_inputs[0])
    assert outputs.shape == (4, 4, 8, 8)
    assert torch.isfinite(outputs).all()
