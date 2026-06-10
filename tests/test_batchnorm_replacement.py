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
                              sima_export_onnx,
                              convert_pt2e,
                              _ensure_bn_tracking_meta,
                              SimaQatWrapper)

import pytest


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(3)
    
    def forward(self, x):
        x = self.bn(x)
        return x


def sima_finalize_qat_model_no_bn_replacement(qat_model: torch.fx.GraphModule) -> torch.nn.Module:
    assert isinstance(qat_model, torch.nn.Module)
    _ensure_bn_tracking_meta(qat_model)
    m = convert_pt2e(qat_model, use_reference_representation=False)
    sima_mod = SimaQatWrapper(source=m, label='fq')
    sima_mod.eval()

    return sima_mod


@pytest.mark.regression
@pytest.mark.parametrize("model", [Model()])
def test_batchnorm_replacement(model: torch.nn.Module):
    input_tensor = torch.randn(1, 3, 224, 224)
    example_inputs = (input_tensor, )

    prepared_model = sima_prepare_qat_model(model, example_inputs, 'cpu')

    prepared_model(example_inputs[0])

    prepared_model.cpu()
    prepared_model_copy = copy.deepcopy(prepared_model)

    converted_model_original = sima_finalize_qat_model_no_bn_replacement(prepared_model)
    out_1 = converted_model_original(input_tensor)

    converted_model_replaced_bn = sima_finalize_qat_model(prepared_model_copy)
    out_2 = converted_model_replaced_bn(input_tensor)

    assert torch.equal(out_1, out_2)
    
    for node in converted_model_replaced_bn.graph.nodes:
        assert (node.target not in [torch.ops.aten._native_batch_norm_legit_no_training.default])
    
    sima_export_onnx(converted_model_replaced_bn, example_inputs, 'batchnorm_replacement.onnx')
