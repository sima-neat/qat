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
import onnx
import pytest
import torch
from torch import nn

from sima_qat.qat_api import (
    sima_export_onnx,
    sima_finalize_qat_model,
    sima_prepare_qat_model,
)


class CheckDeviceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 2, kernel_size=1)
        self.register_buffer("offset", torch.tensor(0.25))

    def forward(self, inputs):
        return self.conv(inputs) + self.offset


def _assert_all_state_on_cpu(model):
    tensors = (*model.parameters(), *model.buffers())
    assert tensors
    assert all(tensor.device == torch.device("cpu") for tensor in tensors)


def test_prepare_finalize_and_export_restore_requested_cpu_device(tmp_path):
    example_inputs = (torch.randn(2, 2, 4, 4),)
    prepared = sima_prepare_qat_model(
        CheckDeviceModel(),
        example_inputs,
        torch.device("cpu"),
    )
    _assert_all_state_on_cpu(prepared)
    prepared(example_inputs[0])

    finalized = sima_finalize_qat_model(prepared)
    _assert_all_state_on_cpu(finalized)
    output_file = tmp_path / "prepared_model_device.onnx"
    returned = sima_export_onnx(
        qat_model=finalized,
        inputs=example_inputs,
        output_file=str(output_file),
        device=torch.device("cpu"),
    )

    assert returned is finalized
    _assert_all_state_on_cpu(returned)
    onnx.checker.check_model(onnx.load(str(output_file)))


def test_unavailable_cuda_is_rejected_before_prepare_or_export(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    example_inputs = (torch.randn(2, 2, 4, 4),)

    with pytest.raises(RuntimeError, match="CUDA was requested"):
        sima_prepare_qat_model(CheckDeviceModel(), example_inputs, "cuda")

    prepared = sima_prepare_qat_model(CheckDeviceModel(), example_inputs, "cpu")
    prepared(example_inputs[0])
    finalized = sima_finalize_qat_model(prepared)
    output_file = tmp_path / "must_not_exist.onnx"
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        sima_export_onnx(
            finalized,
            example_inputs,
            str(output_file),
            device="cuda",
        )
    assert not output_file.exists()
    _assert_all_state_on_cpu(finalized)
