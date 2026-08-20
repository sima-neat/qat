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
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.ao.quantization import FakeQuantize, disable_observer
from torch.ao.quantization.observer import PerChannelMinMaxObserver
from torch.fx import GraphModule

import sima_qat
from sima_qat import qat_api
from sima_qat.qat_api import SimaQatWrapper
from sima_qat.sima_quantizer import SimaMovingAverageMinMaxObserver


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=3)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(72, 3)

    def forward(self, inputs):
        outputs = self.relu(self.conv(inputs))
        return self.fc(self.flatten(outputs))


def _prepared_tiny(example_inputs):
    return qat_api.sima_prepare_qat_model(
        TinyClassifier(),
        example_inputs,
        device="cpu",
    )


@pytest.mark.smoke
def test_qat_public_api_imports_and_supported_runtime(monkeypatch):
    release = tuple(
        int(part)
        for part in torch.__version__.split("+", 1)[0].split(".")[:3]
    )

    assert sima_qat.__name__ == "sima_qat"
    assert sima_qat.__all__ == [
        "sima_prepare_qat_model",
        "sima_finalize_qat_model",
        "sima_export_onnx",
        "__version__",
    ]
    for public_name in sima_qat.__all__[:-1]:
        assert callable(getattr(sima_qat, public_name))
    assert callable(qat_api.sima_prepare_qat_model)
    assert callable(qat_api.sima_finalize_qat_model)
    assert callable(qat_api.sima_export_onnx)
    expected_version = (
        Path(__file__).resolve().parents[3] / "VERSION.in"
    ).read_text(encoding="utf-8").strip()
    assert sima_qat.__version__ == expected_version

    # A source checkout must not report an older installed distribution.
    monkeypatch.setattr(sima_qat, "version", lambda _distribution: "999.0.0")
    assert sima_qat._resolve_version() == expected_version

    assert release == (2, 3, 1)
    assert qat_api._torch_release("2.3.1+cu121") == (2, 3, 1)

    # The public lifecycle must not accidentally regain the Python-3.12-
    # incompatible PT2E/Dynamo entry points.
    assert not hasattr(qat_api, "capture_pre_autograd_graph")
    assert not hasattr(qat_api, "prepare_qat_pt2e")
    assert not hasattr(qat_api, "convert_pt2e")
    assert not hasattr(qat_api, "_export_training_graph")


@pytest.mark.smoke
def test_prepare_train_and_finalize_lifecycle_preserves_optimizer_parameters():
    torch.manual_seed(7)
    model = TinyClassifier()
    example_inputs = (torch.randn(2, 1, 8, 8),)
    original_parameters = tuple(model.parameters())
    optimizer = torch.optim.SGD(original_parameters, lr=0.05)

    prepared = qat_api.sima_prepare_qat_model(model, example_inputs, "cpu")

    assert isinstance(prepared, GraphModule)
    assert isinstance(prepared, SimaQatWrapper)
    assert prepared.qat_stage == "scaffold"
    assert prepared.training
    assert prepared.qat_state.dtype == torch.int8
    assert prepared.qat_state.tolist() == [0]
    assert prepared.qat_backend_version.dtype == torch.int16
    assert prepared.qat_backend_version.tolist() == [1]
    assert {id(parameter) for parameter in prepared.parameters()} == {
        id(parameter) for parameter in original_parameters
    }

    activation_fake_quants = [
        module
        for module in prepared.modules()
        if isinstance(module, FakeQuantize)
        and isinstance(
            module.activation_post_process,
            SimaMovingAverageMinMaxObserver,
        )
    ]
    assert activation_fake_quants
    for fake_quant in activation_fake_quants:
        observer = fake_quant.activation_post_process
        assert observer.quant_min == -128
        assert observer.quant_max == 127
        assert observer.dtype == torch.qint8
        assert observer.qscheme == torch.per_tensor_affine

    weight_observers = [
        module.weight_fake_quant
        for module in prepared.modules()
        if hasattr(module, "weight_fake_quant")
    ]
    assert len(weight_observers) == 2
    for observer in weight_observers:
        assert isinstance(observer, PerChannelMinMaxObserver)
        assert not isinstance(observer, FakeQuantize)
        assert observer.quant_min == -127
        assert observer.quant_max == 127
        assert observer.ch_axis == 0

    weights_before = [parameter.detach().clone() for parameter in original_parameters]
    optimizer.zero_grad(set_to_none=True)
    loss = prepared(example_inputs[0]).square().mean()
    loss.backward()
    optimizer.step()

    assert all(observer.min_val.numel() > 0 for observer in weight_observers)
    assert any(
        not torch.equal(before, after.detach())
        for before, after in zip(weights_before, original_parameters)
    )

    assert prepared.eval() is prepared
    assert not prepared.training
    assert prepared.train() is prepared
    assert prepared.training

    finalized = qat_api.sima_finalize_qat_model(prepared)
    assert finalized is prepared
    assert finalized.qat_stage == "fq"
    assert finalized.qat_state.tolist() == [1]
    assert not finalized.training
    assert all(not parameter.requires_grad for parameter in finalized.parameters())
    assert torch.isfinite(finalized(example_inputs[0])).all()
    assert qat_api.sima_finalize_qat_model(finalized) is finalized

    with pytest.raises(RuntimeError, match="training mode is disallowed"):
        finalized.train(True)


def test_prepared_checkpoint_schema_round_trip_and_stage_guards():
    torch.manual_seed(11)
    example_inputs = (torch.randn(2, 1, 8, 8),)
    prepared = _prepared_tiny(example_inputs)
    prepared(example_inputs[0])
    checkpoint = copy.deepcopy(prepared.state_dict())

    restored = _prepared_tiny(example_inputs)
    restored.load_state_dict(checkpoint)
    prepared.apply(disable_observer)
    restored.apply(disable_observer)
    prepared.eval()
    restored.eval()
    torch.testing.assert_close(
        restored(example_inputs[0]),
        prepared(example_inputs[0]),
        rtol=0,
        atol=0,
    )

    legacy_checkpoint = copy.deepcopy(checkpoint)
    legacy_checkpoint.pop("qat_backend_version")
    with pytest.raises(RuntimeError, match="Legacy PT2E checkpoints"):
        restored.load_state_dict(legacy_checkpoint, strict=False)

    wrong_schema = copy.deepcopy(checkpoint)
    wrong_schema["qat_backend_version"].fill_(99)
    with pytest.raises(RuntimeError, match="Unsupported QAT checkpoint schema"):
        restored.load_state_dict(wrong_schema)

    parent = nn.Module()
    parent.add_module("qat_model", _prepared_tiny(example_inputs))
    nested_checkpoint = copy.deepcopy(parent.state_dict())
    parent.load_state_dict(nested_checkpoint)

    nested_wrong_schema = copy.deepcopy(nested_checkpoint)
    nested_wrong_schema["qat_model.qat_backend_version"].fill_(99)
    with pytest.raises(RuntimeError, match="Unsupported QAT checkpoint schema"):
        parent.load_state_dict(nested_wrong_schema)

    nested_legacy = copy.deepcopy(nested_checkpoint)
    nested_legacy.pop("qat_model.qat_backend_version")
    with pytest.raises(RuntimeError, match="Legacy PT2E checkpoints"):
        parent.load_state_dict(nested_legacy, strict=False)

    finalized = qat_api.sima_finalize_qat_model(prepared)
    with pytest.raises(RuntimeError, match="does not match checkpoint"):
        restored.load_state_dict(finalized.state_dict())

    with pytest.raises(RuntimeError, match="cannot be prepared again"):
        qat_api.sima_prepare_qat_model(finalized, example_inputs, "cpu")


def test_lifecycle_rejects_invalid_inputs_and_unfinalized_export(tmp_path):
    example_inputs = (torch.randn(2, 1, 8, 8),)

    with pytest.raises(RuntimeError, match="must be an nn.Module"):
        qat_api.sima_prepare_qat_model("not-a-module", example_inputs, "cpu")
    with pytest.raises(RuntimeError, match="supplied as a tuple"):
        qat_api.sima_prepare_qat_model(TinyClassifier(), list(example_inputs), "cpu")
    with pytest.raises(RuntimeError, match="Finalize expects"):
        qat_api.sima_finalize_qat_model(TinyClassifier())

    prepared = _prepared_tiny(example_inputs)
    with pytest.raises(RuntimeError, match="must be finalized"):
        qat_api.sima_export_onnx(
            prepared,
            example_inputs,
            str(tmp_path / "unfinalized.onnx"),
        )
