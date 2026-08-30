# **************************************************************************
#||                        SiMa.ai CONFIDENTIAL                          ||
#||   Unpublished Copyright (c) 2024 SiMa.ai, All Rights Reserved.       ||
# **************************************************************************
import json

import pytest
import torch

import sima_qat
from sima_qat.session import QATRecipe, load_recipe


class TinyRegressor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 3, padding=1)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(4, 1, 1)

    def forward(self, image):
        return self.conv2(self.relu(self.conv1(image)))


class TinySS2D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Conv2d(3, 2, 1)

    def forward(self, image):
        return self.projection(image)


@pytest.mark.smoke
@pytest.mark.regression
def test_simple_session_api_is_exported():
    assert callable(sima_qat.prepare)
    assert callable(sima_qat.load_recipe)
    assert issubclass(sima_qat.QATSession, torch.nn.Module)


@pytest.mark.regression
def test_session_prepare_calibrate_train_freeze_validate_and_export(tmp_path):
    torch.manual_seed(5)
    original = TinyRegressor()
    original_state = {
        name: tensor.detach().clone() for name, tensor in original.state_dict().items()
    }
    images = torch.randn(4, 3, 8, 8)
    targets = torch.randn(4, 1, 8, 8)

    qat = sima_qat.prepare(original, images[:1], device="cpu")
    assert qat.recipe.name == "strict_int8"
    assert qat.state == "prepared"
    assert sum(parameter.numel() for parameter in qat.parameters()) == sum(
        parameter.numel() for parameter in qat.model.parameters()
    )
    assert all(
        not parameter.requires_grad for parameter in qat.float_teacher.parameters()
    )

    qat.calibrate([(images[:2], targets[:2]), (images[2:], targets[2:])], batches=2)
    assert qat.state == "calibrated"
    assert qat.calibration_batches == 2
    assert qat.training

    optimizer = torch.optim.SGD(qat.parameters(), lr=1e-3)
    optimizer.zero_grad()
    prediction = qat(images[:2])
    task_loss = torch.nn.functional.mse_loss(prediction, targets[:2])
    loss = qat.loss(task_loss)
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    assert loss >= task_loss

    qat.freeze()
    report = qat.validate()
    report.raise_for_failure()
    assert report.weighted_ops == 2
    assert report.weighted_ops_covered == 2
    assert report.activation_fake_quantizers > 0

    resumed = sima_qat.prepare(TinyRegressor(), images[:1], device="cpu")
    resumed.load_state_dict(qat.state_dict())
    assert resumed.state == "frozen"
    for name, tensor in qat.model.state_dict().items():
        torch.testing.assert_close(
            resumed.model.state_dict()[name], tensor, rtol=0, atol=0
        )

    bundle = qat.export(tmp_path / "bundle")
    assert qat.state == "exported"
    assert bundle.onnx_path.is_file()
    assert bundle.manifest_path.is_file()
    assert len(bundle.onnx_sha256) == 64
    assert bundle.quantize_linear_nodes > 0
    assert bundle.dequantize_linear_nodes > 0
    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    assert manifest["onnx"]["sha256"] == bundle.onnx_sha256
    assert manifest["validation"]["passed"]
    assert "board execution" in manifest["limitations"][0]

    # The facade prepares copies; customer source weights and mode remain intact.
    assert original.training
    for name, tensor in original.state_dict().items():
        torch.testing.assert_close(tensor, original_state[name], rtol=0, atol=0)


@pytest.mark.regression
def test_session_enforces_lifecycle_and_loss_order(tmp_path):
    image = torch.randn(1, 3, 8, 8)
    qat = sima_qat.prepare(TinyRegressor(), image, device="cpu")

    with pytest.raises(RuntimeError, match="immediately follow"):
        qat.loss(torch.tensor(1.0))
    with pytest.raises(RuntimeError, match=r"qat.freeze\(\)"):
        qat.export(tmp_path)
    with pytest.raises(RuntimeError, match="observers see data"):
        qat.freeze()


@pytest.mark.regression
def test_auto_recipe_detects_state_space_model_and_supports_mapping_calibration():
    image = torch.randn(1, 3, 4, 4)
    qat = sima_qat.prepare(TinySS2D(), image, target="sima", device="cpu")

    assert qat.target == "modalix"
    assert qat.recipe.name == "strict_int8_ssm"
    assert qat.recipe.activation_observer == "minmax"
    assert qat.recipe.full_range_ste
    assert qat.state_space_regions == ("<root>",)

    qat.calibrate([{"image": image, "unused_label": torch.zeros(1)}], batches=1)
    qat.freeze()
    assert qat.validate().passed


@pytest.mark.regression
def test_yaml_recipe_is_data_only_and_rejects_unknown_fields(tmp_path):
    recipe_path = tmp_path / "qualified.yaml"
    recipe_path.write_text(
        """\
schema_version: 1
name: qualified
activation_observer: histogram
full_range_ste: true
learn_scales: false
shadow_weight: 0.25
strict_int8: true
""",
        encoding="utf-8",
    )
    recipe = load_recipe(recipe_path)
    assert recipe == QATRecipe(
        name="qualified",
        activation_observer="histogram",
        full_range_ste=True,
        learn_scales=False,
        shadow_weight=0.25,
    )

    recipe_path.write_text(
        "schema_version: 1\nname: bad\ncallback: run_me\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Unknown QAT recipe field"):
        load_recipe(recipe_path)
