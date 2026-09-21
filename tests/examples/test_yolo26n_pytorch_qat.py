"""Smoke coverage for the standalone YOLO26n QAT example."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image


EXAMPLE = Path(__file__).parents[2] / "examples" / "yolo26n_pytorch_qat"
sys.path.insert(0, str(EXAMPLE))

from coco import CocoDetectionDataset, collate_detection  # noqa: E402
from evaluate import decode_boxdecode, decode_end2end, detections_to_coco  # noqa: E402
from loss import YOLO26Loss  # noqa: E402
from model import build_yolo26n  # noqa: E402
from sima_qat import (  # noqa: E402
    sima_freeze_batchnorm_stats,
    sima_freeze_qat,
    sima_prepare_qat_model,
)
from train import optimizer_parameter_groups  # noqa: E402


pytestmark = pytest.mark.regression


def _target(batch_size: int = 1) -> dict[str, torch.Tensor]:
    return {
        "batch_idx": torch.arange(batch_size, dtype=torch.long),
        "cls": torch.arange(batch_size, dtype=torch.float32),
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]).repeat(batch_size, 1),
    }


def test_coco_json_loader_produces_normalized_targets(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (10, 20), (10, 20, 30)).save(images / "sample.jpg")
    annotations = tmp_path / "instances.json"
    annotations.write_text(
        json.dumps(
            {
                "images": [{"id": 7, "file_name": "sample.jpg", "width": 10, "height": 20}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 7,
                        "category_id": 5,
                        "bbox": [1, 2, 4, 8],
                        "iscrowd": 0,
                    }
                ],
                "categories": [{"id": index, "name": str(index)} for index in range(1, 81)],
            }
        ),
        encoding="utf-8",
    )
    dataset = CocoDetectionDataset(
        images,
        annotations,
        image_size=32,
        horizontal_flip=0,
    )
    image, target = dataset[0]
    assert image.shape == (3, 32, 32)
    torch.testing.assert_close(
        target["bboxes"],
        torch.tensor([[0.4, 0.3, 0.2, 0.4]]),
    )
    assert target["cls"].tolist() == [4.0]
    batched_images, batched_targets = collate_detection([(image, target)])
    assert batched_images.shape == (1, 3, 32, 32)
    assert batched_targets["batch_idx"].tolist() == [0]


def test_pure_model_shape_and_loss_backward() -> None:
    model = build_yolo26n().train()
    inputs = torch.randn(1, 3, 64, 64)
    outputs = model(inputs)
    assert outputs["one2many"]["boxes"].shape == (1, 4, 84)
    assert outputs["one2many"]["scores"].shape == (1, 80, 84)
    assert outputs["one2one"]["boxes"].shape == (1, 4, 84)
    loss, metrics = YOLO26Loss()(outputs, _target())
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in metrics.values())
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_full_model_dynamic_qat_prepare_forward_and_freeze() -> None:
    inputs = torch.randn(1, 3, 64, 64)
    prepared = sima_prepare_qat_model(
        build_yolo26n().train(),
        (inputs,),
        "cpu",
        dynamic_batch=True,
    )
    outputs = prepared(torch.randn(2, 3, 64, 64))
    loss, _ = YOLO26Loss()(outputs, _target(2))
    loss.backward()
    sima_freeze_qat(prepared)
    assert bool(prepared.qat_frozen.item())


def test_batchnorm_can_freeze_before_quantization_grids() -> None:
    inputs = torch.randn(2, 3, 64, 64)
    prepared = sima_prepare_qat_model(
        build_yolo26n().train(),
        (inputs,),
        "cpu",
        dynamic_batch=True,
    )
    sima_freeze_batchnorm_stats(prepared)
    running_means = {
        name: buffer.detach().clone()
        for name, buffer in prepared.named_buffers()
        if name.endswith("running_mean")
    }

    prepared.train()
    prepared(inputs)

    assert not bool(prepared.qat_frozen.item())
    for name, expected in running_means.items():
        torch.testing.assert_close(
            dict(prepared.named_buffers())[name], expected, rtol=0, atol=0
        )


def test_optimizer_excludes_vector_parameters_from_weight_decay() -> None:
    model = build_yolo26n()
    groups = optimizer_parameter_groups(model, 5e-4)
    decayed = {id(parameter) for parameter in groups[0]["params"]}
    not_decayed = {id(parameter) for parameter in groups[1]["params"]}

    assert groups[0]["weight_decay"] == 5e-4
    assert groups[1]["weight_decay"] == 0
    assert decayed.isdisjoint(not_decayed)
    assert decayed | not_decayed == {id(parameter) for parameter in model.parameters()}
    assert all(parameter.ndim > 1 for parameter in groups[0]["params"])
    assert all(parameter.ndim <= 1 for parameter in groups[1]["params"])


def test_one2one_decode_and_inverse_letterbox() -> None:
    boxes = torch.tensor([[[0.25], [0.5], [0.75], [1.0]]])
    scores = torch.full((1, 80, 1), -20.0)
    scores[0, 3, 0] = 4.0
    feature = torch.zeros(1, 1, 1, 1)
    predictions = {
        "one2one": {
            "boxes": boxes,
            "scores": scores,
            "feats": [feature, feature[:, :, :0, :0], feature[:, :, :0, :0]],
        }
    }
    detections = decode_end2end(predictions, max_detections=1)
    torch.testing.assert_close(
        detections[0][0, :4],
        torch.tensor([2.0, 0.0, 10.0, 12.0]),
    )
    coco = detections_to_coco(
        detections,
        [
            {
                "image_id": 7,
                "original_width": 8,
                "original_height": 8,
                "scale": 1.0,
                "left": 2,
                "top": 2,
            }
        ],
        tuple(range(80)),
    )
    assert coco[0]["image_id"] == 7
    assert coco[0]["category_id"] == 3
    assert coco[0]["bbox"] == [0.0, 0.0, 8.0, 8.0]


def test_boxdecode_path_is_nms_free_with_opt_in_legacy_nms() -> None:
    boxes = torch.tensor(
        [[[0.5, 1.5], [0.5, 0.5], [1.5, 0.5], [0.5, 0.5]]]
    )
    scores = torch.full((1, 80, 2), -20.0)
    scores[0, 2, 0] = 4.0
    scores[0, 3, 0] = 3.0  # BoxDecode keeps only the cell's best class.
    scores[0, 2, 1] = 3.5  # Same-class duplicate is removed by NMS.
    feature = torch.zeros(1, 1, 1, 2)
    predictions = {
        "one2one": {
            "boxes": boxes,
            "scores": scores,
            "feats": [feature, feature[:, :, :0, :0], feature[:, :, :0, :0]],
        }
    }
    detections = decode_boxdecode(predictions, max_detections=2, nms_iou=0.7)
    assert detections[0].shape == (2, 6)
    assert detections[0][:, 5].tolist() == [2.0, 2.0]

    legacy = decode_boxdecode(
        predictions,
        max_detections=2,
        nms_iou=0.7,
        legacy_nms=True,
    )
    assert legacy[0].shape == (1, 6)
    assert legacy[0][0, 5].item() == 2
