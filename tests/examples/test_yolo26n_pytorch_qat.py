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
from loss import YOLO26Loss  # noqa: E402
from model import build_yolo26n  # noqa: E402
from sima_qat import sima_freeze_qat, sima_prepare_qat_model  # noqa: E402


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
    loss, metrics = YOLO26Loss()(outputs, _target(), epoch=0, epochs=2)
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
    loss, _ = YOLO26Loss()(outputs, _target(2), epoch=0, epochs=2)
    loss.backward()
    sima_freeze_qat(prepared)
    assert bool(prepared.qat_frozen.item())
