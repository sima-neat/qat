"""COCO JSON input pipeline with no Ultralytics dataset dependency."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class CocoDetectionDataset(Dataset):
    """Read COCO boxes and produce normalized YOLO-style training tensors."""

    def __init__(
        self,
        images: str | Path,
        annotations: str | Path,
        image_size: int = 640,
        horizontal_flip: float = 0.5,
        limit: int | None = None,
    ) -> None:
        self.images = Path(images)
        self.annotations = Path(annotations)
        self.image_size = image_size
        self.horizontal_flip = horizontal_flip
        if not self.images.is_dir():
            raise FileNotFoundError(f"COCO image directory not found: {self.images}")
        if not self.annotations.is_file():
            raise FileNotFoundError(f"COCO annotation JSON not found: {self.annotations}")

        with self.annotations.open("r", encoding="utf-8") as handle:
            coco = json.load(handle)
        categories = sorted(coco["categories"], key=lambda value: value["id"])
        if len(categories) != 80:
            raise ValueError(
                f"YOLO26n COCO training requires 80 categories, found {len(categories)}"
            )
        self.category_to_index = {
            category["id"]: index for index, category in enumerate(categories)
        }
        self.class_names = tuple(category["name"] for category in categories)
        records = sorted(coco["images"], key=lambda value: value["id"])
        if limit is not None:
            if limit < 1:
                raise ValueError("limit must be positive")
            records = records[:limit]
        self.records = records
        selected_ids = {record["id"] for record in records}
        grouped = defaultdict(list)
        for annotation in coco["annotations"]:
            if (
                annotation["image_id"] in selected_ids
                and not annotation.get("iscrowd", 0)
                and annotation["bbox"][2] > 0
                and annotation["bbox"][3] > 0
            ):
                grouped[annotation["image_id"]].append(annotation)
        self.boxes_by_image = dict(grouped)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Tensor]]:
        record = self.records[index]
        image_path = self.images / record["file_name"]
        with Image.open(image_path) as loaded:
            image = loaded.convert("RGB")
        original_width, original_height = image.size
        scale = min(self.image_size / original_width, self.image_size / original_height)
        resized_width = max(1, round(original_width * scale))
        resized_height = max(1, round(original_height * scale))
        resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
        left = (self.image_size - resized_width) // 2
        top = (self.image_size - resized_height) // 2
        canvas = Image.new("RGB", (self.image_size, self.image_size), (114, 114, 114))
        canvas.paste(resized, (left, top))

        boxes = []
        classes = []
        for annotation in self.boxes_by_image.get(record["id"], ()):
            x, y, width, height = annotation["bbox"]
            center_x = (x + width / 2) * scale + left
            center_y = (y + height / 2) * scale + top
            normalized = (
                center_x / self.image_size,
                center_y / self.image_size,
                width * scale / self.image_size,
                height * scale / self.image_size,
            )
            boxes.append(normalized)
            classes.append(self.category_to_index[annotation["category_id"]])

        box_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        class_tensor = torch.tensor(classes, dtype=torch.float32)
        if self.horizontal_flip and random.random() < self.horizontal_flip:
            canvas = canvas.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if box_tensor.numel():
                box_tensor[:, 0] = 1 - box_tensor[:, 0]

        byte_image = torch.frombuffer(bytearray(canvas.tobytes()), dtype=torch.uint8)
        image_tensor = (
            byte_image.view(self.image_size, self.image_size, 3)
            .permute(2, 0, 1)
            .contiguous()
            .float()
            .div_(255)
        )
        target = {
            "cls": class_tensor,
            "bboxes": box_tensor,
            "image_id": torch.tensor(record["id"], dtype=torch.int64),
        }
        return image_tensor, target


def collate_detection(
    samples: list[tuple[Tensor, dict[str, Tensor]]],
) -> tuple[Tensor, dict[str, Tensor]]:
    images = torch.stack([sample[0] for sample in samples])
    classes = []
    boxes = []
    batch_indices = []
    image_ids = []
    for index, (_, target) in enumerate(samples):
        count = target["cls"].numel()
        classes.append(target["cls"])
        boxes.append(target["bboxes"])
        batch_indices.append(torch.full((count,), index, dtype=torch.long))
        image_ids.append(target["image_id"])
    return images, {
        "cls": torch.cat(classes) if classes else torch.empty(0),
        "bboxes": torch.cat(boxes) if boxes else torch.empty((0, 4)),
        "batch_idx": (
            torch.cat(batch_indices) if batch_indices else torch.empty(0, dtype=torch.long)
        ),
        "image_id": torch.stack(image_ids),
    }
