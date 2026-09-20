#!/usr/bin/env python3
"""Measure YOLO26n FP32 and calibrated frozen fake-INT8 COCO accuracy."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from coco import letterbox_image
from loss import distances_to_boxes, make_anchors
from model import build_yolo26n
from sima_qat import sima_freeze_qat, sima_prepare_qat_model
from train import load_portable_weights


DEFAULT_COCO = Path("/project/ml_datasets/public_datasets/coco/coco_2017")


class CocoImageDataset(Dataset):
    """Deterministic image-only COCO input with inverse letterbox metadata."""

    def __init__(
        self,
        images: str | Path,
        annotations: str | Path,
        image_size: int,
        limit: int | None = None,
        selection_seed: int | None = None,
    ) -> None:
        self.images = Path(images)
        self.annotations = Path(annotations)
        self.image_size = image_size
        if not self.images.is_dir():
            raise FileNotFoundError(f"COCO image directory not found: {self.images}")
        if not self.annotations.is_file():
            raise FileNotFoundError(f"COCO annotation JSON not found: {self.annotations}")
        with self.annotations.open("r", encoding="utf-8") as handle:
            coco = json.load(handle)
        categories = sorted(coco["categories"], key=lambda value: value["id"])
        if len(categories) != 80:
            raise ValueError(f"Expected 80 COCO categories, found {len(categories)}")
        self.category_ids = tuple(category["id"] for category in categories)
        self.records = sorted(coco["images"], key=lambda value: value["id"])
        if limit is not None:
            if limit < 1:
                raise ValueError("limit must be positive")
            if selection_seed is None or limit >= len(self.records):
                self.records = self.records[:limit]
            else:
                self.records = random.Random(selection_seed).sample(self.records, limit)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Any]]:
        record = self.records[index]
        with Image.open(self.images / record["file_name"]) as loaded:
            image = loaded.convert("RGB")
        image_tensor, scale, left, top = letterbox_image(image, self.image_size)
        return image_tensor, {
            "image_id": int(record["id"]),
            "original_width": int(record["width"]),
            "original_height": int(record["height"]),
            "scale": scale,
            "left": left,
            "top": top,
        }


def collate_images(
    samples: list[tuple[Tensor, dict[str, Any]]],
) -> tuple[Tensor, list[dict[str, Any]]]:
    return torch.stack([sample[0] for sample in samples]), [sample[1] for sample in samples]


def _decode_raw_boxes(
    predictions: dict[str, dict[str, Tensor | list[Tensor]]],
    strides: tuple[int, ...] = (8, 16, 32),
) -> tuple[Tensor, Tensor]:
    one2one = predictions["one2one"]
    box_distances = one2one["boxes"]
    class_logits = one2one["scores"]
    features = one2one["feats"]
    if not isinstance(box_distances, Tensor) or not isinstance(class_logits, Tensor):
        raise TypeError("YOLO26 one-to-one outputs must contain box and score tensors")
    if not isinstance(features, list):
        features = list(features)
    anchors, stride = make_anchors(features, strides)
    boxes = distances_to_boxes(box_distances.permute(0, 2, 1), anchors) * stride
    scores = class_logits.permute(0, 2, 1).sigmoid()
    return boxes, scores


def decode_end2end(
    predictions: dict[str, dict[str, Tensor | list[Tensor]]],
    strides: tuple[int, ...] = (8, 16, 32),
    max_detections: int = 300,
    confidence: float = 0.001,
) -> list[Tensor]:
    """Decode the native NMS-free one-to-one head as YOLO26 Detect does."""

    boxes, scores = _decode_raw_boxes(predictions, strides)

    first_k = min(max_detections, scores.shape[1])
    anchor_indices = scores.amax(dim=-1).topk(first_k, dim=1).indices
    selected_scores = scores.gather(
        1,
        anchor_indices[..., None].expand(-1, -1, scores.shape[-1]),
    )
    final_k = min(max_detections, selected_scores.shape[1] * selected_scores.shape[2])
    confidence_values, flat_indices = selected_scores.flatten(1).topk(final_k, dim=1)
    class_indices = flat_indices.remainder(scores.shape[-1])
    selected_anchor_positions = flat_indices.div(scores.shape[-1], rounding_mode="floor")
    final_anchor_indices = anchor_indices.gather(1, selected_anchor_positions)
    selected_boxes = boxes.gather(
        1,
        final_anchor_indices[..., None].expand(-1, -1, 4),
    )
    detections = torch.cat(
        (
            selected_boxes,
            confidence_values[..., None],
            class_indices[..., None].to(selected_boxes.dtype),
        ),
        dim=-1,
    )
    return [image[image[:, 4] > confidence] for image in detections]


def _cxx_lround(value: float) -> int:
    return math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5)


def _boxdecode_class_nms(candidates: Tensor, iou_threshold: float) -> Tensor:
    """Match genericboxdecode's integer, class-aware NMS on one image."""

    if not candidates.numel():
        return candidates
    values = candidates.tolist()
    threshold = _cxx_lround(iou_threshold * 256)
    keep = []
    by_class: dict[int, list[int]] = {}
    geometry = []
    for index, row in enumerate(values):
        class_index = int(row[5])
        by_class.setdefault(class_index, []).append(index)
        x1, y1, x2, y2 = row[:4]
        geometry.append(
            (
                _cxx_lround(x1),
                _cxx_lround(y1),
                _cxx_lround(x2),
                _cxx_lround(y2),
                _cxx_lround((x2 - x1) * (y2 - y1)),
            )
        )
    for class_indices in by_class.values():
        class_indices.sort(key=lambda index: values[index][4], reverse=True)
        suppressed = set()
        for position, index in enumerate(class_indices):
            if index in suppressed:
                continue
            keep.append(index)
            ax1, ay1, ax2, ay2, area_a = geometry[index]
            for candidate in class_indices[position + 1 :]:
                if candidate in suppressed:
                    continue
                bx1, by1, bx2, by2, area_b = geometry[candidate]
                intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(
                    0, min(ay2, by2) - max(ay1, by1)
                )
                union = area_a + area_b - intersection
                scaled_iou = 0 if union <= 0 else intersection * 256 // union
                if scaled_iou > threshold:
                    suppressed.add(candidate)
    return candidates[torch.tensor(keep, dtype=torch.long)]


def decode_boxdecode(
    predictions: dict[str, dict[str, Tensor | list[Tensor]]],
    strides: tuple[int, ...] = (8, 16, 32),
    max_detections: int = 300,
    confidence: float = 0.001,
    nms_iou: float = 0.7,
) -> list[Tensor]:
    """Emulate the current YOLO26 path in genericboxdecode-v2."""

    boxes, scores = _decode_raw_boxes(predictions, strides)
    best_scores, best_classes = scores.max(dim=-1)
    output = []
    for image_boxes, image_scores, image_classes in zip(
        boxes, best_scores, best_classes
    ):
        valid = image_scores >= confidence
        image_boxes = image_boxes[valid]
        image_scores = image_scores[valid]
        image_classes = image_classes[valid]
        count = min(max_detections, image_scores.numel())
        if count:
            image_scores, indices = image_scores.topk(count)
            image_boxes = image_boxes[indices]
            image_classes = image_classes[indices]
        candidates = torch.cat(
            (
                image_boxes,
                image_scores[:, None],
                image_classes[:, None].to(image_boxes.dtype),
            ),
            dim=-1,
        ).cpu()
        output.append(_boxdecode_class_nms(candidates, nms_iou))
    return output


def detections_to_coco(
    detections: list[Tensor],
    metadata: list[dict[str, Any]],
    category_ids: tuple[int, ...],
) -> list[dict[str, Any]]:
    results = []
    for image_detections, image in zip(detections, metadata):
        boxes = image_detections[:, :4].clone()
        boxes[:, 0::2].sub_(image["left"]).div_(image["scale"])
        boxes[:, 1::2].sub_(image["top"]).div_(image["scale"])
        boxes[:, 0::2].clamp_(0, image["original_width"])
        boxes[:, 1::2].clamp_(0, image["original_height"])
        boxes[:, 2:] -= boxes[:, :2]
        for box, score, class_index in zip(
            boxes.cpu().tolist(),
            image_detections[:, 4].cpu().tolist(),
            image_detections[:, 5].to(torch.int64).cpu().tolist(),
        ):
            results.append(
                {
                    "image_id": image["image_id"],
                    "category_id": category_ids[class_index],
                    "bbox": box,
                    "score": score,
                }
            )
    return results


def evaluate_coco(
    annotations: Path,
    predictions: Path,
    image_ids: list[int],
) -> dict[str, float]:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as error:
        raise RuntimeError(
            "COCO metrics require pycocotools; install the example requirements"
        ) from error

    ground_truth = COCO(str(annotations))
    detections = ground_truth.loadRes(str(predictions))
    evaluator = COCOeval(ground_truth, detections, "bbox")
    evaluator.params.imgIds = image_ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    names = (
        "map_50_95",
        "map_50",
        "map_75",
        "map_small",
        "map_medium",
        "map_large",
        "mar_1",
        "mar_10",
        "mar_100",
        "mar_small",
        "mar_medium",
        "mar_large",
    )
    return {name: float(value) for name, value in zip(names, evaluator.stats)}


@torch.inference_mode()
def collect_predictions(
    model: nn.Module,
    dataset: CocoImageDataset,
    device: torch.device,
    batch_size: int,
    workers: int,
    confidence: float,
    max_detections: int,
    nms_iou: float,
    log_interval: int,
) -> tuple[list[dict[str, Any]], float]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        collate_fn=collate_images,
    )
    model.eval()
    results = []
    started = time.monotonic()
    for step, (images, metadata) in enumerate(loader):
        outputs = model(images.to(device, non_blocking=True))
        decoded = decode_boxdecode(
            outputs,
            confidence=confidence,
            max_detections=max_detections,
            nms_iou=nms_iou,
        )
        results.extend(detections_to_coco(decoded, metadata, dataset.category_ids))
        if step % log_interval == 0 or step + 1 == len(loader):
            image_count = min((step + 1) * batch_size, len(dataset))
            print(f"eval step={step + 1}/{len(loader)} images={image_count}")
    return results, time.monotonic() - started


@torch.inference_mode()
def calibrate(
    model: nn.Module,
    dataset: CocoImageDataset,
    device: torch.device,
    batch_size: int,
    workers: int,
    log_interval: int,
) -> float:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        collate_fn=collate_images,
    )
    model.eval()
    started = time.monotonic()
    for step, (images, _) in enumerate(loader):
        model(images.to(device, non_blocking=True))
        if step % log_interval == 0 or step + 1 == len(loader):
            print(
                f"calibration step={step + 1}/{len(loader)} "
                f"images={min((step + 1) * batch_size, len(dataset))}"
            )
    return time.monotonic() - started


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", default="runs/yolo26n_accuracy")
    parser.add_argument("--mode", choices=("fp32", "int8-pretrain", "both"), default="both")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--confidence", type=float, default=0.001)
    parser.add_argument("--max-detections", type=int, default=300)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--calibration-limit", type=int, default=1024)
    parser.add_argument("--calibration-seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--val-images", default=str(DEFAULT_COCO / "val2017"))
    parser.add_argument(
        "--val-annotations",
        default=str(DEFAULT_COCO / "annotations/instances_val2017.json"),
    )
    parser.add_argument("--calibration-images", default=str(DEFAULT_COCO / "train2017"))
    parser.add_argument(
        "--calibration-annotations",
        default=str(DEFAULT_COCO / "annotations/instances_train2017.json"),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.image_size < 32 or args.image_size % 32:
        raise ValueError("image-size must be a positive multiple of 32")
    if args.batch_size < 1 or args.workers < 0 or args.calibration_limit < 1:
        raise ValueError("batch-size/calibration-limit must be positive and workers non-negative")
    if not 0 <= args.confidence <= 1 or not 0 <= args.nms_iou <= 1:
        raise ValueError("confidence and nms-iou must be in [0, 1]")
    if args.max_detections < 1:
        raise ValueError("max-detections must be positive")


def run_variant(
    name: str,
    model: nn.Module,
    dataset: CocoImageDataset,
    args: argparse.Namespace,
    device: torch.device,
    output: Path,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    print(f"Evaluating {name} on {len(dataset)} COCO images")
    predictions, seconds = collect_predictions(
        model,
        dataset,
        device,
        args.batch_size,
        args.workers,
        args.confidence,
        args.max_detections,
        args.nms_iou,
        args.log_interval,
    )
    prediction_path = output / f"{name}_predictions.json"
    prediction_path.write_text(json.dumps(predictions), encoding="utf-8")
    metrics = evaluate_coco(
        Path(args.val_annotations),
        prediction_path,
        [record["id"] for record in dataset.records],
    )
    result = {
        "variant": name,
        "images": len(dataset),
        "detections": len(predictions),
        "inference_seconds": seconds,
        "metrics": metrics,
        **(extra or {}),
    }
    (output / f"{name}_metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation was requested but CUDA is unavailable")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validation = CocoImageDataset(
        args.val_images,
        args.val_annotations,
        args.image_size,
        args.limit,
    )
    weights = load_portable_weights(args.weights)
    results = {}

    if args.mode in ("fp32", "both"):
        fp32 = build_yolo26n()
        fp32.load_state_dict(weights, strict=True)
        fp32.to(device).eval()
        results["fp32"] = run_variant("fp32", fp32, validation, args, device, output)
        del fp32
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.mode in ("int8-pretrain", "both"):
        calibration = CocoImageDataset(
            args.calibration_images,
            args.calibration_annotations,
            args.image_size,
            args.calibration_limit,
            args.calibration_seed,
        )
        eager = build_yolo26n()
        eager.load_state_dict(weights, strict=True)
        eager.train()
        prepared = sima_prepare_qat_model(
            eager,
            (torch.zeros(1, 3, args.image_size, args.image_size),),
            device,
            dynamic_batch=True,
        )
        del eager
        calibration_seconds = calibrate(
            prepared,
            calibration,
            device,
            args.batch_size,
            args.workers,
            args.log_interval,
        )
        sima_freeze_qat(prepared)
        results["int8_pretrain"] = run_variant(
            "int8_pretrain",
            prepared,
            validation,
            args,
            device,
            output,
            {
                "calibration_images": len(calibration),
                "calibration_seconds": calibration_seconds,
                "qat_frozen": bool(prepared.qat_frozen.item()),
            },
        )

    summary = {
        "image_size": args.image_size,
        "confidence": args.confidence,
        "nms_iou": args.nms_iou,
        "max_detections": args.max_detections,
        "results": results,
    }
    if "fp32" in results and "int8_pretrain" in results:
        fp32_metrics = results["fp32"]["metrics"]
        int8_metrics = results["int8_pretrain"]["metrics"]
        summary["int8_minus_fp32"] = {
            name: int8_metrics[name] - fp32_metrics[name]
            for name in fp32_metrics
        }
    (output / "comparison.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote comparison: {output / 'comparison.json'}")


if __name__ == "__main__":
    main()
