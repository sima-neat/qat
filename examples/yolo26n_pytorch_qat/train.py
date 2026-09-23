#!/usr/bin/env python3
"""Fine-tune YOLO26n with SiMa QAT using a plain PyTorch training loop."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from coco import CocoDetectionDataset, collate_detection
from loss import YOLO26Loss
from model import build_yolo26n
from sima_qat import (
    sima_export_onnx,
    sima_finalize_qat_model,
    sima_freeze_qat,
    sima_prepare_qat_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights", required=True, help="Portable checkpoint from convert_checkpoint.py"
    )
    parser.add_argument("--images", required=True, help="COCO training image directory")
    parser.add_argument(
        "--annotations", required=True, help="COCO training annotation JSON"
    )
    parser.add_argument("--output", default="runs/yolo26n_qat")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--gradient-clip", type=float, default=10.0)
    parser.add_argument("--horizontal-flip", type=float, default=0.5)
    parser.add_argument(
        "--freeze-epoch",
        type=int,
        default=None,
        help="Zero-based grid-lock epoch; default is final epoch, -1 disables early locking",
    )
    parser.add_argument("--resume", help="Resume from a training checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--amp", action="store_true", help="Enable CUDA float16 autocast")
    parser.add_argument("--no-export", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> int | None:
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    if args.image_size < 32 or args.image_size % 32:
        raise ValueError("image-size must be a positive multiple of 32")
    if not 0 <= args.horizontal_flip <= 1:
        raise ValueError("horizontal-flip must be between zero and one")
    freeze_epoch = args.freeze_epoch
    if freeze_epoch is None:
        freeze_epoch = args.epochs - 1 if args.epochs > 1 else None
    elif freeze_epoch == -1:
        freeze_epoch = None
    elif not 0 <= freeze_epoch < args.epochs:
        raise ValueError("freeze-epoch must be within the training run or -1")
    return freeze_epoch


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)


def move_batch(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {
        name: value.to(device, non_blocking=True)
        for name, value in batch.items()
        if name != "image_id"
    }


def load_portable_weights(path: str | Path) -> dict[str, Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "sima-pure-pytorch-yolo26n-v1":
        raise RuntimeError(f"Not a supported portable YOLO26n checkpoint: {path}")
    if checkpoint.get("metadata", {}).get("parameters") != 2_572_280:
        raise RuntimeError("Portable checkpoint metadata does not describe YOLO26n")
    return checkpoint["state_dict"]


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    frozen: bool,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "sima-yolo26n-qat-training-v1",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "frozen": frozen,
            "args": vars(args),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    freeze_epoch = validate_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training was requested but CUDA is unavailable")

    output_directory = Path(args.output)
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "run_config.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    dataset = CocoDetectionDataset(
        args.images,
        args.annotations,
        image_size=args.image_size,
        horizontal_flip=args.horizontal_flip,
        limit=args.limit,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        collate_fn=collate_detection,
        worker_init_fn=seed_worker,
        generator=generator,
    )

    eager = build_yolo26n()
    eager.load_state_dict(load_portable_weights(args.weights), strict=True)
    eager.train()
    example = torch.zeros(1, 3, args.image_size, args.image_size)
    qat_model = sima_prepare_qat_model(
        eager,
        (example,),
        device,
    )
    del eager
    criterion = YOLO26Loss().to(device)
    optimizer = torch.optim.AdamW(
        qat_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    start_epoch = 0
    frozen = False
    if args.resume:
        resumed = torch.load(args.resume, map_location=device, weights_only=True)
        if resumed.get("format") != "sima-yolo26n-qat-training-v1":
            raise RuntimeError(f"Unsupported resume checkpoint: {args.resume}")
        qat_model.load_state_dict(resumed["model"], strict=True)
        optimizer.load_state_dict(resumed["optimizer"])
        start_epoch = int(resumed["epoch"]) + 1
        frozen = bool(resumed["frozen"])

    for epoch in range(start_epoch, args.epochs):
        qat_model.train(True)
        if freeze_epoch is not None and epoch == freeze_epoch and not frozen:
            sima_freeze_qat(qat_model)
            frozen = True
            print(f"Locked QAT grids at epoch {epoch}")
        epoch_started = time.monotonic()
        totals = {"loss": 0.0, "box": 0.0, "classification": 0.0, "l1": 0.0}
        for step, (images, batch) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=args.amp and device.type == "cuda",
            ):
                predictions = qat_model(images)
                loss, metrics = criterion(predictions, batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(qat_model.parameters(), args.gradient_clip)
            scaler.step(optimizer)
            scaler.update()

            totals["loss"] += float(loss.detach())
            for name in ("box", "classification", "l1"):
                totals[name] += float(metrics[name])
            if step % args.log_interval == 0:
                print(
                    f"epoch={epoch} step={step}/{len(loader)} "
                    f"loss={float(loss.detach()):.5f} "
                    f"box={float(metrics['box']):.5f} "
                    f"cls={float(metrics['classification']):.5f} "
                    f"l1={float(metrics['l1']):.5f}"
                )

        steps = max(len(loader), 1)
        elapsed = time.monotonic() - epoch_started
        summary = " ".join(f"{name}={value / steps:.5f}" for name, value in totals.items())
        print(f"epoch={epoch} seconds={elapsed:.1f} {summary}")
        save_checkpoint(
            output_directory / "checkpoints" / f"epoch_{epoch:03d}.pt",
            qat_model,
            optimizer,
            epoch,
            frozen,
            args,
        )

    if args.no_export:
        return
    qat_model = qat_model.cpu()
    final_model = sima_finalize_qat_model(qat_model)
    export_input = torch.zeros(1, 3, args.image_size, args.image_size)
    full_onnx = output_directory / "yolo26n_qat_training_outputs.onnx"
    output_names = [
        "o2m_boxes",
        "o2m_scores",
        "o2m_p3",
        "o2m_p4",
        "o2m_p5",
        "o2o_boxes",
        "o2o_scores",
        "o2o_p3",
        "o2o_p4",
        "o2o_p5",
    ]
    sima_export_onnx(
        final_model,
        (export_input,),
        str(full_onnx),
        input_names=["images"],
        output_names=output_names,
        device=torch.device("cpu"),
    )
    print(f"Exported full QDQ graph: {full_onnx}")


if __name__ == "__main__":
    main()
