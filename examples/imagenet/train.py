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
import argparse
from argparse import ArgumentTypeError
from pathlib import Path
from typing import Any

import pytorch_lightning as L
import torch
import torchvision.datasets as datasets
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torchvision import transforms

from .imagenet_dataset import (
    apply_imagenet_target_transform,
    limit_samples_by_class,
    set_dataset_samples,
)
from .imagenet_lit import ImageNet_Model_Trainer


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_OUTPUT_DIR = _REPO_ROOT / "build" / "examples" / "imagenet"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ArgumentTypeError(f"expected a positive integer, got {value}")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise ArgumentTypeError(f"expected a nonnegative integer, got {value}")
    return parsed


def _require_split_dir(data_path: Path, split: str) -> Path:
    split_dir = data_path / split
    if not split_dir.is_dir():
        raise FileNotFoundError(
            f"ImageNet split directory not found: {split_dir}. "
            f"Expected a dataset layout like {data_path}/train and {data_path}/val."
        )
    return split_dir


def _checkpoint_hyperparameters(checkpoint_path: Path) -> dict[str, Any]:
    print(
        "SECURITY: PyTorch Lightning checkpoints use pickle; "
        f"load only a trusted file: {checkpoint_path}"
    )
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:  # torch < 2.6 does not require an explicit policy.
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    hyperparameters = checkpoint.get("hyper_parameters", {})
    return hyperparameters if isinstance(hyperparameters, dict) else {}


def _verify_checkpoint(
    checkpoint_path: Path,
    expected_model: str,
    expected_use_qat: bool,
) -> None:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    expected_prefixes = (
        f"imagenet_{expected_model}_qat_classifier_",
        f"imagenet_{expected_model}_float_classifier_",
    )
    hyperparameters = _checkpoint_hyperparameters(checkpoint_path)
    saved_model = hyperparameters.get("model")
    if saved_model is not None and saved_model != expected_model:
        raise ValueError(
            f"Checkpoint {checkpoint_path} contains model {saved_model!r}, "
            f"not requested model {expected_model!r}."
        )
    if saved_model is None and not checkpoint_path.name.startswith(expected_prefixes):
        raise ValueError(
            f"Cannot verify that checkpoint {checkpoint_path} belongs to "
            f"model {expected_model!r}."
        )
    saved_use_qat = hyperparameters.get("use_qat")
    if saved_use_qat is not None and bool(saved_use_qat) != expected_use_qat:
        saved_mode = "QAT" if saved_use_qat else "float"
        requested_mode = "QAT" if expected_use_qat else "float"
        raise ValueError(
            f"Checkpoint {checkpoint_path} was created in {saved_mode} mode, "
            f"but this command requests {requested_mode} mode. Match the original "
            "mode by adding or removing --disable-qat."
        )


def find_latest_checkpoint(
    root_path: Path,
    model_name: str,
    use_qat: bool | None = None,
) -> Path | None:
    """Return the newest model checkpoint, optionally constrained by mode."""
    modes = ("qat", "float") if use_qat is None else ("qat" if use_qat else "float",)
    prefixes = tuple(
        f"imagenet_{model_name}_{mode}_classifier_" for mode in modes
    )
    candidates = [
        path
        for path in root_path.expanduser().rglob("*.ckpt")
        if path.name.startswith(prefixes)
    ]
    return max(
        candidates,
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
        default=None,
    )


def get_train_dataloader(
    data_path: Path,
    batch_size: int,
    samples_limit: int,
    workers: int,
    crop_size: int,
    pin_memory: bool,
) -> DataLoader:
    """Create the ImageNet training loader from ``data_path/train``."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    train_dataset = datasets.ImageFolder(
        _require_split_dir(data_path, "train"),
        transforms.Compose(
            [
                transforms.RandomResizedCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    apply_imagenet_target_transform(train_dataset)
    set_dataset_samples(
        train_dataset,
        limit_samples_by_class(train_dataset.samples, samples_limit),
    )
    return DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
    )


def get_val_dataloader(
    data_path: Path,
    batch_size: int,
    workers: int,
    resize_size: int,
    crop_size: int,
    pin_memory: bool,
) -> DataLoader:
    """Create the ImageNet validation loader from ``data_path/val``."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    val_dataset = datasets.ImageFolder(
        _require_split_dir(data_path, "val"),
        transforms.Compose(
            [
                transforms.Resize(resize_size),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )
    apply_imagenet_target_transform(val_dataset)
    return DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
    )


def run_train(args: argparse.Namespace) -> None:
    """Run ImageNet training, optionally resuming complete Lightning state."""
    if args.epochs <= 0 or args.batch <= 0 or args.samples_limit <= 0:
        raise ValueError("epochs, batch, and samples-limit must be positive")
    if args.workers < 0:
        raise ValueError("workers must be nonnegative")

    use_cuda = args.device == "cuda"
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")

    output_dir = Path(args.output_dir).expanduser().resolve()
    checkpoint_dir = output_dir / "checkpoints"
    lightning_dir = output_dir / "lightning"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    lightning_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data).expanduser().resolve()
    _require_split_dir(data_dir, "train")
    _require_split_dir(data_dir, "val")
    L.seed_everything(42, workers=True)

    checkpoint_path: Path | None = None
    if args.resume:
        checkpoint_path = find_latest_checkpoint(
            checkpoint_dir, args.model, use_qat=not args.disable_qat
        )
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"No checkpoint for model {args.model!r} found under "
                f"{checkpoint_dir}; run training first or omit --resume."
            )
        checkpoint_path = checkpoint_path.resolve()
        _verify_checkpoint(
            checkpoint_path,
            expected_model=args.model,
            expected_use_qat=not args.disable_qat,
        )
        print(f"Resuming complete Lightning state from: {checkpoint_path}")

    classifier = ImageNet_Model_Trainer(
        model=args.model,
        export_on_end=args.export_on_end,
        use_qat=not args.disable_qat,
        batch_size=args.batch,
        device_train=args.device,
        output_dir=output_dir,
        # A resumed checkpoint provides every tensor; never download weights again.
        weights=None if checkpoint_path else args.weights,
    )
    if classifier.use_qat:
        # Prepare before Lightning creates the optimizer or restores its state.
        classifier._prepare_qat()
    train_loader = get_train_dataloader(
        data_dir,
        args.batch,
        args.samples_limit,
        workers=args.workers,
        crop_size=224,
        pin_memory=use_cuda,
    )
    val_loader = get_val_dataloader(
        data_dir,
        args.batch,
        workers=args.workers,
        resize_size=256,
        crop_size=224,
        pin_memory=use_cuda,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=(
            f"imagenet_{args.model}_"
            f"{'qat' if classifier.use_qat else 'float'}_classifier_{{epoch}}"
        ),
        every_n_epochs=1,
        save_top_k=-1,
        verbose=True,
    )
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if use_cuda else "cpu",
        devices=1,
        default_root_dir=lightning_dir,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(
        model=classifier,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=str(checkpoint_path) if checkpoint_path else None,
    )
    trainer.validate(model=classifier, dataloaders=val_loader)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ImageNet using QAT")
    parser.add_argument("-e", "--epochs", type=_positive_int, default=10, help="Total epochs to train")
    parser.add_argument("-b", "--batch", type=_positive_int, default=1, help="Batch size")
    parser.add_argument(
        "-d",
        "--data",
        type=Path,
        required=True,
        help="ImageFolder dataset root containing train/ and val/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="Directory for checkpoints, logs, graphs, and ONNX exports",
    )
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--model", default="resnet18", help="Torchvision ImageNet model name")
    parser.add_argument(
        "--weights",
        default="none",
        help="Fresh-training weights: DEFAULT, none, or a model-specific enum member",
    )
    parser.add_argument(
        "--samples-limit",
        type=_positive_int,
        default=1281167,
        help="Maximum class-balanced training samples",
    )
    parser.add_argument(
        "--workers",
        type=_nonnegative_int,
        default=4,
        help="DataLoader worker processes",
    )
    parser.add_argument("--export-on-end", action="store_true", help="Export ONNX at training end")
    parser.add_argument("--disable-qat", action="store_true", help="Train and export a float model")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume model, optimizer, scheduler, epoch, and step state from the latest model checkpoint",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_train(get_args())
