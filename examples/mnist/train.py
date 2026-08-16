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
import os
from argparse import ArgumentTypeError, Namespace
from pathlib import Path
from typing import Any, Tuple

import pytorch_lightning as L
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from .mnist_lit import MNIST_Trainer
from sima_qat.misc import find_latest_file_string


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_DIR = _REPO_ROOT / "data" / "mnist"
_DEFAULT_OUTPUT_DIR = _REPO_ROOT / "build" / "examples" / "mnist"


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


def _checkpoint_hyperparameters(checkpoint_path: Path) -> dict[str, Any]:
    """Read Lightning hyperparameters without restoring tensors to an accelerator."""
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


def _verify_resume_mode(checkpoint_path: Path, expected_use_qat: bool) -> None:
    saved_use_qat = _checkpoint_hyperparameters(checkpoint_path).get("use_qat")
    if saved_use_qat is not None and bool(saved_use_qat) != expected_use_qat:
        saved_mode = "QAT" if saved_use_qat else "float"
        requested_mode = "QAT" if expected_use_qat else "float"
        raise ValueError(
            f"Checkpoint {checkpoint_path} was created in {saved_mode} mode, "
            f"but this command requests {requested_mode} mode. Match the original "
            "mode by adding or removing --disable-qat."
        )


def find_latest_checkpoint(root_path: Path, use_qat: bool) -> Path | None:
    """Return the newest checkpoint for the requested quantization mode."""
    mode = "qat" if use_qat else "float"
    latest = find_latest_file_string(
        root_path,
        tag_str=".ckpt",
        name_prefix=f"mnist_{mode}_classifier_",
    )
    return Path(latest) if latest is not None else None


class MNIST_Train(MNIST):
    """MNIST training subset that reserves the last 10,000 training samples."""

    def __init__(self, *args, **kwargs) -> None:
        self.max_train_len = kwargs.pop("max_samples", 50000)
        super().__init__(*args, **kwargs)

    @property
    def raw_folder(self) -> str:
        parent_class_name = self.__class__.__base__.__name__
        return os.path.join(self.root, parent_class_name, "raw")

    def __len__(self) -> int:
        return min(self.max_train_len, 50000)


class MNIST_Validation(MNIST):
    """MNIST validation subset backed by the last 10,000 training samples."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.test_set_start = 50000

    @property
    def raw_folder(self) -> str:
        parent_class_name = self.__class__.__base__.__name__
        return os.path.join(self.root, parent_class_name, "raw")

    def __len__(self) -> int:
        return super().__len__() - self.test_set_start

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        return super().__getitem__(index + self.test_set_start)


def run_train(args: Namespace) -> None:
    """Run MNIST training, optionally resuming the complete Lightning state."""
    if args.epochs <= 0 or args.batch <= 0 or args.samples_limit <= 0:
        raise ValueError("epochs, batch, and samples-limit must be positive")
    if args.workers < 0:
        raise ValueError("workers must be nonnegative")

    use_cuda = args.device == "cuda"
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")

    L.seed_everything(42, workers=True)
    data_dir = Path(args.data).expanduser().resolve()
    if args.download:
        data_dir.mkdir(parents=True, exist_ok=True)
    elif not data_dir.is_dir():
        raise FileNotFoundError(
            f"MNIST dataset cache not found: {data_dir}. Pass --download to create it."
        )

    output_dir = Path(args.output_dir).expanduser().resolve()
    checkpoint_dir = output_dir / "checkpoints"
    lightning_dir = output_dir / "lightning"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    lightning_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path: Path | None = None
    if args.resume:
        checkpoint_path = find_latest_checkpoint(
            checkpoint_dir, use_qat=not args.disable_qat
        )
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"No checkpoint found under {checkpoint_dir}; "
                "run training first or omit --resume."
            )
        checkpoint_path = checkpoint_path.resolve()
        _verify_resume_mode(checkpoint_path, expected_use_qat=not args.disable_qat)
        print(f"Resuming complete Lightning state from: {checkpoint_path}")

    classifier = MNIST_Trainer(
        export_on_end=args.export_on_end,
        use_qat=not args.disable_qat,
        output_dir=output_dir,
    )
    if classifier.use_qat:
        # Prepare before Lightning creates the optimizer or restores its state.
        classifier._prepare_qat()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    dataset_train = MNIST_Train(
        download=args.download,
        root=data_dir,
        transform=transform,
        max_samples=args.samples_limit,
    )
    dataset_test = MNIST_Validation(
        download=args.download,
        root=data_dir,
        transform=transform,
    )
    loader_kwargs = {
        "batch_size": args.batch,
        "pin_memory": use_cuda,
        "num_workers": args.workers,
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(dataset_train, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(dataset_test, shuffle=False, **loader_kwargs)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"mnist_{'qat' if classifier.use_qat else 'float'}_classifier_{{epoch}}",
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


def get_args() -> Namespace:
    """Parse command-line arguments for MNIST training."""
    parser = argparse.ArgumentParser(description="Train MNIST using QAT")
    parser.add_argument("-e", "--epochs", type=_positive_int, default=10, help="Total epochs to train")
    parser.add_argument("-b", "--batch", type=_positive_int, default=16, help="Batch size")
    parser.add_argument(
        "-d",
        "--data",
        type=Path,
        default=_DEFAULT_DATA_DIR,
        help="Dataset cache directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="Directory for checkpoints, logs, graphs, and ONNX exports",
    )
    parser.add_argument("--download", action="store_true", help="Download missing MNIST data")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument(
        "--workers",
        type=_nonnegative_int,
        default=1,
        help="DataLoader worker processes",
    )
    parser.add_argument(
        "--samples-limit",
        type=_positive_int,
        default=50000,
        help="Maximum number of training samples",
    )
    parser.add_argument("--export-on-end", action="store_true", help="Export ONNX at training end")
    parser.add_argument("--disable-qat", action="store_true", help="Train and export a float model")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume model, optimizer, scheduler, epoch, and step state from the latest checkpoint",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_train(get_args())
