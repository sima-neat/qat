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
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytorch_lightning as L
import torch

from .imagenet_lit import ImageNet_Model_Trainer


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_OUTPUT_DIR = _REPO_ROOT / "build" / "examples" / "imagenet"


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


def verify_checkpoint_model(checkpoint_path: Path, model_name: str) -> None:
    """Reject an explicitly selected checkpoint for another architecture."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    hyperparameters = _checkpoint_hyperparameters(checkpoint_path)
    saved_model = hyperparameters.get("model")
    if saved_model is not None and saved_model != model_name:
        raise ValueError(
            f"Checkpoint {checkpoint_path} contains model {saved_model!r}, "
            f"not requested model {model_name!r}."
        )
    expected_prefixes = (
        f"imagenet_{model_name}_qat_classifier_",
        f"imagenet_{model_name}_float_classifier_",
    )
    if saved_model is None and not checkpoint_path.name.startswith(expected_prefixes):
        raise ValueError(
            f"Cannot verify that checkpoint {checkpoint_path} belongs to "
            f"model {model_name!r}."
        )


def find_latest_checkpoint(root_path: Path, model_name: str) -> Path | None:
    """Return the newest QAT or float checkpoint for the exact model."""
    prefixes = (
        f"imagenet_{model_name}_qat_classifier_",
        f"imagenet_{model_name}_float_classifier_",
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


def run_export(args: Namespace) -> Path:
    """Restore an ImageNet checkpoint on CPU, finalize it, and export ONNX."""
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")

    output_dir = Path(args.output_dir).expanduser().resolve()
    checkpoint_dir = output_dir / "checkpoints"
    if not args.ckpt:
        raise FileNotFoundError(
            f"No checkpoint found for model {args.model!r} under {checkpoint_dir}. "
            "Run training first or pass --ckpt /path/to/checkpoint.ckpt."
        )
    checkpoint_path = Path(args.ckpt).expanduser().resolve()
    verify_checkpoint_model(checkpoint_path, args.model)

    L.seed_everything(42)
    print(f"Loading checkpoint file on CPU: {checkpoint_path}")
    classifier = ImageNet_Model_Trainer.load_from_checkpoint(
        checkpoint_path,
        map_location="cpu",
        model=args.model,
        weights=None,
        output_dir=output_dir,
        device_train="cpu",
        export_on_end=False,
    )
    classifier.to(args.device)
    classifier._finalize_qat_model()
    suffix = "qat" if classifier.use_qat else "float"
    output_path = (
        output_dir
        / "exports"
        / f"{args.model}_checkpoint_{suffix}.onnx"
    )
    classifier.to_onnx(output_path)
    return output_path


def get_args() -> Namespace:
    """Parse exporter arguments and resolve the latest matching checkpoint."""
    parser = argparse.ArgumentParser(
        description="Export the most recent ImageNet checkpoint to ONNX"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Torchvision model name used to verify and select checkpoints",
    )
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("-c", "--ckpt", type=Path, default=None, help="Checkpoint to load")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="Training output directory containing checkpoints and exports",
    )
    args = parser.parse_args()
    if args.ckpt is None:
        args.ckpt = find_latest_checkpoint(
            Path(args.output_dir).expanduser().resolve() / "checkpoints",
            args.model,
        )
    return args


if __name__ == "__main__":
    run_export(get_args())
