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

import pytorch_lightning as L
import torch

from .mnist_lit import MNIST_Trainer
from sima_qat.misc import find_latest_file_string


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_OUTPUT_DIR = _REPO_ROOT / "build" / "examples" / "mnist"


def find_latest_checkpoint(root_path: Path) -> Path | None:
    """Return the newest MNIST QAT or float checkpoint."""
    latest = find_latest_file_string(
        root_path,
        tag_str=".ckpt",
        name_prefix=("mnist_qat_classifier_", "mnist_float_classifier_"),
    )
    return Path(latest) if latest is not None else None


def run_export(args: Namespace) -> Path:
    """Restore a trained MNIST checkpoint on CPU and export it to ONNX."""
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")

    output_dir = Path(args.output_dir).expanduser().resolve()
    checkpoint_dir = output_dir / "checkpoints"
    if not args.ckpt:
        raise FileNotFoundError(
            f"No checkpoint found under {checkpoint_dir}; "
            "run training first or pass --ckpt."
        )
    checkpoint_path = Path(args.ckpt).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    L.seed_everything(42)
    print(
        "SECURITY: PyTorch Lightning checkpoints use pickle; "
        f"load only a trusted file: {checkpoint_path}"
    )
    print(f"Loading checkpoint file on CPU: {checkpoint_path}")
    classifier = MNIST_Trainer.load_from_checkpoint(
        checkpoint_path,
        map_location="cpu",
        output_dir=output_dir,
        export_on_end=False,
    )
    classifier.to(args.device)
    classifier._finalize_qat_model()
    suffix = "qat" if classifier.use_qat else "float"
    output_path = output_dir / "exports" / f"mnist_checkpoint_{suffix}.onnx"
    classifier.to_onnx(output_path)
    return output_path


def get_args() -> Namespace:
    """Parse exporter arguments and resolve the latest checkpoint by default."""
    parser = argparse.ArgumentParser(
        description="Export the most recent MNIST checkpoint to ONNX"
    )
    parser.add_argument("-c", "--ckpt", type=Path, default=None, help="Checkpoint to load")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="Training output directory containing checkpoints and exports",
    )
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()
    if args.ckpt is None:
        args.ckpt = find_latest_checkpoint(
            Path(args.output_dir).expanduser().resolve() / "checkpoints"
        )
    return args


if __name__ == "__main__":
    run_export(get_args())
