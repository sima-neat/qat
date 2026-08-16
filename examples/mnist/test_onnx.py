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
"""Validate a generated MNIST ONNX model with CPU ONNX Runtime."""

import logging
import time
from argparse import ArgumentParser, ArgumentTypeError, Namespace
from pathlib import Path
from typing import Dict

import numpy as np
import onnx
import onnxruntime
import pytorch_lightning as L
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm

from sima_qat.misc import find_latest_file_string


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_DIR = _REPO_ROOT / "data" / "mnist"
_DEFAULT_EXPORT_DIR = _REPO_ROOT / "build" / "examples" / "mnist" / "exports"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ArgumentTypeError(f"expected a positive integer, got {value}")
    return parsed


def _accuracy(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise ArgumentTypeError("accuracy must be between 0.0 and 1.0")
    return parsed


def _create_cpu_session(onnx_path: Path) -> onnxruntime.InferenceSession:
    return onnxruntime.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )


def validate_model_and_input(onnx_path: Path, input_: np.ndarray) -> list[np.ndarray]:
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    ort_session = _create_cpu_session(onnx_path)

    input_t = ort_session.get_inputs()[0]
    logging.info("Input %s, shape: %s", input_t.name, input_t.shape)
    output_t = ort_session.get_outputs()[0]
    logging.info("Output %s, shape: %s", output_t.name, output_t.shape)
    return ort_session.run(None, {input_t.name: input_})


class MNISTIterator:
    """Indexable view of the torchvision MNIST test split."""

    def __init__(
        self,
        ds_root: str | Path,
        download: bool,
        samples_limit: int | None = None,
    ) -> None:
        root = Path(ds_root).expanduser().resolve()
        if download:
            root.mkdir(parents=True, exist_ok=True)
        elif not root.is_dir():
            raise FileNotFoundError(
                f"MNIST dataset cache not found: {root}. Pass --download to create it."
            )
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        self.mnist_dataset_test = MNIST(
            download=download,
            root=str(root),
            transform=transform,
            train=False,
        )
        self.samples_limit = samples_limit

    def __len__(self) -> int:
        length = len(self.mnist_dataset_test)
        return min(length, self.samples_limit) if self.samples_limit else length

    def __getitem__(self, index: int) -> Dict:
        sample, target = self.mnist_dataset_test[index]
        return {"sample": sample, "gt": target}


def run_accuracy_test(
    ort_session: onnxruntime.InferenceSession,
    dataset_test: object,
) -> float:
    """Run the dataset against the model and return top-1 accuracy in [0, 1]."""
    input_t = ort_session.get_inputs()[0]
    sample_count = len(dataset_test)
    if sample_count <= 0:
        raise ValueError("Accuracy validation requires at least one sample")

    correct = 0
    inf_start = time.perf_counter()
    for index in tqdm(range(sample_count)):
        sample = dataset_test[index]
        image = sample["sample"]
        if hasattr(image, "detach"):
            image = image.detach().cpu().numpy()
        nn_input = np.expand_dims(image, axis=0)
        outputs = ort_session.run(None, {input_t.name: nn_input})
        prediction = int(np.argmax(outputs[0][0]))
        correct += int(prediction == sample["gt"])

    elapsed = time.perf_counter() - inf_start
    logging.info("Throughput: %.2f samples/s", sample_count / elapsed)
    return correct / sample_count


def get_args() -> Namespace:
    """Get CLI arguments."""
    recent_onnx_file = find_latest_file_string(str(_DEFAULT_EXPORT_DIR))
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--onnx",
        type=Path,
        default=recent_onnx_file,
        help="ONNX model to validate (defaults to the latest example export)",
    )
    parser.add_argument(
        "--dsroot",
        type=Path,
        default=_DEFAULT_DATA_DIR,
        help="MNIST dataset cache directory",
    )
    parser.add_argument("--download", action="store_true", help="Download missing MNIST data")
    parser.add_argument(
        "--samples-limit",
        type=_positive_int,
        default=None,
        help="Maximum number of validation samples",
    )
    parser.add_argument(
        "--min-accuracy",
        type=_accuracy,
        default=None,
        help="Fail with a nonzero exit status when top-1 accuracy is below this value",
    )
    parser.add_argument("-v", "--verbosity", default="INFO", help="Logging verbosity level")
    return parser.parse_args()


def main() -> float:
    args = get_args()
    logging.getLogger().setLevel(args.verbosity)
    L.seed_everything(42)

    if not args.onnx:
        raise FileNotFoundError(
            f"No ONNX model found under {_DEFAULT_EXPORT_DIR}; pass --onnx explicitly."
        )
    onnx_path = Path(args.onnx).expanduser().resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    logging.info("Loading ONNX file: %s", onnx_path)

    input_ = np.random.rand(1, 1, 28, 28).astype(np.float32)
    logging.info("Testing model on input shape: %s", input_.shape)
    validate_model_and_input(onnx_path, input_)
    logging.info("ONNX validation succeeded.")

    dataset_test = MNISTIterator(
        ds_root=args.dsroot,
        download=args.download,
        samples_limit=args.samples_limit,
    )
    accuracy = run_accuracy_test(_create_cpu_session(onnx_path), dataset_test)
    logging.info("Top-1 accuracy: %.6f", accuracy)
    if args.min_accuracy is not None and accuracy < args.min_accuracy:
        raise SystemExit(
            f"Top-1 accuracy {accuracy:.6f} is below required minimum "
            f"{args.min_accuracy:.6f}"
        )
    return accuracy


if __name__ == "__main__":
    main()
