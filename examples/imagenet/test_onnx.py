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
"""Validate an explicitly selected ImageNet ONNX model with CPU ONNX Runtime."""

import logging
import time
from argparse import ArgumentParser, ArgumentTypeError, Namespace
from pathlib import Path
from typing import Dict

import numpy as np
import onnx
import onnxruntime
import pytorch_lightning as L
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm

from .imagenet_dataset import (
    apply_imagenet_target_transform,
    limit_samples_by_class,
    set_dataset_samples,
)


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


class ImageNetIterator:
    """Indexable view of an ImageNet-style split folder."""

    def __init__(
        self,
        ds_root: str | Path,
        split: str = "val",
        samples_limit: int | None = None,
    ) -> None:
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        split_dir = Path(ds_root).expanduser().resolve() / split
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"ImageNet split directory not found: {split_dir}. "
                f"Expected a dataset layout like {split_dir.parent}/train "
                f"and {split_dir.parent}/val."
            )

        self.imagenet_dataset = datasets.ImageFolder(split_dir, transform=transform)
        apply_imagenet_target_transform(self.imagenet_dataset)
        if samples_limit is not None:
            set_dataset_samples(
                self.imagenet_dataset,
                limit_samples_by_class(
                    self.imagenet_dataset.samples,
                    samples_limit,
                ),
            )

    def __len__(self) -> int:
        return len(self.imagenet_dataset)

    def __getitem__(self, index: int) -> Dict:
        sample, target = self.imagenet_dataset[index]
        return {"sample": sample, "gt": target}


def validate_model_and_input(onnx_path: Path, input_: np.ndarray) -> list[np.ndarray]:
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    ort_session = _create_cpu_session(onnx_path)

    input_t = ort_session.get_inputs()[0]
    logging.info("Input %s, shape: %s", input_t.name, input_t.shape)
    output_t = ort_session.get_outputs()[0]
    logging.info("Output %s, shape: %s", output_t.name, output_t.shape)
    return ort_session.run(None, {input_t.name: input_})


def run_accuracy_test(
    ort_session: onnxruntime.InferenceSession,
    dataset_test: object,
) -> float:
    """Run the dataset and return top-1 accuracy in [0, 1]."""
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
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True, help="ONNX model to validate")
    parser.add_argument(
        "--dsroot",
        type=Path,
        required=True,
        help="ImageFolder dataset root containing the selected split",
    )
    parser.add_argument("--split", default="val", help="Dataset split directory")
    parser.add_argument(
        "--samples-limit",
        type=_positive_int,
        default=None,
        help="Maximum number of class-balanced evaluation samples",
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

    onnx_path = Path(args.onnx).expanduser().resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    logging.info("Loading ONNX file: %s", onnx_path)

    input_ = np.random.rand(1, 3, 224, 224).astype(np.float32)
    logging.info("Testing model on input shape: %s", input_.shape)
    validate_model_and_input(onnx_path, input_)
    logging.info("ONNX validation succeeded.")

    dataset_test = ImageNetIterator(
        ds_root=args.dsroot,
        split=args.split,
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
