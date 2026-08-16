"""Regression tests for deterministic example artifact discovery."""

import os
from pathlib import Path

from examples.imagenet.export_onnx import (
    find_latest_checkpoint as find_latest_imagenet_export_checkpoint,
)
from examples.imagenet.train import (
    find_latest_checkpoint as find_latest_imagenet_train_checkpoint,
)
from examples.mnist.export_onnx import (
    find_latest_checkpoint as find_latest_mnist_export_checkpoint,
)
from examples.mnist.train import (
    find_latest_checkpoint as find_latest_mnist_checkpoint,
)
from sima_qat.misc import find_latest_file_string


def test_latest_file_discovery_is_recursive_and_deterministic(
    tmp_path: Path,
) -> None:
    older = tmp_path / "first" / "model.onnx"
    newer = tmp_path / "second" / "model.onnx"
    ignored = tmp_path / "second" / "checkpoint.ckpt"
    onnx_backup = tmp_path / "second" / "model.onnx.bak"
    checkpoint_partial = tmp_path / "second" / "checkpoint.ckpt.tmp"
    for path in (older, newer, ignored, onnx_backup, checkpoint_partial):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"artifact")

    os.utime(older, ns=(1_000_000_000, 1_000_000_000))
    os.utime(newer, ns=(2_000_000_000, 2_000_000_000))
    os.utime(ignored, ns=(3_000_000_000, 3_000_000_000))
    os.utime(onnx_backup, ns=(4_000_000_000, 4_000_000_000))
    os.utime(checkpoint_partial, ns=(5_000_000_000, 5_000_000_000))

    assert find_latest_file_string(tmp_path) == str(newer)
    assert find_latest_file_string(tmp_path, ".ckpt") == str(ignored)
    assert find_latest_file_string(tmp_path / "missing") is None


def test_checkpoint_discovery_is_model_and_mode_scoped(tmp_path: Path) -> None:
    checkpoints = tmp_path / "checkpoints"
    mnist_qat = checkpoints / "mnist_qat_classifier_epoch=0.ckpt"
    mnist_float = checkpoints / "mnist_float_classifier_epoch=0.ckpt"
    unrelated_mnist = checkpoints / "unrelated_epoch=9.ckpt"
    imagenet_qat = checkpoints / "imagenet_resnet18_qat_classifier_epoch=0.ckpt"
    imagenet_float = checkpoints / "imagenet_resnet18_float_classifier_epoch=0.ckpt"
    other_model = checkpoints / "imagenet_resnet50_qat_classifier_epoch=0.ckpt"
    partial = checkpoints / "imagenet_resnet18_qat_classifier_epoch=9.ckpt.tmp"
    artifacts = (
        mnist_qat,
        mnist_float,
        unrelated_mnist,
        imagenet_qat,
        imagenet_float,
        other_model,
        partial,
    )
    for index, path in enumerate(artifacts, start=1):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"checkpoint")
        timestamp = index * 1_000_000_000
        os.utime(path, ns=(timestamp, timestamp))

    assert find_latest_mnist_checkpoint(checkpoints, use_qat=True) == mnist_qat
    assert find_latest_mnist_checkpoint(checkpoints, use_qat=False) == mnist_float
    assert find_latest_mnist_export_checkpoint(checkpoints) == mnist_float
    assert (
        find_latest_imagenet_train_checkpoint(
            checkpoints, "resnet18", use_qat=True
        )
        == imagenet_qat
    )
    assert (
        find_latest_imagenet_train_checkpoint(
            checkpoints, "resnet18", use_qat=False
        )
        == imagenet_float
    )
    assert (
        find_latest_imagenet_train_checkpoint(checkpoints, "resnet18")
        == imagenet_float
    )
    assert (
        find_latest_imagenet_export_checkpoint(checkpoints, "resnet18")
        == imagenet_float
    )
