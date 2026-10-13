"""Smoke coverage for the ImageNet training example."""

from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import sys
from pathlib import Path

import pytest
from PIL import Image
from torch.utils.data import Subset


EXAMPLE = Path(__file__).parents[2] / "examples" / "imagenet"
SPEC = importlib.util.spec_from_file_location("imagenet_train", EXAMPLE / "train.py")
assert SPEC is not None and SPEC.loader is not None
train = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(EXAMPLE))
try:
    SPEC.loader.exec_module(train)
finally:
    sys.path.remove(str(EXAMPLE))


pytestmark = pytest.mark.regression


@pytest.mark.parametrize("device,pin_memory", [("cpu", False), ("cuda", True)])
def test_training_uses_single_device_and_configurable_workers(
    monkeypatch, device, pin_memory
):
    loader_calls = []
    trainer_calls = []

    class Classifier:
        def to(self, selected_device):
            assert selected_device == device

    class Trainer:
        def __init__(self, **kwargs):
            trainer_calls.append(kwargs)

        def fit(self, **kwargs):
            pass

        def validate(self, **kwargs):
            pass

    def loader(*args, **kwargs):
        loader_calls.append((args, kwargs))
        return object()

    monkeypatch.setattr(train, "ImageNet_Model_Trainer", lambda **kwargs: Classifier())
    monkeypatch.setattr(train, "get_train_dataloader", loader)
    monkeypatch.setattr(train, "get_val_dataloader", loader)
    monkeypatch.setattr(train, "ModelCheckpoint", lambda **kwargs: object())
    monkeypatch.setattr(train.L, "Trainer", Trainer)

    train.run_train(argparse.Namespace(
        resume=False,
        disable_qat=True,
        freeze_epoch=None,
        epochs=1,
        model="resnet18",
        export_on_end=False,
        batch=2,
        device=device,
        data="/data/imagenet",
        samples_limit=8,
        workers=0,
    ))

    assert trainer_calls[0]["accelerator"] == device
    assert trainer_calls[0]["devices"] == 1
    assert len(loader_calls) == 2
    assert all(call[1]["pin_memory"] is pin_memory for call in loader_calls)
    assert loader_calls[0][0][3] == 0
    assert loader_calls[1][0][2] == 0


@pytest.mark.parametrize("workers,persistent", [(0, False), (2, True)])
def test_train_loader_handles_worker_count(monkeypatch, workers, persistent):
    captured = {}

    class Dataset:
        samples = [(f"image{index}.jpg", index // 2) for index in range(10)]

        def __len__(self):
            return len(self.samples)

    monkeypatch.setattr(train.datasets, "ImageFolder", lambda *args, **kwargs: Dataset())

    def data_loader(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(train, "DataLoader", data_loader)
    train.get_train_dataloader("/data/imagenet", 2, 4, workers, 224)

    assert captured["num_workers"] == workers
    assert captured["persistent_workers"] is persistent


def test_worker_default_and_explicit_override(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py"])
    assert train.get_args().workers == 0
    monkeypatch.setattr(sys, "argv", ["train.py", "--workers", "4"])
    assert train.get_args().workers == 4


def _imagefolder(tmp_path, counts):
    for label, count in enumerate(counts):
        directory = tmp_path / "train" / f"class{label}"
        directory.mkdir(parents=True)
        for index in range(count):
            Image.new("RGB", (8, 8), (label * 40, index * 20, 0)).save(
                directory / f"{index}.png"
            )
    return str(tmp_path)


@pytest.mark.parametrize("limit", [3, 7, 8])
def test_sample_limit_is_balanced_reproducible_and_preserves_metadata(tmp_path, limit):
    data_path = _imagefolder(tmp_path, [5, 5, 5, 5])
    loader = train.get_train_dataloader(data_path, 4, limit, 0, 8)
    subset = loader.dataset
    assert isinstance(subset, Subset)
    assert len(subset) == limit
    assert len(set(subset.indices)) == limit

    counts = Counter(int(label) for _, labels in loader for label in labels)
    assert len(counts) == min(limit, 4)
    assert max(counts.values()) - min(counts.values()) <= 1

    repeated = train.get_train_dataloader(data_path, 4, limit, 0, 8)
    assert subset.indices == repeated.dataset.indices
    source = subset.dataset
    assert len(source) == 20
    assert source.samples == source.imgs
    assert source.targets == [label for _, label in source.samples]


def test_sample_limit_redistributes_exhausted_classes(tmp_path):
    data_path = _imagefolder(tmp_path, [1, 2, 5])
    loader = train.get_train_dataloader(data_path, 4, 6, 0, 8)
    counts = Counter(int(label) for _, labels in loader for label in labels)
    assert counts == {0: 1, 1: 2, 2: 3}


@pytest.mark.parametrize("limit", [8, 10])
def test_sample_limit_keeps_full_dataset_when_limit_is_large(tmp_path, limit):
    data_path = _imagefolder(tmp_path, [4, 4])
    loader = train.get_train_dataloader(data_path, 4, limit, 0, 8)
    assert not isinstance(loader.dataset, Subset)
    assert len(loader.dataset) == 8


@pytest.mark.parametrize("limit", [0, -1])
def test_sample_limit_rejects_nonpositive_limits(tmp_path, limit):
    data_path = _imagefolder(tmp_path, [4, 4])
    with pytest.raises(ValueError, match="--samples-limit must be positive"):
        train.get_train_dataloader(data_path, 4, limit, 0, 8)
