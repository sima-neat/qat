"""Smoke coverage for the ImageNet training example."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest


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
        samples = list(range(10))

    monkeypatch.setattr(train.datasets, "ImageFolder", lambda *args, **kwargs: Dataset())

    def data_loader(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(train, "DataLoader", data_loader)
    train.get_train_dataloader("/data/imagenet", 2, 4, workers, 224)

    assert captured["num_workers"] == workers
    assert captured["persistent_workers"] is persistent
