import os

import pytest

# Generated ONNX models (and any other relative test output) are collected here, under the
# repo root, instead of being scattered into the current working directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXPORT_DIR = os.path.join(_REPO_ROOT, 'exported_models')


@pytest.fixture(autouse=True)
def _run_each_test_in_export_dir(monkeypatch):
    """Run every test from the repo's ``exported_models/`` directory.

    Several tests (and the CIFAR training harness) export ONNX with *relative* filenames
    -- e.g. 'flatten_dropout.onnx', '<model>_model.onnx'. Those resolve against the current
    working directory, which used to be the repo root, so the artifacts littered the project.
    chdir'ing into a dedicated ``exported_models/`` dir keeps every generated model in one
    predictable, gitignored place.

    Tests that need the (large, cached) CIFAR dataset reference it via an absolute path,
    so they are unaffected by this chdir.
    """
    os.makedirs(_EXPORT_DIR, exist_ok=True)
    monkeypatch.chdir(_EXPORT_DIR)
