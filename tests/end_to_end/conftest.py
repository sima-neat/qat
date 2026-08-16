from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXPORT_DIR = _REPO_ROOT / "exported_models"


@pytest.fixture(autouse=True)
def _run_each_end_to_end_test_in_export_dir(monkeypatch):
    """Run end-to-end tests from the repository's generated-output directory.

    The CIFAR training harness writes ONNX and optional graph dumps with relative
    filenames. Keep those artifacts in one predictable, gitignored location.
    Dataset caches use absolute paths derived from the test module location and
    are unaffected by this working-directory change.
    """
    _EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(_EXPORT_DIR)
