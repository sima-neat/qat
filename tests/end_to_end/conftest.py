import os
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_DIR = _REPO_ROOT / "data"
_CIFAR_BATCH_FILES = (
    "batches.meta",
    "data_batch_1",
    "data_batch_2",
    "data_batch_3",
    "data_batch_4",
    "data_batch_5",
    "test_batch",
)


def _read_bool_env(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise pytest.UsageError(
        f"{name} must be one of 1/0, true/false, yes/no, or on/off; got {value!r}."
    )


@pytest.fixture(scope="session")
def allow_data_download() -> bool:
    """Whether an E2E test may download missing CIFAR-10 data."""
    return _read_bool_env("SIMA_QAT_ALLOW_DATA_DOWNLOAD")


@pytest.fixture(scope="session")
def cifar_data_dir(allow_data_download: bool) -> Path:
    """Resolve the shared CIFAR-10 cache without mutating the process CWD."""
    configured = os.environ.get("SIMA_QAT_TEST_DATA_DIR")
    data_dir = Path(configured).expanduser() if configured else _DEFAULT_DATA_DIR
    data_dir = data_dir.resolve()

    batch_dir = data_dir / "cifar-10-batches-py"
    cache_complete = all((batch_dir / name).is_file() for name in _CIFAR_BATCH_FILES)
    if not cache_complete and not allow_data_download:
        pytest.skip(
            f"CIFAR-10 data is not cached under {data_dir}. Set "
            "SIMA_QAT_TEST_DATA_DIR to an existing cache or set "
            "SIMA_QAT_ALLOW_DATA_DOWNLOAD=1 to permit a download."
        )

    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir
