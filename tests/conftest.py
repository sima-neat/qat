"""Repository-wide pytest configuration helpers."""

from pathlib import Path

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    """Create the parent of a user-selected nested pytest base directory."""
    base_temp = config.getoption("basetemp")
    if base_temp:
        Path(base_temp).expanduser().resolve().parent.mkdir(
            parents=True,
            exist_ok=True,
        )
