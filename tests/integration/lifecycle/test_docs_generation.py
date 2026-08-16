"""Regression tests for the import-free API documentation generator."""

import ast
import runpy
import subprocess
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
_GENERATOR = _REPO_ROOT / "scripts" / "generate_api_docs.py"


def _generator_namespace():
    return runpy.run_path(str(_GENERATOR))


def test_generated_api_docs_are_current():
    result = subprocess.run(
        [sys.executable, str(_GENERATOR), "--check"],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_generator_honors_all_order_and_complete_function_signatures():
    namespace = _generator_namespace()
    tree = ast.parse(
        """
CONSTANT = 7

def hidden():
    pass

async def public(
    first: int,
    /,
    second: str = "value",
    *items: float,
    required: bool,
    option: int = 3,
    **metadata: object,
) -> tuple[str, int]:
    pass

__all__ = ["public", "CONSTANT"]
"""
    )

    declarations = namespace["public_declarations"](tree)
    assert [name for name, _ in declarations] == ["public", "CONSTANT"]
    assert namespace["function_signature"](declarations[0][1]) == (
        "async public(first: int, /, second: str = 'value', "
        "*items: float, required: bool, option: int = 3, "
        "**metadata: object) -> tuple[str, int]"
    )
