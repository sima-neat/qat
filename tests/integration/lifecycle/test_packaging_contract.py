"""Regression tests for the shared Model Compiler packaging contract."""

import json
from pathlib import Path
import tomllib


_REPO_ROOT = Path(__file__).resolve().parents[3]
_RETIRED_ENV_FILES = (
    "setup.py",
    "setup_env.sh",
    "tox.ini",
    "requirements.txt",
    "requirements-test.txt",
    "requirements-torch28-control.txt",
    "requirements-distill.txt",
)


def test_pyproject_is_the_only_python_package_metadata_source() -> None:
    for relative_path in _RETIRED_ENV_FILES:
        assert not (_REPO_ROOT / relative_path).exists(), relative_path

    configuration = tomllib.loads(
        (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    project = configuration["project"]

    assert project["name"] == "sima-qat"
    assert project["dynamic"] == ["version"]
    assert project["requires-python"] == "==3.12.3"
    assert project["dependencies"] == [
        "numpy==1.26.4",
        "torch==2.3.1",
        "torchvision==0.18.1",
        "onnx==1.17.0",
        "onnxruntime==1.21.1",
        "pytorch-lightning==2.4.0",
    ]
    assert project["optional-dependencies"] == {
        "tests": ["pytest>=8.2,<10"],
    }
    assert configuration["tool"]["setuptools"]["dynamic"]["version"] == {
        "file": ["VERSION.in"],
    }
    assert configuration["tool"]["setuptools"]["packages"]["find"] == {
        "include": ["sima_qat", "sima_qat.*"],
        "namespaces": False,
    }


def test_source_manifest_targets_the_exact_model_compiler_stack() -> None:
    source = json.loads(
        (_REPO_ROOT / "scripts" / "source.json").read_text(encoding="utf-8")
    )

    assert source == {
        "sdk_version": "2.1.3",
        "python_version": "3.12.3",
        "shared_environment": {
            "provider": "model-compiler",
            "packages": {
                "sima-frontend": "2.1.3.dev0+master.391",
                "sima-mlc": "2.1.3.dev0+master.186",
                "numpy": "1.26.4",
                "torch": "2.3.1",
                "torchvision": "0.18.1",
                "onnx": "1.17.0",
                "onnxruntime": "1.21.1",
                "pytorch-lightning": "2.4.0",
            },
        },
    }


def test_installer_never_resolves_or_installs_shared_dependencies() -> None:
    installer = (
        _REPO_ROOT / "scripts" / "install_qat_wheels.sh"
    ).read_text(encoding="utf-8")

    assert "QAT_ALLOW_COMPATIBLE_TEST_ENV" not in installer
    assert "--dry-run" not in installer
    assert installer.count("-m pip install") == 2
    for invocation in installer.split("-m pip install")[1:]:
        option_lines = invocation.splitlines()[:8]
        assert any("--no-deps" in line for line in option_lines)
        assert any("--no-index" in line for line in option_lines)
