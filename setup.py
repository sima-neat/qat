"""Packaging metadata for the SiMa QAT distribution.

Packaging metadata reads are side-effect free and never rewrite source files.
The build command may refresh only its guarded repository build/ outputs.
"""

from pathlib import Path
import re
import shutil

import setuptools
from setuptools.command.build_py import build_py


ROOT = Path(__file__).resolve().parent
VERSION_PATTERN = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")
LEGAL_FILENAMES = (
    "LICENSE",
    "LICENSE.txt",
    "LICENSE.md",
    "NOTICE",
    "NOTICE.txt",
    "NOTICE.md",
)


class CleanBuildPy(build_py):
    """Build into a clean, repository-owned generated package directory."""

    def run(self) -> None:
        raw_build_lib = Path(self.build_lib)
        build_lib = raw_build_lib.resolve()
        build_root = (ROOT / "build").resolve()
        if raw_build_lib.is_symlink():
            raise RuntimeError(f"Refusing symlinked build_lib: {raw_build_lib}")
        if build_lib.exists():
            if build_lib == build_root or not build_lib.is_relative_to(build_root):
                raise RuntimeError(
                    f"Refusing to clean build_lib outside {build_root}: {build_lib}"
                )
            shutil.rmtree(build_lib)
        super().run()


def read_requirements(filename: str) -> list[str]:
    """Return concrete requirement lines from a repository requirements file."""
    return [
        line
        for raw_line in (ROOT / filename).read_text(encoding="utf-8").splitlines()
        if (line := raw_line.strip()) and not line.startswith("#")
    ]


def read_version() -> str:
    """Read and validate the immutable PEP 440 package version."""
    package_version = (ROOT / "VERSION.in").read_text(encoding="utf-8").strip()
    if VERSION_PATTERN.fullmatch(package_version) is None:
        raise RuntimeError(
            "VERSION.in must contain only a numeric MAJOR.MINOR.PATCH version; "
            f"found {package_version!r}"
        )
    return package_version


runtime_requirements = read_requirements("requirements.txt")
test_requirements = read_requirements("requirements-test.txt")
distill_requirements = read_requirements("requirements-distill.txt")
legal_files = [
    name
    for name in LEGAL_FILENAMES
    if (ROOT / name).is_file() and (ROOT / name).stat().st_size > 0
]
long_description = (ROOT / "README.md").read_text(encoding="utf-8")


setuptools.setup(
    name="sima-qat",
    version=read_version(),
    author="SiMa.ai",
    author_email="support@sima.ai",
    description="SiMa.ai QAT implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://sima.ai",
    packages=setuptools.find_packages(include=("sima_qat", "sima_qat.*")),
    cmdclass={"build_py": CleanBuildPy},
    install_requires=runtime_requirements,
    extras_require={
        "tests": test_requirements,
        "dev": test_requirements,
        "distill": distill_requirements,
    },
    python_requires="==3.12.3",
    license="Proprietary",
    license_files=legal_files,
    classifiers=[
        "Programming Language :: Python :: 3.12",
        "Operating System :: POSIX :: Linux",
    ],
)
