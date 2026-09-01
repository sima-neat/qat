#!/usr/bin/env python3
"""Fail closed unless a QAT extension bundle is complete and self-consistent."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath

REQUIRED_PACKAGE_MEMBERS = {
    "sima_qat/VERSION",
    "sima_qat/__init__.py",
    "sima_qat/depthart.py",
    "sima_qat/dynamic_p64.py",
    "sima_qat/misc.py",
    "sima_qat/onnx_ops.py",
    "sima_qat/qat_api.py",
    "sima_qat/session.py",
    "sima_qat/sima_quantizer.py",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_bundle(bundle: Path) -> dict[str, object]:
    if not bundle.is_dir():
        raise RuntimeError(f"QAT bundle directory does not exist: {bundle}")

    metadata_path = bundle / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("name") != "sima-neat-qat":
        raise RuntimeError("QAT bundle has an unexpected artifact name")

    resources = metadata.get("resources")
    checksums = metadata.get("resources-checksum")
    if not isinstance(resources, list) or not isinstance(checksums, dict):
        raise TypeError("QAT metadata has no resource/checksum contract")
    if set(resources) != set(checksums):
        raise RuntimeError("QAT metadata resources and checksums differ")

    for name in resources:
        if not isinstance(name, str) or PurePosixPath(name).name != name:
            raise RuntimeError(f"QAT resource must be a plain filename: {name!r}")
        path = bundle / name
        if not path.is_file():
            raise RuntimeError(f"QAT resource is missing: {name}")
        if sha256_file(path) != checksums[name]:
            raise RuntimeError(f"QAT resource checksum mismatch: {name}")

    wheels = sorted(bundle.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"QAT bundle must contain exactly one wheel, found {wheels}")
    manifest = (bundle / "manifest.txt").read_text(encoding="utf-8").splitlines()
    if manifest != [wheels[0].name]:
        raise RuntimeError(f"QAT wheel manifest changed: {manifest}")

    with zipfile.ZipFile(wheels[0]) as archive:
        members = set(archive.namelist())
    if not REQUIRED_PACKAGE_MEMBERS <= members:
        raise RuntimeError(f"QAT wheel is missing package members: {sorted(REQUIRED_PACKAGE_MEMBERS - members)}")
    forbidden = sorted(
        name for name in members if PurePosixPath(name).parts[0] in {"examples", "tests"}
    )
    if forbidden:
        raise RuntimeError(f"QAT wheel contains development-only packages: {forbidden}")

    return {
        "bundle": str(bundle.resolve()),
        "version": metadata.get("version"),
        "resources": resources,
        "wheel": wheels[0].name,
        "wheel_sha256": sha256_file(wheels[0]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_bundle(args.bundle), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
