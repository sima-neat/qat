#!/usr/bin/env python3
import argparse
from email.parser import BytesParser
import hashlib
import json
from pathlib import Path
import re
import zipfile


LEGAL_RESOURCE_NAMES = (
    "LICENSE",
    "LICENSE.txt",
    "LICENSE.md",
    "NOTICE",
    "NOTICE.txt",
    "NOTICE.md",
)


def normalized_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def validate_wheel_identity(
    wheel_path: Path,
    expected_name: str,
    expected_version: str,
) -> None:
    with zipfile.ZipFile(wheel_path) as wheel_archive:
        metadata_members = [
            name
            for name in wheel_archive.namelist()
            if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_members) != 1:
            raise SystemExit(
                f"Expected one dist-info/METADATA in {wheel_path.name}; "
                f"found {metadata_members}"
            )
        wheel_metadata = BytesParser().parsebytes(
            wheel_archive.read(metadata_members[0])
        )
        leaked_members = [
            name
            for name in wheel_archive.namelist()
            if name in {"tests", "examples"}
            or name.startswith(("tests/", "examples/"))
        ]
        if leaked_members:
            raise SystemExit(
                f"Wheel contains non-runtime packages: {leaked_members}"
            )

    actual_name = wheel_metadata.get("Name", "")
    actual_version = wheel_metadata.get("Version", "")
    if normalized_distribution_name(actual_name) != normalized_distribution_name(
        expected_name
    ):
        raise SystemExit(
            f"Wheel distribution is {actual_name!r}, expected {expected_name!r}"
        )
    if actual_version != expected_version:
        raise SystemExit(
            f"Wheel version is {actual_version!r}, expected {expected_version!r}"
        )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def human_mb(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):.1f} MB"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate metadata.json for QAT sima-cli installation."
    )
    parser.add_argument(
        "--artifacts-dir",
        required=True,
        help="Directory containing bundle artifacts.",
    )
    parser.add_argument(
        "--output", required=True, help="Path for generated metadata.json."
    )
    parser.add_argument("--name", default="sima-neat-qat")
    parser.add_argument("--version", required=True)
    parser.add_argument("--release", default="stable")
    parser.add_argument("--description", default="SiMa.ai NEAT QAT extension")
    parser.add_argument("--host-os", default="linux")
    parser.add_argument("--installer-script", default="install_qat_wheels.sh")
    parser.add_argument("--smoke-test", default="smoke_test_qat.py")
    parser.add_argument("--source-manifest", default="source.json")
    parser.add_argument("--wheel-manifest", default="manifest.txt")
    parser.add_argument("--package-name", default="sima-qat")
    parser.add_argument("--package-version", required=True)
    parser.add_argument(
        "--strict-inventory",
        action="store_true",
        help="Reject any file outside the generated bundle contract.",
    )
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    if not artifacts_dir.is_dir():
        raise SystemExit(f"artifacts-dir does not exist: {artifacts_dir}")

    wheel_artifacts = sorted(
        path
        for path in artifacts_dir.iterdir()
        if path.is_file() and path.suffix == ".whl"
    )
    if len(wheel_artifacts) != 1:
        raise SystemExit(
            f"Expected exactly one wheel artifact in {artifacts_dir}; "
            f"found {[path.name for path in wheel_artifacts]}"
        )
    validate_wheel_identity(
        wheel_artifacts[0], args.package_name, args.package_version
    )

    installer = artifacts_dir / args.installer_script
    if not installer.is_file():
        raise SystemExit(f"Installer script not found: {installer}")

    wheel_manifest = artifacts_dir / args.wheel_manifest
    wheel_manifest.write_text(
        "".join(f"{path.name}\n" for path in wheel_artifacts),
        encoding="utf-8",
    )

    smoke_test = artifacts_dir / args.smoke_test
    if not smoke_test.is_file():
        raise SystemExit(f"Smoke test not found: {smoke_test}")

    resources = [path.name for path in wheel_artifacts] + [
        wheel_manifest.name,
        installer.name,
        smoke_test.name,
    ]
    source_manifest = artifacts_dir / args.source_manifest
    if source_manifest.is_file():
        resources.append(source_manifest.name)

    for name in LEGAL_RESOURCE_NAMES:
        legal_resource = artifacts_dir / name
        if not legal_resource.exists():
            continue
        if (
            legal_resource.is_symlink()
            or not legal_resource.is_file()
            or legal_resource.stat().st_size == 0
        ):
            raise SystemExit(
                f"Legal bundle resource must be a non-empty regular file: {legal_resource}"
            )
        resources.append(name)

    checksums = {}
    total_download_bytes = 0
    for name in resources:
        path = artifacts_dir / name
        checksums[name] = sha256_file(path)
        total_download_bytes += path.stat().st_size

    metadata = {
        "name": args.name,
        "version": args.version,
        "release": args.release,
        "description": args.description,
        "platforms": [
            {"type": "host", "os": [args.host_os]},
            {"type": "palette"},
        ],
        "resources": resources,
        "resources-checksum": checksums,
        "selectable-resources": [],
        "size": {
            "download": human_mb(total_download_bytes),
            "install": human_mb(total_download_bytes),
        },
        "installation": {
            "script": f"bash ./{installer.name}",
            "post-message": (
                "[bold green]Successfully installed QAT.[/bold green]\n\n"
                "Run [green]activate-model-compiler[/green] to use QAT in the shared "
                "Model Compiler environment. Reinstall QAT after Model Compiler updates."
            ),
        },
    }

    output = Path(args.output)
    if args.strict_inventory and output.parent.resolve() != artifacts_dir.resolve():
        raise SystemExit(
            "Strict inventory requires metadata.json to be written inside "
            f"the artifacts directory: {artifacts_dir}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    if args.strict_inventory:
        expected_inventory = set(resources) | {output.name}
        actual_inventory = {path.name for path in artifacts_dir.iterdir()}
        if actual_inventory != expected_inventory:
            missing = sorted(expected_inventory - actual_inventory)
            unexpected = sorted(actual_inventory - expected_inventory)
            raise SystemExit(
                "Bundle inventory mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
