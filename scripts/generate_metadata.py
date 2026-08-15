#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def human_mb(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):.1f} MB"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate metadata.json for QAT sima-cli installation.")
    parser.add_argument("--artifacts-dir", required=True, help="Directory containing bundle artifacts.")
    parser.add_argument("--output", required=True, help="Path for generated metadata.json.")
    parser.add_argument("--name", default="sima-neat-qat")
    parser.add_argument("--version", required=True)
    parser.add_argument("--release", default="stable")
    parser.add_argument("--description", default="SiMa.ai NEAT QAT extension")
    parser.add_argument("--host-os", default="linux")
    parser.add_argument("--installer-script", default="install_qat_wheels.sh")
    parser.add_argument("--smoke-test", default="smoke_test_qat.py")
    parser.add_argument("--source-manifest", default="source.json")
    parser.add_argument("--wheel-manifest", default="manifest.txt")
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    if not artifacts_dir.is_dir():
        raise SystemExit(f"artifacts-dir does not exist: {artifacts_dir}")

    wheel_artifacts = sorted(p for p in artifacts_dir.iterdir() if p.is_file() and p.suffix == ".whl")
    if not wheel_artifacts:
        raise SystemExit(f"No wheel artifacts found in {artifacts_dir}")

    installer = artifacts_dir / args.installer_script
    if not installer.is_file():
        raise SystemExit(f"Installer script not found: {installer}")

    wheel_manifest = artifacts_dir / args.wheel_manifest
    wheel_manifest.write_text("".join(f"{path.name}\n" for path in wheel_artifacts), encoding="utf-8")

    smoke_test = artifacts_dir / args.smoke_test
    if not smoke_test.is_file():
        raise SystemExit(f"Smoke test not found: {smoke_test}")

    resources = [path.name for path in wheel_artifacts] + [wheel_manifest.name, installer.name, smoke_test.name]
    source_manifest = artifacts_dir / args.source_manifest
    if source_manifest.is_file():
        resources.append(source_manifest.name)

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
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
