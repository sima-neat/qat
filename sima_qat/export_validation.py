"""Read-only checks for a frozen QAT export, separate from target qualification.

This module does not train, rewrite ONNX, invoke a compiler or grant an export
deployment status. In particular, a diagnostic target-reference graph is not a
compiler input just because ONNX Runtime can execute it.

It can also run as a standalone script in a CPU environment with onnx/numpy,
without importing the training API: ``python export_validation.py inspect ...``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _shape(value: Any) -> list[int | str | None]:
    return [
        dimension.dim_value
        if dimension.HasField("dim_value")
        else dimension.dim_param or None
        for dimension in value.type.tensor_type.shape.dim
    ]


def inspect_export(
    path: str | Path, *, checkpoint_sha256: str | None = None
) -> dict[str, Any]:
    """Inspect declarations and structural blockers, without changing any bytes.

    ``compiler_input_preflight`` means only that no explicit exclusion was
    found. It does not attest compiler support, precision, numerical fidelity,
    task accuracy, or hardware behavior. Those require independent checks.
    """
    import onnx

    path = Path(path)
    if checkpoint_sha256 is not None and not re.fullmatch(
        r"[0-9a-f]{64}", checkpoint_sha256
    ):
        raise ValueError("checkpoint_sha256 must be a lowercase SHA-256")
    before = sha256_file(path)
    graph = onnx.load(path, load_external_data=False)
    metadata: dict[str, str] = {}
    for entry in graph.metadata_props:
        if entry.key in metadata:
            raise ValueError(f"duplicate ONNX metadata key: {entry.key}")
        metadata[entry.key] = entry.value
    blockers = []
    eligible = metadata.get("sima_qat.compiler_eligible")
    purpose = metadata.get("sima_qat.artifact_purpose", "")
    if eligible != "true":
        blockers.append(f"compiler_eligible is {eligible!r}, not an explicit 'true'")
    if "NONDEPLOYABLE" in purpose.upper() or "DIAGNOSTIC" in purpose.upper():
        blockers.append(f"artifact purpose excludes direct deployment: {purpose}")
    checkpoint = metadata.get("sima_qat.final_checkpoint_sha256")
    if not checkpoint or not re.fullmatch(r"[0-9a-f]{64}", checkpoint):
        blockers.append("missing or malformed final checkpoint identity")
    if checkpoint_sha256 is not None and checkpoint != checkpoint_sha256:
        blockers.append(
            "export checkpoint identity differs from the requested checkpoint"
        )
    external = [
        value.name
        for value in graph.graph.initializer
        if value.data_location == onnx.TensorProto.EXTERNAL
    ]
    if external:
        blockers.append(
            "external tensor files require a separately verified file-set binding"
        )
    guards = [
        value.name
        for value in graph.graph.output
        if value.type.tensor_type.elem_type == onnx.TensorProto.BOOL
    ]
    if guards:
        blockers.append(
            "diagnostic validity guards are public outputs: " + ", ".join(guards)
        )
    onnx.checker.check_model(graph)
    after = sha256_file(path)
    if before != after:
        raise ValueError("ONNX changed during inspection")
    return {
        "schema": "sima_qat.export_inspection/v1",
        "onnx_sha256": before,
        "size_bytes": path.stat().st_size,
        "checkpoint_sha256": checkpoint,
        "purpose": purpose,
        "declared_compiler_eligible": eligible,
        "compiler_input_preflight": not blockers,
        "blockers": blockers,
        "ports": {
            role: [
                {
                    "name": value.name,
                    "shape": _shape(value),
                    "onnx_dtype": value.type.tensor_type.elem_type,
                }
                for value in values
            ]
            for role, values in (
                ("inputs", graph.graph.input),
                ("outputs", graph.graph.output),
            )
        },
        "operations": dict(
            sorted(
                Counter(
                    f"{node.domain or 'ai.onnx'}::{node.op_type}"
                    for node in graph.graph.node
                ).items()
            )
        ),
        "target_qualification": "NOT_RUN",
        "source_unchanged": True,
    }


def require_compiler_input(inspection: dict[str, Any]) -> None:
    if inspection.get("schema") != "sima_qat.export_inspection/v1":
        raise ValueError("unrecognized export inspection")
    if inspection.get("compiler_input_preflight") is not True or inspection.get(
        "blockers"
    ):
        reasons = (
            "; ".join(inspection.get("blockers", [])) or "inspection is not eligible"
        )
        raise ValueError("QAT export is not a direct compiler input: " + reasons)


def compare_output_files(expected: str | Path, actual: str | Path) -> dict[str, Any]:
    """Compare trusted captured arrays; this does not run either model.

    Reject mismatched names, shapes, dtypes, non-finite values and invalid
    domain guards. Signed zero and NaN encodings are not silently equated.
    """
    import numpy as np

    paths = [Path(expected), Path(actual)]
    before = [sha256_file(path) for path in paths]
    rows = {}
    with (
        np.load(paths[0], allow_pickle=False) as left,
        np.load(paths[1], allow_pickle=False) as right,
    ):
        if set(left.files) != set(right.files) or not left.files:
            raise ValueError("output names differ or the output set is empty")
        for name in sorted(left.files):
            a, b = left[name], right[name]
            if a.shape != b.shape or a.dtype != b.dtype:
                raise ValueError(f"{name}: output shape or dtype differs")
            if (
                a.dtype.kind not in "biuf"
                or not np.isfinite(a).all()
                or not np.isfinite(b).all()
            ):
                raise ValueError(f"{name}: unsupported dtype or non-finite output")
            if a.dtype.kind == "b" and (not a.all() or not b.all()):
                raise ValueError(f"{name}: output validity guard failed")
            byte_rows_a = (
                np.ascontiguousarray(a).view(np.uint8).reshape(a.size, a.dtype.itemsize)
            )
            byte_rows_b = (
                np.ascontiguousarray(b).view(np.uint8).reshape(b.size, b.dtype.itemsize)
            )
            differing = np.any(byte_rows_a != byte_rows_b, axis=1)
            # FLOAT64 subtraction can erase a one-unit INT64 difference above
            # 2**53. Integer captures need an exact subtraction first.
            if a.dtype.kind in "iu" and a.dtype.itemsize == 8:
                delta = np.abs(a.astype(object) - b.astype(object))
                max_abs = max(delta.flat, default=0)
            else:
                delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
                max_abs = delta.max(initial=0)
            rows[name] = {
                "shape": list(a.shape),
                "dtype": str(a.dtype),
                "values": a.size,
                "bitwise_exact": not bool(differing.any()),
                "different_bit_patterns": int(differing.sum()),
                "max_abs": float(max_abs),
            }
    if before != [sha256_file(path) for path in paths]:
        raise ValueError("captured outputs changed during comparison")
    return {
        "schema": "sima_qat.captured_output_comparison/v1",
        "expected_sha256": before[0],
        "actual_sha256": before[1],
        "outputs": rows,
        "bitwise_exact": all(row["bitwise_exact"] for row in rows.values()),
        "execution_scope": "existing_arrays_only_no_new_model_execution",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--onnx", type=Path, required=True)
    inspect.add_argument("--checkpoint-sha256")
    inspect.add_argument("--require-compiler-input", action="store_true")
    compare = commands.add_parser("compare")
    compare.add_argument("--expected", type=Path, required=True)
    compare.add_argument("--actual", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            result = inspect_export(args.onnx, checkpoint_sha256=args.checkpoint_sha256)
            print(json.dumps(result, indent=2, allow_nan=False))
            if args.require_compiler_input:
                require_compiler_input(result)
        else:
            result = compare_output_files(args.expected, args.actual)
            print(json.dumps(result, indent=2, allow_nan=False))
            return 0 if result["bitwise_exact"] else 1
    except (ValueError, OSError) as error:
        parser.exit(2, f"{error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
