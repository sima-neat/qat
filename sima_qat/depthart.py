"""One-call DepthART dynamic-p64 QAT setup.

The QAT package deliberately uses a small duck-typed API: it does not import
DepthART or bake customer module paths into the library. A compatible scan
module exposes ``enable_depthart_dynamic_p64_scan_`` and
``freeze_depthart_dynamic_p64_``; the source DepthART SS2D implementation
provides those methods.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch import nn

from sima_qat.dynamic_p64 import (
    P64ProductFixedPoint,
    P64ResidualAddFixedPoint,
    P64ToStaticFixedPoint,
)

P64BaseProfile = Mapping[str, float]


@dataclass(frozen=True)
class DepthARTP64PreparationReport:
    stacks: tuple[str, ...]
    blocks: int
    observing: bool


def prepare_depthart_dynamic_p64(
    model: nn.Module,
    *,
    profiles: Mapping[str, Sequence[P64BaseProfile]] | None = None,
    safety_margin: float = 1.0,
    observe: bool = True,
) -> DepthARTP64PreparationReport:
    """Attach exact dynamic-p64 QAT to every compatible DepthART scan.

    ``profiles`` is keyed by ``named_modules()`` path. Missing entries use
    observer-driven base calibration; extra entries fail closed so a stale
    profile can never silently bind to a different model revision.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    requested = dict(profiles or {})
    matched: set[str] = set()
    stacks: list[str] = []
    blocks = 0
    for name, module in model.named_modules():
        enable = getattr(module, "enable_depthart_dynamic_p64_scan_", None)
        if enable is None:
            continue
        if not name:
            raise RuntimeError("DepthART scan must have a stable named_modules path")
        base_profiles = requested.get(name)
        enable(
            stack_id=name,
            bases=None if base_profiles is None else list(base_profiles),
            safety_margin=safety_margin,
            observe=observe,
        )
        matched.add(name)
        stacks.append(name)
        blocks += len(getattr(module, "depthart_dynamic_p64_steps", ()))
    if not stacks:
        raise RuntimeError(
            "model contains no module with enable_depthart_dynamic_p64_scan_()")
    stale = set(requested).difference(matched)
    if stale:
        raise ValueError(
            "DepthART p64 profile stack(s) not found: " + ", ".join(sorted(stale)))
    return DepthARTP64PreparationReport(tuple(stacks), blocks, bool(observe))


def freeze_depthart_dynamic_p64(model: nn.Module) -> dict[str, list[dict]]:
    """Freeze every dynamic base and return compiler annotation contracts."""
    contracts: dict[str, list[dict]] = {}
    for name, module in model.named_modules():
        steps = getattr(module, "depthart_dynamic_p64_steps", None)
        if steps is None or not len(steps):
            continue
        freeze = getattr(module, "freeze_depthart_dynamic_p64_", None)
        if freeze is None:
            raise RuntimeError(f"{name}: dynamic steps have no freeze API")
        freeze()
        contracts[name] = [step.compiler_contract() for step in steps]
    if not contracts:
        raise RuntimeError("no prepared DepthART dynamic-p64 stack found")
    return contracts


def enable_depthart_dynamic_p64_fake_quant(
    model: nn.Module, enabled: bool = True
) -> int:
    """Switch all prepared scans between float calibration and exact INT8."""
    changed = 0
    for module in model.modules():
        steps = getattr(module, "depthart_dynamic_p64_steps", None)
        if steps is None:
            continue
        for step in steps:
            step.enable_fake_quant(enabled)
            changed += 1
    if not changed:
        raise RuntimeError("no prepared DepthART dynamic-p64 stack found")
    return changed


def _canonical_sha256(document: object) -> str:
    raw = json.dumps(
        document, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_depthart_dynamic_p64_compile_profile(
    model: nn.Module,
    onnx_path: str | Path,
) -> dict:
    """Bind frozen QAT contracts to exact standard-op ONNX source nodes.

    The exported graph contains a real-valued Mul/Add topology shell and one
    scalar-one Mul marker for each compiler-exact micro-boundary.  Serial scan
    uses ``readout_marker``; associative scan uses ``tree_compose_marker`` and
    immediately publishes the result on a normal static INT8 Q/DQ grid.  The
    marker's registered initializer encodes the exact ``named_modules()``
    path, so this function discovers every occurrence by dataflow instead of
    fragile graph order or substring roles.
    """

    try:
        import onnx
        from onnx import helper, numpy_helper
    except ImportError as exc:
        raise RuntimeError("onnx is required to build the DepthART compile profile") from exc

    onnx_path = Path(onnx_path).resolve()
    graph = onnx.load(str(onnx_path)).graph
    nodes = list(graph.node)
    producer = {
        str(output): node for node in nodes for output in node.output if output
    }
    if len(producer) != sum(bool(output) for node in nodes for output in node.output):
        raise RuntimeError("ONNX graph contains duplicate tensor producers")
    initializers = {
        value.name: numpy_helper.to_array(value) for value in graph.initializer
    }
    consumers: dict[str, list] = {}
    for node in nodes:
        for value in node.input:
            if value:
                consumers.setdefault(str(value), []).append(node)

    steps: dict[str, object] = {}
    for name, module in model.named_modules():
        if (hasattr(module, "compiler_contract")
                and hasattr(module, "readout_marker")
                and hasattr(module, "output_quant")):
            contract = module.compiler_contract()
            steps[name] = (module, contract)
    if not steps:
        raise RuntimeError("model contains no frozen DepthART dynamic-p64 step")

    def _tokens(value: str) -> tuple[str, ...]:
        return tuple(token for token in re.split(r"[/.]+", value) if token)

    step_tokens = {name: _tokens(name) for name in steps}
    occurrences: dict[str, list] = {key: [] for key in steps}

    def _constant_through_identity(value: str):
        current = value
        for _ in range(8):
            if current in initializers:
                return initializers[current]
            source = producer.get(current)
            if source is not None and source.op_type == "Constant":
                attrs = {
                    attribute.name: helper.get_attribute_value(attribute)
                    for attribute in source.attribute
                }
                tensor = attrs.get("value")
                return None if tensor is None else numpy_helper.to_array(tensor)
            if (source is None or source.op_type != "Identity"
                    or len(source.input) != 1):
                return None
            current = source.input[0]
        raise RuntimeError(f"ONNX constant identity chain is too deep at {value!r}")

    def _provably_zero(value: str) -> bool:
        """Recognize only shape-preserving views of an exact zero constant.

        This deliberately is not a general constant folder.  Its sole purpose
        is to reject an exported tree composition whose state operand is the
        recurrence's known initial zero.  Expand/reshape-like views cannot
        change a zero value, so accepting them is a mathematical proof rather
        than a graph-name heuristic.  Unknown operators fail closed to
        ``False`` and remain ordinary live bindings.
        """

        current = value
        for _ in range(12):
            constant = _constant_through_identity(current)
            if constant is not None:
                array = np.asarray(constant)
                return bool(array.size and np.array_equal(array, np.zeros_like(array)))
            source = producer.get(current)
            if (source is None or not source.input
                    or source.op_type not in {
                        "Cast", "Expand", "Flatten", "Reshape", "Squeeze",
                        "Transpose", "Unsqueeze",
                    }):
                return False
            current = str(source.input[0])
        raise RuntimeError(f"ONNX zero-view chain is too deep at {value!r}")

    def _tree_publication(marker) -> tuple[object, object, float, int]:
        if len(marker.output) != 1:
            raise RuntimeError(
                f"{marker.name}: tree marker must have one output")
        quantizers = consumers.get(str(marker.output[0]), [])
        if len(quantizers) != 1 or quantizers[0].op_type != "QuantizeLinear":
            raise RuntimeError(
                f"{marker.name}: q+p64 must publish through exactly one "
                "QuantizeLinear before any structural consumer")
        quantize = quantizers[0]
        if len(quantize.input) < 3 or len(quantize.output) != 1:
            raise RuntimeError(
                f"{quantize.name}: tree publication requires affine INT8 QDQ")
        scale = _constant_through_identity(str(quantize.input[1]))
        zero_point = _constant_through_identity(str(quantize.input[2]))
        if (scale is None or zero_point is None
                or np.asarray(scale).size != 1
                or np.asarray(zero_point).size != 1):
            raise RuntimeError(
                f"{quantize.name}: tree publication qparams are not scalar constants")
        dequantizers = consumers.get(str(quantize.output[0]), [])
        if (len(dequantizers) != 1
                or dequantizers[0].op_type != "DequantizeLinear"):
            raise RuntimeError(
                f"{quantize.name}: tree publication lacks one matching DequantizeLinear")
        dequantize = dequantizers[0]
        if (len(dequantize.input) < 3
                or dequantize.input[1] != quantize.input[1]
                or dequantize.input[2] != quantize.input[2]):
            raise RuntimeError(
                f"{dequantize.name}: tree Q/DQ does not share exact qparams")
        return (
            quantize, dequantize,
            float(np.asarray(scale).reshape(-1)[0]),
            int(np.asarray(zero_point).reshape(-1)[0]),
        )

    marker_kinds = ("readout_marker", "tree_compose_marker")
    for node in nodes:
        marker_kind = next(
            (kind for kind in marker_kinds if kind in node.name), None)
        if node.op_type != "Mul" or len(node.input) != 2 or marker_kind is None:
            continue
        marker_inputs = []
        for value in node.input:
            one = _constant_through_identity(value)
            if (one is not None and one.shape == (1, 1, 1)
                    and np.array_equal(
                        one, np.ones((1, 1, 1), dtype=one.dtype))):
                marker_inputs.append(value)
        if len(marker_inputs) != 1:
            raise RuntimeError(
                f"{node.name}: readout marker lacks one exact scalar-one initializer")
        marker_input = marker_inputs[0]
        # Legacy ONNX export may deduplicate identical one buffers and name
        # every initializer after the first step.  Node scope remains exact.
        # Match the longest normalized named_modules() suffix and require a
        # unique winner; this is source identity, not graph execution order.
        scope = _tokens(node.name.split(f"/{marker_kind}", 1)[0])
        scores = {}
        for name, candidate in step_tokens.items():
            score = 0
            for width in range(1, min(len(scope), len(candidate)) + 1):
                if scope[-width:] == candidate[-width:]:
                    score = width
            scores[name] = score
        best = max(scores.values(), default=0)
        matches = [name for name, score in scores.items() if score == best]
        minimum = min(4, len(step_tokens[matches[0]])) if len(matches) == 1 else 4
        if best < minimum or len(matches) != 1:
            raise RuntimeError(
                f"{node.name}: readout scope does not uniquely bind a prepared "
                f"step; best_suffix_tokens={best} candidates={matches}")
        step_name = matches[0]
        dynamic_input = node.input[0] if node.input[1] == marker_input else node.input[1]
        state_add = producer.get(dynamic_input)
        if state_add is None or state_add.op_type != "Add" or len(state_add.input) != 2:
            raise RuntimeError(
                f"{node.name}: readout input is not produced by one Add")
        product = producer.get(state_add.input[0])
        if product is None or product.op_type != "Mul" or len(product.input) != 2:
            raise RuntimeError(
                f"{node.name}: state Add lhs is not produced by one Mul")
        if marker_kind == "tree_compose_marker" and _provably_zero(
                str(product.input[0])):
            raise RuntimeError(
                f"{product.name}: tree state operand is provably zero; omit "
                "the mathematically dead a*0+b composition in the PyTorch "
                "architecture (prepare_depthart_dynamic_p64 enables the "
                "identity-free first chunk automatically)")
        publication = (
            _tree_publication(node)
            if marker_kind == "tree_compose_marker" else None)
        occurrences[step_name].append(
            (marker_kind, product, state_add, node, publication))

    bindings = []
    bound_nodes: set[str] = set()
    for module_name, (_module, contract) in steps.items():
        rows = occurrences[module_name]
        if not rows:
            raise RuntimeError(f"{module_name}: no ONNX recurrence occurrences found")
        marker_types = {row[0] for row in rows}
        if len(marker_types) != 1:
            raise RuntimeError(
                f"{module_name}: serial and tree markers cannot share one export")
        marker_kind = marker_types.pop()
        if marker_kind == "tree_compose_marker":
            # Each associative composition begins and ends on the same
            # ordinary symmetric tree-state INT8 grid.  q+p64 is live only
            # inside this product/add/readout triplet, so stock structural
            # operators never need a hidden side-band ABI.
            product_lhs_base = contract["output_base"]
            product_rhs_base = contract["transition_base"]
            add_rhs_base = contract["output_base"]
            readout_scale = contract["tree_state_scale"]
            product_lhs_qparams = (
                contract["tree_state_scale"],
                contract["tree_state_zero_point"],
            )
            add_rhs_qparams = product_lhs_qparams
        else:
            product_lhs_base = contract["state_base"]
            product_rhs_base = contract["transition_base"]
            add_rhs_base = contract["injection_base"]
            readout_scale = contract["readout_scale"]
            product_lhs_qparams = None
            add_rhs_qparams = (
                contract["injection_static_scale"],
                contract["injection_static_zero_point"],
            )
        product_fixed = P64ProductFixedPoint.derive(
            product_lhs_base, product_rhs_base, contract["product_base"])
        add_fixed = P64ResidualAddFixedPoint.derive(
            contract["product_base"], add_rhs_base,
            contract["output_base"])
        readout_fixed = P64ToStaticFixedPoint.derive(
            contract["output_base"], readout_scale)
        for timestep, (
            _marker_kind, product, state_add, readout, publication,
        ) in enumerate(rows):
            names = (product.name, state_add.name, readout.name)
            if any(not name or name in bound_nodes for name in names):
                raise RuntimeError(
                    f"duplicate/empty DepthART source node binding: {names}")
            bound_nodes.update(names)
            common = {
                "stack_id": contract["stack_id"],
                "timestep": timestep,
                "block_index": contract["block_index"],
                "num_blocks": contract["num_blocks"],
            }
            bindings.append({
                **common,
                "module": module_name,
                "scan_primitive": (
                    "tree_compose" if marker_kind == "tree_compose_marker"
                    else "serial_recurrence"),
                "product": {
                    "source_node": product.name,
                    "lhs_base": product_lhs_base,
                    "rhs_base": product_rhs_base,
                    "rhs_input_scale": contract["transition_static_scale"],
                    "rhs_input_zero_point": contract[
                        "transition_static_zero_point"],
                    "output_base": contract["product_base"],
                    "multiplier": product_fixed.multiplier,
                    "shift": product_fixed.shift,
                    "pre_shift": product_fixed.pre_shift,
                },
                "state_add": {
                    "source_node": state_add.name,
                    "lhs_base": contract["product_base"],
                    "rhs_base": add_rhs_base,
                    "rhs_input_scale": add_rhs_qparams[0],
                    "rhs_input_zero_point": add_rhs_qparams[1],
                    "output_base": contract["output_base"],
                    "lhs_multiplier": add_fixed.lhs_multiplier,
                    "lhs_shift": add_fixed.lhs_shift,
                    "rhs_multiplier": add_fixed.rhs_multiplier,
                    "rhs_shift": add_fixed.rhs_shift,
                },
                "readout": {
                    "source_node": readout.name,
                    "input_base": contract["output_base"],
                    "output_scale": readout_scale,
                    "multiplier": readout_fixed.multiplier,
                    "shift": readout_fixed.shift,
                },
            })
            if product_lhs_qparams is not None:
                bindings[-1]["product"].update({
                    "lhs_input_scale": product_lhs_qparams[0],
                    "lhs_input_zero_point": product_lhs_qparams[1],
                })
                quantize, dequantize, scale, zero_point = publication
                if (not np.isclose(
                        scale, float(contract["tree_state_scale"]),
                        rtol=1.0e-7, atol=0.0)
                        or zero_point != int(contract["tree_state_zero_point"])):
                    raise RuntimeError(
                        f"{readout.name}: tree publication Q/DQ does not match "
                        f"the frozen tree-state grid: onnx=({scale}, {zero_point}) "
                        f"contract=({contract['tree_state_scale']}, "
                        f"{contract['tree_state_zero_point']})")
                bindings[-1]["publication"] = {
                    "quantize_source_node": quantize.name,
                    "dequantize_source_node": dequantize.name,
                    "scale": scale,
                    "zero_point": zero_point,
                }

    document = {
        "schema": "sima-depthart-dynamic-p64-compile-profile/v1",
        "onnx_path": str(onnx_path),
        "onnx_sha256": _file_sha256(onnx_path),
        "carrier_abi": "pow2-p64-code-as-int8-hwc16-v2",
        "bindings": bindings,
    }
    document["manifest_sha256"] = _canonical_sha256(document)
    return document


def write_depthart_dynamic_p64_compile_profile(
    model: nn.Module,
    onnx_path: str | Path,
    output_path: str | Path,
) -> dict:
    """Build and atomically write the AFE/N2A compile profile."""

    document = build_depthart_dynamic_p64_compile_profile(model, onnx_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    return document
