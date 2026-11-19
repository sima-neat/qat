"""Keep positive division guards on the INT8 activation grid.

A guard such as ``sum + 1e-6`` or ``clamp_min(norm, 1e-6)`` keeps a
denominator positive in floating point. During QAT the denominator's fake
quantizer rounds the tiny epsilon to the zero code, so ``0 / 0`` becomes NaN.
For a recognized guard on a nonnegative grid, floor the fake-quantized
denominator at one grid step: the smallest positive value that grid represents.
Finalization replaces the floor with the frozen step, which ONNX exports as a
scalar Clip.
"""

from typing import Optional

import torch
from torch.ao.quantization.fake_quantize import FakeQuantizeBase
from torch.fx import GraphModule, Node

_aten = torch.ops.aten
_GUARD_META = "sima_denominator_guard"
_DIVISIONS = (_aten.div.Tensor,)
_SHAPE_ONLY = (
    _aten.expand.default,
    _aten.expand_as.default,
    _aten.flatten.using_ints,
    _aten.permute.default,
    _aten.reshape.default,
    _aten.squeeze.dim,
    _aten.transpose.int,
    _aten.unsqueeze.default,
    _aten.view.default,
)
_MAX_SEARCH_DEPTH = 8


def denominator_floor(scale: torch.Tensor, zero_point: torch.Tensor, quant_min: int) -> torch.Tensor:
    """Return one grid step on a nonnegative grid, and no floor otherwise."""
    nonnegative = zero_point == quant_min
    return torch.where(nonnegative, scale, torch.full_like(scale, float("-inf")))


def _fake_quantizer(gm: GraphModule, node) -> Optional[FakeQuantizeBase]:
    if not isinstance(node, Node) or node.op != "call_module":
        return None
    module = gm.get_submodule(str(node.target))
    if isinstance(module, FakeQuantizeBase) and not getattr(module, "is_per_channel", False):
        return module
    return None


def _scalar(gm: GraphModule, value) -> Optional[float]:
    while _fake_quantizer(gm, value) is not None:
        value = value.args[0]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, Node) and value.op == "get_attr":
        owner, _, name = str(value.target).rpartition(".")
        tensor = getattr(gm.get_submodule(owner) if owner else gm, name)
        if isinstance(tensor, torch.Tensor) and tensor.numel() == 1:
            return float(tensor)
    return None


def _is_positive(gm: GraphModule, value) -> bool:
    scalar = _scalar(gm, value)
    return scalar is not None and scalar > 0


def _has_positive_guard(gm: GraphModule, node) -> bool:
    """Trace a denominator back through quantizers and shape-only operations."""
    for _ in range(_MAX_SEARCH_DEPTH):
        if not isinstance(node, Node):
            return False
        if _fake_quantizer(gm, node) is not None or (
            node.op == "call_function" and node.target in _SHAPE_ONLY
        ):
            node = node.args[0]
            continue
        if node.op != "call_function":
            return False
        if node.target == _aten.add.Tensor:
            return len(node.args) >= 2 and any(_is_positive(gm, value) for value in node.args[:2])
        if node.target == _aten.clamp_min.default:
            return _is_positive(gm, node.args[1])
        if node.target == _aten.clamp.default:
            low = node.args[1] if len(node.args) > 1 else node.kwargs.get("min")
            return _is_positive(gm, low)
        return False
    return False


def insert_denominator_guards(gm: GraphModule) -> int:
    """Floor fake-quantized denominators that the model already guards."""
    inserted = 0
    for division in list(gm.graph.nodes):
        if division.op != "call_function" or division.target not in _DIVISIONS:
            continue
        denominator = division.args[1]
        quantizer = _fake_quantizer(gm, denominator)
        if quantizer is None or not _has_positive_guard(gm, denominator.args[0]):
            continue
        quant_min = int(quantizer.activation_post_process.quant_min)
        with gm.graph.inserting_before(division):
            scale = gm.graph.get_attr(f"{denominator.target}.scale")
            zero_point = gm.graph.get_attr(f"{denominator.target}.zero_point")
            floor = gm.graph.call_function(denominator_floor, (scale, zero_point, quant_min))
            guarded = gm.graph.call_function(_aten.clamp_min.Tensor, (denominator, floor))
        guarded.meta[_GUARD_META] = str(denominator.target)
        division.replace_input_with(denominator, guarded)
        inserted += 1
    if inserted:
        gm.recompile()
    return inserted


def freeze_denominator_guards(gm: GraphModule) -> None:
    """Replace each live floor with its frozen step before PT2E conversion."""
    changed = False
    for guarded in list(gm.graph.nodes):
        if _GUARD_META not in guarded.meta:
            continue
        denominator, floor = guarded.args
        quantizer = gm.get_submodule(guarded.meta[_GUARD_META])
        step = float(quantizer.scale.reshape(-1)[0])
        nonnegative = int(quantizer.zero_point.reshape(-1)[0]) == int(
            quantizer.activation_post_process.quant_min
        )
        if nonnegative:
            with gm.graph.inserting_before(guarded):
                frozen = gm.graph.call_function(_aten.clamp_min.default, (denominator, step))
            guarded.replace_all_uses_with(frozen)
        else:
            guarded.replace_all_uses_with(denominator)
        gm.graph.erase_node(guarded)
        scale, zero_point, _ = floor.args
        gm.graph.erase_node(floor)
        for attribute in (scale, zero_point):
            if not attribute.users:
                gm.graph.erase_node(attribute)
        changed = True
    if changed:
        gm.recompile()
