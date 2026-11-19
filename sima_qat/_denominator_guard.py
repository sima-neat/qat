"""Keep positive division guards above zero on the INT8 activation grid.

A guard such as ``sum + 1e-6`` over a nonnegative sum, or ``clamp_min(norm,
1e-6)``, keeps a denominator positive in floating point. INT8 fake quantization
rounds the small epsilon to the zero code, so the division returns ``0 / 0``
and training fails on NaN. While fake quantization is enabled, floor a guarded
denominator at one grid step, the smallest positive value the grid holds.
Divisions without such a guard are unchanged. Finalization freezes the floor,
which ONNX exports as a scalar Clip.
"""

from typing import Optional

import torch
from torch.ao.quantization.fake_quantize import FakeQuantizeBase
from torch.fx import GraphModule, Node

_aten = torch.ops.aten
_FLOOR_META = "sima_denominator_floor"
# Operations whose output keeps the sign of their first input's elements.
_SIGN_PRESERVING = (
    _aten.clone.default,
    _aten.expand.default,
    _aten.expand_as.default,
    _aten.mean.default,
    _aten.mean.dim,
    _aten.permute.default,
    _aten.reshape.default,
    _aten.squeeze.dim,
    _aten.sum.default,
    _aten.sum.dim_IntList,
    _aten.transpose.int,
    _aten.unsqueeze.default,
    _aten.view.default,
)
_NONNEGATIVE = (
    _aten._softmax.default,
    _aten.abs.default,
    _aten.exp.default,
    _aten.linalg_vector_norm.default,
    _aten.relu.default,
    _aten.sigmoid.default,
    _aten.softmax.int,
)


def denominator_floor(scale: torch.Tensor, fake_quant_enabled: torch.Tensor) -> torch.Tensor:
    """Return one grid step while fake quantization is enabled, and -inf (no floor) otherwise.

    The floor is zero-dimensional so a scalar denominator keeps its shape.
    """
    return torch.where(fake_quant_enabled.bool(), scale, torch.full_like(scale, float("-inf"))).reshape(())


def _fake_quantizer(gm: GraphModule, value) -> Optional[FakeQuantizeBase]:
    if not isinstance(value, Node) or value.op != "call_module":
        return None
    module = gm.get_submodule(str(value.target))
    if isinstance(module, FakeQuantizeBase) and not getattr(module, "is_per_channel", False):
        return module
    return None


def _source(gm: GraphModule, value):
    """Follow a value back through fake quantizers and sign-preserving operations."""
    while _fake_quantizer(gm, value) is not None or (
        isinstance(value, Node) and value.op == "call_function" and value.target in _SIGN_PRESERVING
    ):
        value = value.args[0]
    return value


def _scalar(gm: GraphModule, value) -> Optional[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, Node) and value.op == "get_attr":
        owner, _, name = str(value.target).rpartition(".")
        tensor = getattr(gm.get_submodule(owner) if owner else gm, name)
        if isinstance(tensor, torch.Tensor) and tensor.numel() == 1:
            return float(tensor)
    return None


def _is_nonnegative(gm: GraphModule, value) -> bool:
    value = _source(gm, value)
    scalar = _scalar(gm, value)
    if scalar is not None:
        return scalar >= 0
    if not isinstance(value, Node) or value.op != "call_function":
        return False
    if value.target in _NONNEGATIVE:
        return True
    if value.target == _aten.add.Tensor:
        return all(_is_nonnegative(gm, operand) for operand in value.args[:2])
    if value.target == _aten.clamp_min.default:
        return _is_nonnegative(gm, value.args[1])
    return False


def _is_guarded_positive(gm: GraphModule, value) -> bool:
    """Return whether the model bounds the value above zero with a positive constant."""
    value = _source(gm, value)
    scalar = _scalar(gm, value)
    if scalar is not None:
        return scalar > 0
    if not isinstance(value, Node) or value.op != "call_function":
        return False
    if value.target == _aten.add.Tensor:
        left, right = value.args[:2]
        return (_is_guarded_positive(gm, left) and _is_nonnegative(gm, right)) or (
            _is_nonnegative(gm, left) and _is_guarded_positive(gm, right)
        )
    if value.target == _aten.clamp_min.default:
        return _is_guarded_positive(gm, value.args[1])
    return False


def insert_denominator_floors(gm: GraphModule) -> None:
    """Floor the fake-quantized denominator of each division the model guards."""
    for division in list(gm.graph.nodes):
        if division.op != "call_function" or division.target != _aten.div.Tensor:
            continue
        denominator = division.args[1]
        if _fake_quantizer(gm, denominator) is None or not _is_guarded_positive(gm, denominator):
            continue
        with gm.graph.inserting_before(division):
            scale = gm.graph.get_attr(f"{denominator.target}.scale")
            enabled = gm.graph.get_attr(f"{denominator.target}.fake_quant_enabled")
            floor = gm.graph.call_function(denominator_floor, (scale, enabled))
            floored = gm.graph.call_function(_aten.clamp_min.Tensor, (denominator, floor))
        floored.meta[_FLOOR_META] = str(denominator.target)
        division.update_arg(1, floored)
    gm.recompile()


def freeze_denominator_floors(gm: GraphModule) -> None:
    """Replace each floor with its frozen grid step before PT2E conversion."""
    for floored in list(gm.graph.nodes):
        if _FLOOR_META not in floored.meta:
            continue
        step = float(gm.get_submodule(floored.meta[_FLOOR_META]).scale)
        denominator, floor = floored.args
        scale, enabled = floor.args
        with gm.graph.inserting_before(floored):
            frozen = gm.graph.call_function(_aten.clamp_min.default, (denominator, step))
        floored.replace_all_uses_with(frozen)
        for node in (floored, floor, scale, enabled):
            gm.graph.erase_node(node)
    gm.recompile()
