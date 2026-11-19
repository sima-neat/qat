"""Keep fake-quantized division denominators above zero.

INT8 fake quantization rounds a small denominator, such as ``sum + 1e-6``, to
the zero code, so the division returns ``0 / 0`` and training fails on NaN.
While the denominator's grid starts at zero and fake quantization is enabled,
floor it at one grid step, the smallest positive value the grid holds. Finalization freezes the floor, which
ONNX exports as a scalar Clip.
"""

import torch
from torch.ao.quantization.fake_quantize import FakeQuantizeBase
from torch.fx import GraphModule

_aten = torch.ops.aten
_FLOOR_META = "sima_denominator_floor"


def denominator_floor(
    scale: torch.Tensor, zero_point: torch.Tensor, fake_quant_enabled: torch.Tensor, quant_min: int
) -> torch.Tensor:
    """Return one grid step on an enabled nonnegative grid, and -inf (no floor) otherwise.

    The floor is zero-dimensional so a scalar denominator keeps its shape.
    """
    active = (zero_point == quant_min) & fake_quant_enabled.bool()
    return torch.where(active, scale, torch.full_like(scale, float("-inf"))).reshape(())


def insert_denominator_floors(gm: GraphModule) -> None:
    """Floor the fake-quantized denominator of every division."""
    for division in list(gm.graph.nodes):
        if division.op != "call_function" or division.target != _aten.div.Tensor:
            continue
        denominator = division.args[1]
        if getattr(denominator, "op", None) != "call_module":
            continue
        quantizer = gm.get_submodule(str(denominator.target))
        if not isinstance(quantizer, FakeQuantizeBase) or getattr(quantizer, "is_per_channel", False):
            continue
        quant_min = int(quantizer.activation_post_process.quant_min)
        with gm.graph.inserting_before(division):
            scale = gm.graph.get_attr(f"{denominator.target}.scale")
            zero_point = gm.graph.get_attr(f"{denominator.target}.zero_point")
            enabled = gm.graph.get_attr(f"{denominator.target}.fake_quant_enabled")
            floor = gm.graph.call_function(denominator_floor, (scale, zero_point, enabled, quant_min))
            floored = gm.graph.call_function(_aten.clamp_min.Tensor, (denominator, floor))
        floored.meta[_FLOOR_META] = str(denominator.target)
        division.replace_input_with(denominator, floored)
    gm.recompile()


def freeze_denominator_floors(gm: GraphModule) -> None:
    """Replace each floor with its frozen step before PT2E conversion."""
    for floored in list(gm.graph.nodes):
        if _FLOOR_META not in floored.meta:
            continue
        quantizer = gm.get_submodule(floored.meta[_FLOOR_META])
        denominator, floor = floored.args
        scale, zero_point, enabled, quant_min = floor.args
        # The converted model always quantizes, so the frozen floor ignores the switch.
        always = torch.ones_like(quantizer.fake_quant_enabled)
        step = float(denominator_floor(quantizer.scale, quantizer.zero_point, always, quant_min))
        replacement = denominator
        if step > 0:
            with gm.graph.inserting_before(floored):
                replacement = gm.graph.call_function(_aten.clamp_min.default, (denominator, step))
        floored.replace_all_uses_with(replacement)
        for node in (floored, floor, scale, zero_point, enabled):
            gm.graph.erase_node(node)
    gm.recompile()
