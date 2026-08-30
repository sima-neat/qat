# **************************************************************************
#||                        SiMa.ai CONFIDENTIAL                          ||
#||   Unpublished Copyright (c) 2024 SiMa.ai, All Rights Reserved.       ||
# ***************************************************************************
"""Task-aware, export-stable refinement of strict INT8 activation ranges.

The implementation is deliberately conservative: it only edits existing
per-tensor fake-quantizer grids, never changes graph topology, and excludes
grids directly coupled to a weighted operator. Candidate edits are evaluated
transactionally and are committed only when the requested task metric improves.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from torch import nn
from torch.ao.quantization.fake_quantize import FakeQuantizeBase
from torch.fx import GraphModule, Node

from sima_qat.qat_api import _SHIFT_AWARE_OPS, _stage_learned_activation_grid

RangeEvaluator = Callable[[nn.Module, Iterable[Any]], float | Mapping[str, float]]

_WEIGHTED_OPS = _SHIFT_AWARE_OPS | {torch.ops.aten.conv_transpose2d.input}
_STATE_SPACE_TOKENS = (
    "mamba",
    "selectivescan",
    "selective_scan",
    "statespace",
    "state_space",
    "ss2d",
    "vimblock",
)
_PRODUCER_KINDS = frozenset({"multiply", "add", "concat"})
_RANGE_KINDS = frozenset({"unit_interval", "nonnegative", "any"})


@dataclass(frozen=True)
class ActivationRangeSelector:
    """Semantic selector for activation grids eligible for refinement.

    ``recurrent_products()`` is the qualified default for state-space models:
    unit-interval Multiply outputs inside recurrent/state-space module stacks,
    excluding grids directly adjacent to Conv/Linear operators.
    """

    producer_kinds: tuple[str, ...] = ("multiply",)
    range_kinds: tuple[str, ...] = ("unit_interval",)
    state_space_only: bool = True
    module_path_contains: str | None = None
    exclude_weighted_adjacent: bool = True

    def __post_init__(self) -> None:
        unknown_producers = set(self.producer_kinds) - _PRODUCER_KINDS
        if unknown_producers:
            raise ValueError(
                "unknown producer kind(s): " + ", ".join(sorted(unknown_producers))
            )
        unknown_ranges = set(self.range_kinds) - _RANGE_KINDS
        if unknown_ranges:
            raise ValueError(
                "unknown range kind(s): " + ", ".join(sorted(unknown_ranges))
            )
        if not self.producer_kinds:
            raise ValueError("producer_kinds cannot be empty")
        if not self.range_kinds:
            raise ValueError("range_kinds cannot be empty")

    @classmethod
    def recurrent_products(cls) -> ActivationRangeSelector:
        """Select positive unit-grid products in state-space/recurrent regions."""

        return cls()

    @classmethod
    def unit_interval_products(cls) -> ActivationRangeSelector:
        """Select unit-grid products throughout a model."""

        return cls(state_space_only=False)


@dataclass(frozen=True)
class ActivationRangeRefinementReport:
    """Auditable result from :func:`refine_activation_ranges`."""

    policy: str
    group_by: str
    metric: str
    higher_is_better: bool
    factors: tuple[float, ...]
    selected_fake_quantizers: int
    selection_sha256: str
    selection: tuple[Mapping[str, Any], ...]
    candidates: tuple[Mapping[str, Any], ...]
    groups: tuple[Mapping[str, Any], ...]
    baseline_metrics: Mapping[str, float]
    best_metrics: Mapping[str, float]
    best_factor: float | None
    metric_improvement: float
    committed: bool
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_activation_fake_quant(module: nn.Module) -> bool:
    return isinstance(module, FakeQuantizeBase) and module.qscheme not in (
        torch.per_channel_affine,
        torch.per_channel_symmetric,
    )


def _fake_quant_from_node(model: GraphModule, node: Any) -> FakeQuantizeBase | None:
    if not isinstance(node, Node) or node.op != "call_module":
        return None
    module = model.get_submodule(str(node.target))
    return module if _is_activation_fake_quant(module) else None


def _weighted_adjacent_fake_quantizers(model: GraphModule) -> set[int]:
    adjacent: set[int] = set()
    for node in model.graph.nodes:
        if node.op != "call_function" or node.target not in _WEIGHTED_OPS:
            continue
        for argument in node.args[:2]:
            fake_quant = _fake_quant_from_node(model, argument)
            if fake_quant is not None:
                adjacent.add(id(fake_quant))
        for user in node.users:
            fake_quant = _fake_quant_from_node(model, user)
            if fake_quant is not None:
                adjacent.add(id(fake_quant))
    return adjacent


def _producer_kind(node: Node) -> str | None:
    if node.op != "call_function":
        return None
    target = str(node.target)
    if "aten.mul." in target:
        return "multiply"
    if "aten.add." in target:
        return "add"
    if "aten.cat." in target or "aten.concat." in target:
        return "concat"
    return None


def _range_kind(fake_quant: FakeQuantizeBase) -> str:
    if fake_quant.scale.numel() != 1 or fake_quant.zero_point.numel() != 1:
        return "any"
    scale = float(fake_quant.scale.detach().cpu().item())
    zero_point = int(fake_quant.zero_point.detach().cpu().item())
    lower = (fake_quant.quant_min - zero_point) * scale
    upper = (fake_quant.quant_max - zero_point) * scale
    if math.isclose(lower, 0.0, abs_tol=max(1e-8, scale * 1e-4)):
        if math.isclose(upper, 1.0, rel_tol=1e-4, abs_tol=1e-6):
            return "unit_interval"
        return "nonnegative"
    return "any"


def _matches_range(kind: str, requested: tuple[str, ...]) -> bool:
    if "any" in requested:
        return True
    if kind == "unit_interval" and "nonnegative" in requested:
        return True
    return kind in requested


def _module_stack(node: Node) -> str:
    return str(node.meta.get("nn_module_stack", ""))


def _innermost_module_identity(node: Node) -> str:
    """Return the captured class identity without matching parent package paths."""

    stack = node.meta.get("nn_module_stack", "")
    if isinstance(stack, Mapping) and stack:
        frame = next(reversed(stack.values()))
        identity = frame[1] if isinstance(frame, (tuple, list)) and len(frame) > 1 else frame
        if isinstance(identity, type):
            return identity.__qualname__.lower()
        return str(identity).rsplit(".", 1)[-1].lower()
    return str(stack).rsplit(".", 1)[-1].lower()


def _innermost_module_path(node: Node) -> str:
    stack = node.meta.get("nn_module_stack", "")
    if isinstance(stack, Mapping) and stack:
        frame = next(reversed(stack.values()))
        if isinstance(frame, (tuple, list)) and frame:
            return str(frame[0]) or "<root>"
        return str(next(reversed(stack.keys())))
    return "<unknown>"


def _select_activation_ranges(
    model: GraphModule,
    selector: ActivationRangeSelector,
) -> tuple[dict[str, FakeQuantizeBase], tuple[dict[str, Any], ...]]:
    weighted_adjacent = (
        _weighted_adjacent_fake_quantizers(model)
        if selector.exclude_weighted_adjacent
        else set()
    )
    selected: dict[str, FakeQuantizeBase] = {}
    rows: list[dict[str, Any]] = []
    ids: dict[int, str] = {}
    for node in model.graph.nodes:
        if node.op != "call_module" or not node.args:
            continue
        fake_quant = _fake_quant_from_node(model, node)
        if fake_quant is None or id(fake_quant) in weighted_adjacent:
            continue
        producer = node.args[0]
        if not isinstance(producer, Node):
            continue
        producer_kind = _producer_kind(producer)
        if producer_kind not in selector.producer_kinds:
            continue
        stack = _module_stack(producer)
        if selector.state_space_only and not any(
            token in _innermost_module_identity(producer)
            for token in _STATE_SPACE_TOKENS
        ):
            continue
        if (
            selector.module_path_contains is not None
            and selector.module_path_contains not in stack
        ):
            continue
        range_kind = _range_kind(fake_quant)
        if not _matches_range(range_kind, selector.range_kinds):
            continue
        if fake_quant.scale.numel() != 1 or fake_quant.zero_point.numel() != 1:
            continue
        canonical = ids.get(id(fake_quant))
        if canonical is None:
            canonical = str(node.target)
            ids[id(fake_quant)] = canonical
            selected[canonical] = fake_quant
        scale = float(fake_quant.scale.detach().cpu().item())
        zero_point = int(fake_quant.zero_point.detach().cpu().item())
        rows.append(
            {
                "fake_quantizer": canonical,
                "graph_target": str(node.target),
                "producer": producer.name,
                "producer_kind": producer_kind,
                "range_kind": range_kind,
                "scale": scale,
                "zero_point": zero_point,
                "representable_min": (fake_quant.quant_min - zero_point) * scale,
                "representable_max": (fake_quant.quant_max - zero_point) * scale,
                "module_path": _innermost_module_path(producer),
            }
        )
    return selected, tuple(rows)


def _metric_mapping(value: float | Mapping[str, float]) -> dict[str, float]:
    if isinstance(value, Mapping):
        result = {str(name): float(metric) for name, metric in value.items()}
    else:
        result = {"score": float(value)}
    if not result or not all(math.isfinite(metric) for metric in result.values()):
        raise ValueError("evaluator must return finite task metrics")
    return result


def _policy_selector(
    policy: str | ActivationRangeSelector,
) -> tuple[str, ActivationRangeSelector]:
    if isinstance(policy, ActivationRangeSelector):
        return "custom", policy
    if policy in {"auto", "recurrent_products"}:
        return policy, ActivationRangeSelector.recurrent_products()
    if policy == "unit_interval_products":
        return policy, ActivationRangeSelector.unit_interval_products()
    raise ValueError(
        "policy must be 'auto', 'recurrent_products', "
        "'unit_interval_products', or ActivationRangeSelector"
    )


def _set_live_scales(
    selected: Mapping[str, FakeQuantizeBase],
    original_scales: Mapping[str, torch.Tensor],
    factor: float,
) -> None:
    with torch.no_grad():
        for name, fake_quant in selected.items():
            requested = original_scales[name] * factor
            fake_quant.scale.copy_(requested)
            if hasattr(fake_quant, "log_scale"):
                fake_quant.log_scale.copy_(requested.clamp_min(1e-12).log())


def _commit_export_stable_scales(
    selected: Mapping[str, FakeQuantizeBase],
    original_scales: Mapping[str, torch.Tensor],
    original_zero_points: Mapping[str, torch.Tensor],
    factors: Mapping[str, float],
) -> None:
    staged = []
    for name, fake_quant in selected.items():
        factor = factors[name]
        scale, zero_point, lower, upper = _stage_learned_activation_grid(
            fake_quant,
            original_scales[name] * factor,
            original_zero_points[name],
        )
        staged.append((fake_quant, scale, zero_point, lower, upper))
    with torch.no_grad():
        for fake_quant, scale, zero_point, lower, upper in staged:
            observer = fake_quant.activation_post_process
            observer.min_val.resize_(lower.shape).copy_(lower)
            observer.max_val.resize_(upper.shape).copy_(upper)
            fake_quant.scale.resize_(scale.shape).copy_(scale)
            fake_quant.zero_point.resize_(zero_point.shape).copy_(zero_point)
            if hasattr(fake_quant, "log_scale"):
                fake_quant.log_scale.copy_(scale.clamp_min(1e-12).log())


def refine_activation_ranges(
    qat_model: GraphModule,
    data: Iterable[Any],
    evaluator: RangeEvaluator,
    *,
    metric: str | None = None,
    factors: Sequence[float] = (1.0, 2.0, 4.0),
    higher_is_better: bool = True,
    min_improvement: float = 0.0,
    policy: str | ActivationRangeSelector = "auto",
    group_by: str = "module",
) -> ActivationRangeRefinementReport:
    """Search activation-grid factors and transactionally commit an improvement.

    This low-level API requires a frozen SiMa QAT graph. Most users should call
    ``qat.refine_ranges(...)`` on :class:`~sima_qat.QATSession` instead.
    ``data`` must be re-iterable because the evaluator is called once per
    candidate. The evaluator owns task preprocessing and metric computation.
    """

    if not isinstance(qat_model, GraphModule):
        raise TypeError("qat_model must be a torch.fx.GraphModule")
    if not callable(evaluator):
        raise TypeError("evaluator must be callable")
    if not bool(getattr(qat_model, "qat_frozen", torch.tensor([False])).item()):
        raise RuntimeError("range refinement requires sima_freeze_qat() first")
    if iter(data) is data:
        raise TypeError("data must be re-iterable, not a one-shot iterator")
    if not math.isfinite(min_improvement) or min_improvement < 0:
        raise ValueError("min_improvement must be finite and non-negative")
    if group_by not in {"module", "all"}:
        raise ValueError("group_by must be 'module' or 'all'")
    normalized_factors: list[float] = []
    for value in (1.0, *factors):
        factor = float(value)
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError("all factors must be finite and positive")
        if factor not in normalized_factors:
            normalized_factors.append(factor)

    policy_name, selector = _policy_selector(policy)
    selected, rows = _select_activation_ranges(qat_model, selector)
    if not selected:
        raise RuntimeError(
            "no eligible activation grids matched the range-refinement policy"
        )
    original_scales = {
        name: module.scale.detach().clone() for name, module in selected.items()
    }
    original_zero_points = {
        name: module.zero_point.detach().clone() for name, module in selected.items()
    }
    row_by_name = {str(row["fake_quantizer"]): row for row in rows}
    paths_by_name: dict[str, set[str]] = {}
    for row in rows:
        paths_by_name.setdefault(str(row["fake_quantizer"]), set()).add(
            str(row["module_path"])
        )
    shared_across_groups = {
        name: paths
        for name, paths in paths_by_name.items()
        if group_by == "module" and len(paths) > 1
    }
    if shared_across_groups:
        details = "; ".join(
            f"{name}: {', '.join(sorted(paths))}"
            for name, paths in sorted(shared_across_groups.items())
        )
        raise RuntimeError(
            "eligible fake quantizer aliases span multiple module groups; "
            f"use group_by='all' or separate the shared grid: {details}"
        )
    grouped: dict[str, list[str]] = {}
    for name in selected:
        group = (
            str(row_by_name[name]["module_path"])
            if group_by == "module"
            else "<all>"
        )
        grouped.setdefault(group, []).append(name)
    candidate_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    committed_factors: dict[str, float] = {}
    was_training = qat_model.training
    qat_model.eval()
    try:
        _set_live_scales(selected, original_scales, 1.0)
        with torch.no_grad():
            baseline_metrics = _metric_mapping(evaluator(qat_model, data))
        if metric is None:
            if len(baseline_metrics) != 1:
                raise ValueError(
                    "metric= is required when evaluator returns multiple metrics"
                )
            metric_name = next(iter(baseline_metrics))
        else:
            metric_name = metric
            if metric_name not in baseline_metrics:
                raise ValueError(
                    f"metric {metric_name!r} is absent from evaluator output"
                )
        current_metrics = baseline_metrics
        direction = 1.0 if higher_is_better else -1.0
        for group, names in grouped.items():
            modules = {name: selected[name] for name in names}
            scales = {name: original_scales[name] for name in names}
            group_candidates = [
                {"group": group, "factor": 1.0, "metrics": current_metrics}
            ]
            for factor in normalized_factors:
                if factor == 1.0:
                    continue
                _set_live_scales(modules, scales, factor)
                with torch.no_grad():
                    metrics = _metric_mapping(evaluator(qat_model, data))
                if metric_name not in metrics:
                    raise ValueError(
                        f"metric {metric_name!r} is absent from evaluator output"
                    )
                group_candidates.append(
                    {"group": group, "factor": factor, "metrics": metrics}
                )
            best = max(
                group_candidates,
                key=lambda row: direction * row["metrics"][metric_name],
            )
            improvement = direction * (
                best["metrics"][metric_name] - current_metrics[metric_name]
            )
            group_committed = best["factor"] != 1.0 and improvement > min_improvement
            if group_committed:
                _set_live_scales(modules, scales, best["factor"])
                current_metrics = best["metrics"]
                committed_factors.update({name: best["factor"] for name in names})
            else:
                _set_live_scales(modules, scales, 1.0)
            candidate_rows.extend(group_candidates)
            group_rows.append(
                {
                    "group": group,
                    "fake_quantizers": len(names),
                    "best_factor": best["factor"],
                    "metric_improvement": improvement,
                    "committed": group_committed,
                }
            )
    except Exception:
        _set_live_scales(selected, original_scales, 1.0)
        raise
    finally:
        qat_model.train(was_training)

    committed = bool(committed_factors)
    improvement = direction * (
        current_metrics[metric_name] - baseline_metrics[metric_name]
    )
    if committed:
        _commit_export_stable_scales(
            {name: selected[name] for name in committed_factors},
            {name: original_scales[name] for name in committed_factors},
            {name: original_zero_points[name] for name in committed_factors},
            committed_factors,
        )
    else:
        _set_live_scales(selected, original_scales, 1.0)

    selection_payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    selection_sha256 = hashlib.sha256(selection_payload.encode("utf-8")).hexdigest()
    notes = (
        "Existing Q/DQ topology is unchanged.",
        "Weighted-adjacent activation grids were excluded."
        if selector.exclude_weighted_adjacent
        else "Weighted-adjacent activation grids were allowed by the custom selector.",
        "Committed grids were staged into observers for PT2E export persistence."
        if committed
        else "No candidate passed the task-metric improvement gate; original grids were restored.",
    )
    return ActivationRangeRefinementReport(
        policy=policy_name,
        group_by=group_by,
        metric=metric_name,
        higher_is_better=higher_is_better,
        factors=tuple(normalized_factors),
        selected_fake_quantizers=len(selected),
        selection_sha256=selection_sha256,
        selection=rows,
        candidates=tuple(candidate_rows),
        groups=tuple(group_rows),
        baseline_metrics=baseline_metrics,
        best_metrics=current_metrics,
        best_factor=(
            next(iter(committed_factors.values()))
            if len(set(committed_factors.values())) == 1 and committed_factors
            else (1.0 if not committed_factors else None)
        ),
        metric_improvement=float(improvement),
        committed=committed,
        notes=notes,
    )


__all__ = [
    "ActivationRangeRefinementReport",
    "ActivationRangeSelector",
    "refine_activation_ranges",
]
