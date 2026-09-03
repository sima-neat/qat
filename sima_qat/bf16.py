#**************************************************************************
#||                        SiMa.ai CONFIDENTIAL                          ||
#||   Unpublished Copyright (c) 2024 SiMa.ai, All Rights Reserved.       ||
#**************************************************************************
# NOTICE:  All information contained herein is, and remains the property of
# SiMa.ai. The intellectual and technical concepts contained herein are
# proprietary to SiMa and may be covered by U.S. and Foreign Patents,
# patents in process, and are protected by trade secret or copyright law.
#
# Dissemination of this information or reproduction of this material is
# strictly forbidden unless prior written permission is obtained from
# SiMa.ai.  Access to the source code contained herein is hereby forbidden
# to anyone except current SiMa.ai employees, managers or contractors who
# have executed Confidentiality and Non-disclosure agreements explicitly
# covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure of this source code, which includes information
# that is confidential and/or proprietary, and is a trade secret of SiMa.ai.
#**************************************************************************
"""Target-required and attention-aware BF16 planning and simulation."""

from __future__ import annotations

import copy
import math
import operator
import re
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import torch
from torch import Tensor, nn
from torch.fx import GraphModule, Node
from torch.fx.node import map_arg


BF16OpType = Literal["matmul", "softmax"]
_VALID_OP_TYPES = frozenset(("matmul", "softmax"))

_MATMUL_TARGETS = frozenset(
    (
        torch.ops.aten.mm.default,
        torch.ops.aten.matmul.default,
        torch.ops.aten.bmm.default,
        torch.ops.aten.baddbmm.default,
    )
)
_SOFTMAX_TARGETS = frozenset(
    (
        torch.ops.aten.softmax.int,
        torch.ops.aten._softmax.default,
    )
)
_GRID_SAMPLE_TARGETS = frozenset(
    (
        torch.ops.aten.grid_sampler.default,
        torch.ops.aten.grid_sampler_2d.default,
        torch.ops.aten.grid_sampler_3d.default,
    )
)
_SDPA_TARGET = torch.ops.aten.scaled_dot_product_attention.default
_LAYOUT_TARGETS = frozenset(
    (
        operator.getitem,
        torch.ops.aten.clone.default,
        torch.ops.aten.contiguous.default,
        torch.ops.aten.detach.default,
        torch.ops.aten.flatten.using_ints,
        torch.ops.aten.permute.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten.select.int,
        torch.ops.aten.slice.Tensor,
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.t.default,
        torch.ops.aten.transpose.int,
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.view.default,
        torch.ops.aten._to_copy.default,
        torch.ops.aten._unsafe_view.default,
    )
)
_SCORE_ARITHMETIC_TARGETS = frozenset(
    (
        torch.ops.aten.add.Tensor,
        torch.ops.aten.sub.Tensor,
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.div.Tensor,
        torch.ops.aten.clamp.default,
        torch.ops.aten.max.default,
        torch.ops.aten.max.dim,
        torch.ops.aten.masked_fill.Scalar,
        torch.ops.aten.masked_fill_.Scalar,
        torch.ops.aten.logical_not.default,
    )
)
_SCORE_PATH_TARGETS = _LAYOUT_TARGETS | _SCORE_ARITHMETIC_TARGETS
_VALUE_PATH_TARGETS = _LAYOUT_TARGETS | frozenset(
    (
        torch.ops.aten.dropout.default,
        torch.ops.aten.dropout_.default,
    )
)
_ROUND_OUTPUT_TARGETS = (
    _MATMUL_TARGETS
    | _SOFTMAX_TARGETS
    | _GRID_SAMPLE_TARGETS
    | (
        _SCORE_ARITHMETIC_TARGETS
        - frozenset(
            (
                torch.ops.aten.logical_not.default,
                torch.ops.aten.max.dim,
            )
        )
    )
)


@dataclass(frozen=True)
class BF16Rule:
    """Select BF16 attention operators under matching PyTorch module paths."""

    module_path_regex: str
    op_types: tuple[BF16OpType, ...] = ("matmul", "softmax")

    def __post_init__(self) -> None:
        try:
            re.compile(self.module_path_regex)
        except re.error as error:
            raise ValueError(
                f"Invalid BF16 module_path_regex {self.module_path_regex!r}: {error}"
            ) from error
        if not self.op_types:
            raise ValueError("BF16Rule.op_types must not be empty")
        unsupported = set(self.op_types) - _VALID_OP_TYPES
        if unsupported:
            raise ValueError(
                "Unsupported BF16 operator type(s): "
                + ", ".join(sorted(unsupported))
            )


@dataclass(frozen=True)
class _AttentionRegion:
    score: str
    softmax: str
    values: tuple[str, ...]
    nodes: tuple[str, ...]
    module_paths: tuple[str, ...]

    @property
    def anchors(self) -> dict[str, tuple[str, ...]]:
        return {
            "matmul": (self.score, *self.values),
            "softmax": (self.softmax,),
        }


class _BF16EntrySTE(torch.autograd.Function):
    """Round an island input during training and disappear during export."""

    @staticmethod
    def forward(ctx: Any, value: Tensor) -> Tensor:
        del ctx
        rounded = value.to(torch.bfloat16).to(value.dtype)
        return value + (rounded - value).detach()

    @staticmethod
    def backward(ctx: Any, gradient: Tensor) -> Tensor:
        del ctx
        return gradient

    @staticmethod
    def symbolic(graph: Any, value: Any) -> Any:
        del graph
        return value


class _BF16OutputSTE(torch.autograd.Function):
    """Round a BF16 result and export an AFE precision annotation."""

    @staticmethod
    def forward(ctx: Any, value: Tensor) -> Tensor:
        del ctx
        rounded = value.to(torch.bfloat16).to(value.dtype)
        return value + (rounded - value).detach()

    @staticmethod
    def backward(ctx: Any, gradient: Tensor) -> Tensor:
        del ctx
        return gradient

    @staticmethod
    def symbolic(graph: Any, value: Any) -> Any:
        return graph.op(
            "ai.sima::AnnotatePrecision",
            value,
            precision_s="bfloat16",
        ).setType(value.type())


class _BF16Simulation(nn.Module):
    """Training BF16 rounding with optional precision-annotation export."""

    def __init__(self, annotate_output: bool) -> None:
        super().__init__()
        self.annotate_output = annotate_output

    def forward(self, value: Tensor) -> Tensor:
        function = _BF16OutputSTE if self.annotate_output else _BF16EntrySTE
        return function.apply(value)


def _module_path(node: Node) -> str:
    stack = node.meta.get("nn_module_stack", {})
    for value in reversed(tuple(stack.values())):
        if isinstance(value, tuple) and value:
            return str(value[0]) or "<root>"
    return ""


def _is_float_node(node: Node) -> bool:
    value = node.meta.get("val")
    return not isinstance(value, Tensor) or value.is_floating_point()


def _is_static_tensor(node: Node, memo: dict[Node, bool] | None = None) -> bool:
    if memo is None:
        memo = {}
    if node in memo:
        return memo[node]
    if node.op == "get_attr":
        memo[node] = True
        return True
    if node.op != "call_function" or node.target not in _LAYOUT_TARGETS:
        memo[node] = False
        return False
    input_nodes = node.all_input_nodes
    result = bool(input_nodes) and all(_is_static_tensor(value, memo) for value in input_nodes)
    memo[node] = result
    return result


def _matmul_operands(node: Node) -> tuple[Node, ...]:
    if node.op != "call_function" or node.target not in _MATMUL_TARGETS:
        return ()
    indexes = (1, 2) if node.target == torch.ops.aten.baddbmm.default else (0, 1)
    operands = []
    for index in indexes:
        if index >= len(node.args) or not isinstance(node.args[index], Node):
            return ()
        operands.append(node.args[index])
    return tuple(operands)


def _eligible_matmul(node: Node) -> bool:
    if node.op != "call_function" or node.target not in _MATMUL_TARGETS:
        return False
    operands = _matmul_operands(node)
    return bool(operands) and all(
        _is_float_node(operand) and not _is_static_tensor(operand)
        for operand in operands
    )


def _find_upstream_scores(start: Node) -> list[Node]:
    pending = [(start, 0)]
    visited: set[Node] = set()
    scores: list[tuple[int, Node]] = []
    while pending:
        node, depth = pending.pop(0)
        if node in visited or depth > 64:
            continue
        visited.add(node)
        if _eligible_matmul(node):
            scores.append((depth, node))
            continue
        if node.op == "call_function" and node.target in _SCORE_PATH_TARGETS:
            pending.extend((value, depth + 1) for value in node.all_input_nodes)
    if not scores:
        return []
    nearest = min(depth for depth, _ in scores)
    return [node for depth, node in scores if depth == nearest]


def _find_downstream_values(softmax: Node) -> list[Node]:
    pending = [(user, 1) for user in softmax.users]
    visited: set[Node] = set()
    values: list[tuple[int, Node]] = []
    while pending:
        node, depth = pending.pop(0)
        if node in visited or depth > 64:
            continue
        visited.add(node)
        if _eligible_matmul(node):
            values.append((depth, node))
            continue
        if node.op == "call_function" and node.target in _VALUE_PATH_TARGETS:
            pending.extend((user, depth + 1) for user in node.users)
    if not values:
        return []
    nearest = min(depth for depth, _ in values)
    return [node for depth, node in values if depth == nearest]


def _reachable_forward(source: Node, targets: frozenset[object]) -> set[Node]:
    pending = list(source.users)
    visited: set[Node] = set()
    while pending and len(visited) < 512:
        node = pending.pop()
        if node in visited:
            continue
        if node.op != "call_function" or node.target not in targets:
            continue
        visited.add(node)
        pending.extend(node.users)
    return visited


def _reachable_backward(source: Node, targets: frozenset[object]) -> set[Node]:
    pending = list(source.all_input_nodes)
    visited: set[Node] = set()
    while pending and len(visited) < 512:
        node = pending.pop()
        if node in visited:
            continue
        if node.op != "call_function" or node.target not in targets:
            continue
        visited.add(node)
        pending.extend(node.all_input_nodes)
    return visited


def _score_path_nodes(score: Node, softmax: Node) -> set[Node]:
    forward = _reachable_forward(score, _SCORE_PATH_TARGETS | _SOFTMAX_TARGETS)
    backward = _reachable_backward(
        softmax, _SCORE_PATH_TARGETS | _MATMUL_TARGETS
    )
    return {score, softmax} | (forward & backward)


def _value_path_nodes(softmax: Node, value: Node) -> set[Node]:
    forward = _reachable_forward(softmax, _VALUE_PATH_TARGETS | _MATMUL_TARGETS)
    backward = _reachable_backward(value, _VALUE_PATH_TARGETS | _SOFTMAX_TARGETS)
    return {softmax, value} | (forward & backward)


def _pre_score_nodes(score: Node) -> set[Node]:
    selected: set[Node] = set()
    for operand in _matmul_operands(score):
        current = operand
        while current.op == "call_function" and current.target in _LAYOUT_TARGETS:
            selected.add(current)
            tensor_inputs = current.all_input_nodes
            if len(tensor_inputs) != 1:
                break
            current = tensor_inputs[0]
        if current.op != "call_function" or current.target not in {
            torch.ops.aten.mul.Tensor,
            torch.ops.aten.div.Tensor,
        }:
            continue
        dynamic_inputs = [
            value
            for value in current.all_input_nodes
            if not _is_static_tensor(value)
        ]
        scalar_argument = any(not isinstance(value, Node) for value in current.args)
        if len(dynamic_inputs) == 1 and scalar_argument:
            selected.add(current)
    return selected


def _attention_regions(model: GraphModule) -> list[_AttentionRegion]:
    regions = []
    for softmax in model.graph.nodes:
        if softmax.op != "call_function" or softmax.target not in _SOFTMAX_TARGETS:
            continue
        if not softmax.args or not isinstance(softmax.args[0], Node):
            continue
        scores = _find_upstream_scores(softmax.args[0])
        values = _find_downstream_values(softmax)
        if len(scores) != 1 or not values:
            continue
        score = scores[0]
        nodes = _score_path_nodes(score, softmax) | _pre_score_nodes(score)
        for value in values:
            nodes |= _value_path_nodes(softmax, value)
        module_paths = tuple(sorted({_module_path(node) for node in nodes if _module_path(node)}))
        regions.append(
            _AttentionRegion(
                score=score.name,
                softmax=softmax.name,
                values=tuple(sorted(node.name for node in values)),
                nodes=tuple(sorted(node.name for node in nodes)),
                module_paths=module_paths,
            )
        )
    return regions


def _copy_source_meta(source: Node, destination: Node, *, copy_value: bool = False) -> None:
    for key in ("nn_module_stack", "source_fn_stack", "stack_trace"):
        if key in source.meta:
            destination.meta[key] = copy.deepcopy(source.meta[key])
    if copy_value and "val" in source.meta:
        destination.meta["val"] = source.meta["val"]


def _infer_value_meta(node: Node) -> None:
    """Evaluate a newly created ATen node on captured FakeTensor metadata."""
    args = map_arg(node.args, lambda value: value.meta["val"])
    kwargs = map_arg(node.kwargs, lambda value: value.meta["val"])
    node.meta["val"] = node.target(*args, **kwargs)


def decompose_scaled_dot_product_attention(model: GraphModule) -> GraphModule:
    """Rewrite supported Torch SDPA calls into explicit attention operations."""

    for node in list(model.graph.nodes):
        if node.op != "call_function" or node.target != _SDPA_TARGET:
            continue
        query, key, value = node.args[:3]
        if not all(isinstance(argument, Node) for argument in (query, key, value)):
            raise ValueError(f"SDPA node {node.name} has unsupported non-tensor operands")
        mask = node.kwargs.get(
            "attn_mask", node.args[3] if len(node.args) > 3 else None
        )
        dropout = float(
            node.kwargs.get(
                "dropout_p", node.args[4] if len(node.args) > 4 else 0.0
            )
        )
        causal = bool(
            node.kwargs.get(
                "is_causal", node.args[5] if len(node.args) > 5 else False
            )
        )
        scale = node.kwargs.get("scale", node.args[6] if len(node.args) > 6 else None)
        enable_gqa = bool(
            node.kwargs.get("enable_gqa", node.args[7] if len(node.args) > 7 else False)
        )
        if causal or enable_gqa:
            options = []
            if causal:
                options.append("causal attention")
            if enable_gqa:
                options.append("grouped-query attention")
            raise ValueError(
                f"SDPA node {node.name} uses unsupported " + " and ".join(options)
            )
        if scale is None:
            query_value = query.meta.get("val")
            if not isinstance(query_value, Tensor):
                raise ValueError(f"SDPA node {node.name} has no captured query shape")
            scale = 1.0 / math.sqrt(int(query_value.shape[-1]))
        if not isinstance(scale, (float, int)):
            raise ValueError(f"SDPA node {node.name} uses a dynamic scale")

        with model.graph.inserting_before(node):
            created_nodes: list[Node] = []
            scaled_query = model.graph.call_function(
                torch.ops.aten.mul.Tensor, (query, float(scale))
            )
            created_nodes.append(scaled_query)
            _infer_value_meta(scaled_query)
            transposed_key = model.graph.call_function(
                torch.ops.aten.transpose.int, (key, -2, -1)
            )
            created_nodes.append(transposed_key)
            _infer_value_meta(transposed_key)
            score = model.graph.call_function(
                torch.ops.aten.matmul.default, (scaled_query, transposed_key)
            )
            created_nodes.append(score)
            _infer_value_meta(score)
            score_input = score
            if isinstance(mask, Node):
                mask_value = mask.meta.get("val")
                if isinstance(mask_value, Tensor) and mask_value.dtype == torch.bool:
                    inverted_mask = model.graph.call_function(
                        torch.ops.aten.logical_not.default, (mask,)
                    )
                    created_nodes.append(inverted_mask)
                    _infer_value_meta(inverted_mask)
                    score_input = model.graph.call_function(
                        torch.ops.aten.masked_fill.Scalar,
                        (score, inverted_mask, float("-inf")),
                    )
                    created_nodes.append(score_input)
                    _infer_value_meta(score_input)
                else:
                    score_input = model.graph.call_function(
                        torch.ops.aten.add.Tensor, (score, mask)
                    )
                    created_nodes.append(score_input)
                    _infer_value_meta(score_input)
            elif mask is not None:
                raise ValueError(f"SDPA node {node.name} uses an unsupported mask")
            probability = model.graph.call_function(
                torch.ops.aten.softmax.int, (score_input, -1)
            )
            created_nodes.append(probability)
            _infer_value_meta(probability)
            if dropout:
                probability = model.graph.call_function(
                    torch.ops.aten.dropout.default, (probability, dropout, True)
                )
                created_nodes.append(probability)
                _infer_value_meta(probability)
            output = model.graph.call_function(
                torch.ops.aten.matmul.default, (probability, value)
            )
            created_nodes.append(output)
            _infer_value_meta(output)

        for created in created_nodes:
            _copy_source_meta(node, created)
        _copy_source_meta(node, output, copy_value=True)
        output.meta["sima_decomposed_sdpa"] = True
        node.replace_all_uses_with(output)
        model.graph.erase_node(node)

    model.graph.lint()
    model.recompile()
    return model


def resolve_bf16_plan(
    model: GraphModule,
    rules: Sequence[BF16Rule] | None,
) -> dict[str, Any]:
    """Resolve mandatory target BF16 and the optional attention policy."""

    regions = _attention_regions(model)
    nodes_by_name = {node.name: node for node in model.graph.nodes}
    mandatory_nodes = sorted(
        node.name
        for node in model.graph.nodes
        if node.op == "call_function" and node.target in _GRID_SAMPLE_TARGETS
    )
    selected: set[str] = set(mandatory_nodes)
    selected_anchors: set[str] = set()
    unmatched_softmaxes = [
        node
        for node in model.graph.nodes
        if node.op == "call_function"
        and node.target in _SOFTMAX_TARGETS
        and all(node.name != region.softmax for region in regions)
    ]

    if rules is None:
        mode = "automatic"
        for region in regions:
            selected.update(region.nodes)
    elif not rules:
        mode = "disabled"
    else:
        mode = "custom"
        for rule in rules:
            expression = re.compile(rule.module_path_regex)
            matching_modules = {
                _module_path(node)
                for node in model.graph.nodes
                if _module_path(node) and expression.search(_module_path(node))
            }
            if not matching_modules:
                raise ValueError(
                    f"BF16 rule {rule.module_path_regex!r} matched no PyTorch module path"
                )
            rule_anchors: set[str] = set()
            for region in regions:
                for op_type in rule.op_types:
                    for name in region.anchors[op_type]:
                        path = _module_path(nodes_by_name[name])
                        if expression.search(path):
                            rule_anchors.add(name)
            if not rule_anchors:
                raise ValueError(
                    f"BF16 rule {rule.module_path_regex!r} matched modules but no "
                    "eligible attention operators"
                )
            selected_anchors.update(rule_anchors)

        for region in regions:
            anchors = {name for values in region.anchors.values() for name in values}
            active = anchors & selected_anchors
            if not active:
                continue
            selected.update(active)
            if active == anchors:
                selected.update(region.nodes)

    selected_nodes = [nodes_by_name[name] for name in selected]
    round_outputs = sorted(
        node.name
        for node in selected_nodes
        if node.op == "call_function" and node.target in _ROUND_OUTPUT_TARGETS
    )
    # aten.max.dim returns (values, indices); its value extraction is the BF16
    # numerical boundary that appears as ReduceMax in ONNX.
    for node in selected_nodes:
        if (
            node.op == "call_function"
            and node.target == operator.getitem
            and node.args
            and isinstance(node.args[0], Node)
            and node.args[0].target == torch.ops.aten.max.dim
            and len(node.args) > 1
            and node.args[1] == 0
        ):
            round_outputs.append(node.name)
    round_outputs = sorted(set(round_outputs))

    report_regions = []
    for region in regions:
        region_selected = sorted(set(region.nodes) & selected)
        report_regions.append(
            {
                "score": region.score,
                "softmax": region.softmax,
                "values": list(region.values),
                "module_paths": list(region.module_paths),
                "selected_nodes": region_selected,
            }
        )

    unsupported = []
    for node in unmatched_softmaxes:
        path = _module_path(node)
        if "attn" in path.lower() or "attention" in path.lower():
            unsupported.append(
                {
                    "node": node.name,
                    "module_path": path,
                    "reason": "unsupported_attention_topology",
                }
            )

    normalized_rules = None if rules is None else [
        {
            "module_path_regex": rule.module_path_regex,
            "op_types": list(rule.op_types),
        }
        for rule in rules
    ]
    return {
        "schema": "sima-bf16-plan/v1",
        "mode": mode,
        "rules": normalized_rules,
        "mandatory_nodes": mandatory_nodes,
        "selected_nodes": sorted(selected),
        "round_output_nodes": round_outputs,
        "regions": report_regions,
        "unsupported_regions": unsupported,
    }


def insert_bf16_simulation(model: GraphModule, plan: dict[str, Any]) -> GraphModule:
    """Insert BF16 STE rounding at selected operation inputs and outputs."""

    selected = set(plan["selected_nodes"])
    nodes_by_name = {node.name: node for node in model.graph.nodes}
    module_index = 0

    def add_simulation(source: Node, *, annotate_output: bool, before: Node | None = None) -> Node:
        nonlocal module_index
        name = f"sima_bf16_{'output' if annotate_output else 'input'}_{module_index}"
        module_index += 1
        model.add_submodule(name, _BF16Simulation(annotate_output))
        context = model.graph.inserting_before(before) if before is not None else model.graph.inserting_after(source)
        with context:
            result = model.graph.call_module(name, (source,))
        # FakeTensor values in PT2E metadata cannot be deep-copied because
        # they intentionally have no backing storage.
        result.meta = dict(source.meta)
        return result

    # Round every floating tensor as it enters the selected subgraph. Layout
    # nodes are part of the island even though their own outputs do not need a
    # second rounding operation.
    for name in tuple(plan["selected_nodes"]):
        node = nodes_by_name.get(name)
        if node is None:
            raise RuntimeError(f"BF16 operation {name!r} disappeared during QAT preparation")
        new_args = list(node.args)
        changed = False
        for index, argument in enumerate(new_args):
            if not isinstance(argument, Node) or argument.name in selected:
                continue
            if not _is_float_node(argument):
                continue
            new_args[index] = add_simulation(argument, annotate_output=False, before=node)
            changed = True
        if changed:
            node.args = tuple(new_args)

    for name in tuple(plan["round_output_nodes"]):
        node = nodes_by_name[name]
        output = add_simulation(node, annotate_output=True)
        node.replace_all_uses_with(output)
        output.args = (node,)

    model.graph.lint()
    model.recompile()
    return model


def mark_bf16_nodes(model: GraphModule, plan: dict[str, Any]) -> None:
    """Tag selected nodes so the INT8 quantizer can exclude the whole island."""

    selected = set(plan["selected_nodes"])
    missing = selected - {node.name for node in model.graph.nodes}
    if missing:
        raise RuntimeError("BF16 plan refers to missing node(s): " + ", ".join(sorted(missing)))
    for node in model.graph.nodes:
        if node.name in selected:
            node.meta["sima_precision"] = "bfloat16"


def fuse_int8_attention_boundaries(model: GraphModule) -> None:
    """Avoid a redundant A8 boundary between score accumulation and Softmax."""

    for region in _attention_regions(model):
        nodes = {node.name: node for node in model.graph.nodes}
        score = nodes[region.score]
        softmax = nodes[region.softmax]
        if (
            score.meta.get("sima_precision") == "bfloat16"
            or softmax.meta.get("sima_precision") == "bfloat16"
        ):
            continue
        score_annotation = score.meta.get("quantization_annotation")
        if score_annotation is not None and score_annotation._annotated:
            score_annotation.output_qspec = None
        softmax_annotation = softmax.meta.get("quantization_annotation")
        if softmax_annotation is not None and softmax_annotation._annotated:
            softmax_annotation.input_qspec_map = {}
        for name in region.nodes:
            node = nodes[name]
            if node in (score, softmax):
                continue
            annotation = node.meta.get("quantization_annotation")
            if annotation is not None and annotation._annotated:
                annotation.input_qspec_map = {}
                annotation.output_qspec = None
