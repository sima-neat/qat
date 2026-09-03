"""Attention precision policy, BF16 simulation, and checkpoint contracts."""

from __future__ import annotations

import copy

import onnx
import pytest
import torch
from torch import Tensor, nn
from torch.ao.quantization.fake_quantize import FakeQuantizeBase
from torch.nn import functional as F

from sima_qat import (
    BF16Rule,
    sima_export_onnx,
    sima_finalize_qat_model,
    sima_freeze_qat,
    sima_prepare_qat_model,
)


pytestmark = pytest.mark.regression


def _bf16_round(value: Tensor) -> Tensor:
    return value.to(torch.bfloat16).to(value.dtype)


class AttentionCore(nn.Module):
    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        scores = torch.matmul(query, key.transpose(-2, -1)) * 0.5
        probabilities = torch.softmax(scores, dim=-1)
        return torch.matmul(probabilities, value)


class ProjectedAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.query = nn.Linear(8, 8)
        self.key = nn.Linear(8, 8)
        self.value = nn.Linear(8, 8)
        self.attention = AttentionCore()
        self.output = nn.Linear(8, 8)

    def forward(self, inputs: Tensor) -> Tensor:
        query = self.query(inputs).reshape(1, 2, 4, 8)
        key = self.key(inputs).reshape(1, 2, 4, 8)
        value = self.value(inputs).reshape(1, 2, 4, 8)
        attended = self.attention(query, key, value)
        return self.output(attended.reshape(1, 8, 8))


class SDPAttention(nn.Module):
    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return torch.nn.functional.scaled_dot_product_attention(query, key, value)


class StaticWeightMatMul(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(8, 8))

    def forward(self, inputs: Tensor) -> Tensor:
        return torch.matmul(inputs, self.weight)


class DeformableLikeAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = nn.Softmax(dim=-1)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.attention(inputs).sum(dim=-1)


class GridSampleModel(nn.Module):
    def forward(self, inputs: Tensor, grid: Tensor) -> Tensor:
        return F.grid_sample(
            inputs,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )


class BidirectionalAttention(nn.Module):
    """Grounding-DINO-style shared score matrix with two Softmax branches."""

    def forward(self, visual: Tensor, language: Tensor) -> tuple[Tensor, Tensor]:
        scores = torch.bmm(visual, language.transpose(1, 2))
        visual_scores = torch.clamp(
            scores - scores.max(dim=2, keepdim=True)[0],
            min=-50_000.0,
            max=50_000.0,
        )
        language_scores = torch.clamp(
            scores - scores.max(dim=1, keepdim=True)[0],
            min=-50_000.0,
            max=50_000.0,
        )
        visual_probabilities = torch.softmax(visual_scores, dim=2)
        language_probabilities = torch.softmax(language_scores, dim=1)
        visual_output = torch.bmm(visual_probabilities, language)
        language_output = torch.bmm(
            language_probabilities.transpose(1, 2), visual
        )
        return visual_output, language_output


def _attention_inputs() -> tuple[Tensor, Tensor, Tensor]:
    return (
        torch.randn(1, 2, 4, 8),
        torch.randn(1, 2, 4, 8),
        torch.randn(1, 2, 4, 8),
    )


def _fake_quantizers(model: nn.Module) -> list[FakeQuantizeBase]:
    return [
        module
        for module in model.modules()
        if isinstance(module, FakeQuantizeBase)
    ]


def test_bf16_rule_validates_regex_and_operator_types() -> None:
    assert BF16Rule(r"^attention$").op_types == ("matmul", "softmax")
    with pytest.raises(ValueError, match="Invalid BF16"):
        BF16Rule("[")
    with pytest.raises(ValueError, match="must not be empty"):
        BF16Rule(".*", ())
    with pytest.raises(ValueError, match="Unsupported BF16"):
        BF16Rule(".*", ("conv",))  # type: ignore[arg-type]


def test_automatic_policy_promotes_attention_but_not_projections() -> None:
    inputs = torch.randn(1, 8, 8)
    prepared = sima_prepare_qat_model(ProjectedAttention(), (inputs,), "cpu")
    plan = prepared.meta["sima_bf16_plan"]

    assert plan["mode"] == "automatic"
    assert len(plan["regions"]) == 1
    assert {"matmul", "softmax"} <= {
        "matmul" if "matmul" in name else "softmax"
        for name in plan["round_output_nodes"]
    }
    assert any("sima_bf16_output" in name for name, _ in prepared.named_modules())
    # The four Linear projections remain W8A8 and contribute weight and
    # activation fake quantizers around the BF16 attention island.
    assert len(_fake_quantizers(prepared)) >= 8
    assert torch.isfinite(prepared(inputs)).all()


def test_empty_rules_keep_attention_in_strict_w8a8() -> None:
    inputs = _attention_inputs()
    prepared = sima_prepare_qat_model(
        AttentionCore(),
        inputs,
        "cpu",
        bf16_rules=(),
    )

    assert prepared.meta["sima_bf16_plan"]["mode"] == "disabled"
    assert prepared.meta["sima_bf16_plan"]["selected_nodes"] == []
    assert not any("sima_bf16" in name for name, _ in prepared.named_modules())
    assert len(_fake_quantizers(prepared)) >= 3
    assert torch.isfinite(prepared(*inputs)).all()


def test_grid_sample_is_mandatory_bf16_when_attention_is_disabled(
    tmp_path,
) -> None:
    inputs = torch.randn(1, 3, 6, 7)
    grid = torch.empty(1, 4, 5, 2).uniform_(-1, 1)
    prepared = sima_prepare_qat_model(
        GridSampleModel(),
        (inputs, grid),
        "cpu",
        bf16_rules=(),
    )
    plan = prepared.meta["sima_bf16_plan"]

    assert plan["mode"] == "disabled"
    assert len(plan["mandatory_nodes"]) == 1
    assert plan["selected_nodes"] == plan["mandatory_nodes"]
    assert plan["round_output_nodes"] == plan["mandatory_nodes"]

    expected = _bf16_round(
        F.grid_sample(
            _bf16_round(inputs),
            _bf16_round(grid),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
    )
    torch.testing.assert_close(prepared(inputs, grid), expected, rtol=0, atol=0)

    sima_freeze_qat(prepared)
    finalized = sima_finalize_qat_model(prepared)
    output_path = tmp_path / "bf16_grid_sample.onnx"
    sima_export_onnx(
        finalized,
        (inputs, grid),
        str(output_path),
        device="cpu",
    )
    exported = onnx.load(output_path)
    grid_samples = [
        node for node in exported.graph.node if node.op_type == "GridSample"
    ]
    assert len(grid_samples) == 1
    annotations = [
        node
        for node in exported.graph.node
        if node.domain == "ai.sima" and node.op_type == "AnnotatePrecision"
    ]
    assert len(annotations) == 1
    assert annotations[0].input[0] == grid_samples[0].output[0]


def test_regex_rules_replace_automatic_selection() -> None:
    inputs = torch.randn(1, 8, 8)
    prepared = sima_prepare_qat_model(
        ProjectedAttention(),
        (inputs,),
        "cpu",
        bf16_rules=(BF16Rule(r"^attention$", ("softmax",)),),
    )
    plan = prepared.meta["sima_bf16_plan"]

    assert plan["mode"] == "custom"
    assert len(plan["selected_nodes"]) == 1
    assert "softmax" in plan["selected_nodes"][0]

    with pytest.raises(ValueError, match="matched no PyTorch module path"):
        sima_prepare_qat_model(
            ProjectedAttention(),
            (inputs,),
            "cpu",
            bf16_rules=(BF16Rule(r"^decoder\.missing$"),),
        )


def test_bf16_forward_matches_explicit_rounding_and_keeps_gradients() -> None:
    query, key, value = _attention_inputs()
    query.requires_grad_()
    key.requires_grad_()
    value.requires_grad_()
    prepared = sima_prepare_qat_model(
        AttentionCore(),
        (query, key, value),
        "cpu",
    )

    actual = prepared(query, key, value)
    rounded_query = _bf16_round(query)
    rounded_key = _bf16_round(key)
    rounded_value = _bf16_round(value)
    scores = _bf16_round(torch.matmul(rounded_query, rounded_key.transpose(-2, -1)))
    scores = _bf16_round(scores * 0.5)
    probabilities = _bf16_round(torch.softmax(scores, dim=-1))
    expected = _bf16_round(torch.matmul(probabilities, rounded_value))

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().mean().backward()
    assert all(
        tensor.grad is not None and torch.isfinite(tensor.grad).all()
        for tensor in (query, key, value)
    )


def test_sdpa_is_normalized_to_explicit_bf16_attention() -> None:
    inputs = _attention_inputs()
    eager = SDPAttention().eval()
    expected = eager(*inputs)
    prepared = sima_prepare_qat_model(eager, inputs, "cpu")

    targets = {
        node.target
        for node in prepared.graph.nodes
        if node.op == "call_function"
    }
    assert torch.ops.aten.scaled_dot_product_attention.default not in targets
    assert torch.ops.aten.matmul.default in targets
    assert torch.ops.aten.softmax.int in targets
    assert prepared.meta["sima_bf16_plan"]["regions"]
    # BF16 simulation is expected to perturb SDPA slightly, but graph surgery
    # itself must preserve its numerical contract closely.
    torch.testing.assert_close(prepared(*inputs), expected, rtol=2e-2, atol=2e-2)


def test_branching_bidirectional_attention_is_one_bf16_island() -> None:
    inputs = (torch.randn(2, 5, 8), torch.randn(2, 7, 8))
    prepared = sima_prepare_qat_model(BidirectionalAttention(), inputs, "cpu")
    plan = prepared.meta["sima_bf16_plan"]

    assert len(plan["regions"]) == 2
    assert len({region["score"] for region in plan["regions"]}) == 1
    assert all(region["selected_nodes"] for region in plan["regions"])
    outputs = prepared(*inputs)
    assert all(torch.isfinite(output).all() for output in outputs)


def test_static_weight_matmul_stays_w8a8_and_is_not_treated_as_attention() -> None:
    inputs = torch.randn(2, 8)
    prepared = sima_prepare_qat_model(StaticWeightMatMul(), (inputs,), "cpu")

    assert prepared.meta["sima_bf16_plan"]["selected_nodes"] == []
    assert _fake_quantizers(prepared)


def test_unrecognized_attention_softmax_is_reported_not_promoted() -> None:
    inputs = torch.randn(1, 2, 4, 8)
    prepared = sima_prepare_qat_model(
        DeformableLikeAttention(),
        (inputs,),
        "cpu",
    )
    plan = prepared.meta["sima_bf16_plan"]

    assert plan["selected_nodes"] == []
    assert plan["unsupported_regions"]
    assert plan["unsupported_regions"][0]["reason"] == "unsupported_attention_topology"


def test_checkpoint_requires_the_same_resolved_precision_policy() -> None:
    inputs = _attention_inputs()
    automatic = sima_prepare_qat_model(AttentionCore(), inputs, "cpu")
    checkpoint = copy.deepcopy(automatic.state_dict())

    resumed = sima_prepare_qat_model(AttentionCore(), inputs, "cpu")
    resumed.load_state_dict(checkpoint)

    strict_w8a8 = sima_prepare_qat_model(
        AttentionCore(), inputs, "cpu", bf16_rules=()
    )
    with pytest.raises(RuntimeError, match="precision policy does not match"):
        strict_w8a8.load_state_dict(checkpoint)


def test_bf16_attention_exports_annotations_and_all_inputs(tmp_path) -> None:
    inputs = _attention_inputs()
    prepared = sima_prepare_qat_model(AttentionCore(), inputs, "cpu")
    prepared(*inputs)
    sima_freeze_qat(prepared)
    finalized = sima_finalize_qat_model(prepared)

    output_path = tmp_path / "bf16_attention.onnx"
    sima_export_onnx(
        finalized,
        inputs,
        str(output_path),
        input_names=["query", "key", "value"],
        output_names=["output"],
        device="cpu",
    )
    exported = onnx.load(output_path)
    onnx.checker.check_model(exported)

    annotations = [
        node
        for node in exported.graph.node
        if node.domain == "ai.sima" and node.op_type == "AnnotatePrecision"
    ]
    assert len(exported.graph.input) == 3
    assert len(annotations) >= 4
    assert not any("sima_bf16_input" in node.name for node in exported.graph.node)
    assert all(
        next(attribute.s for attribute in node.attribute if attribute.name == "precision")
        == b"bfloat16"
        for node in annotations
    )


def test_strict_w8a8_attention_exports_qdq_without_bf16_annotations(
    tmp_path,
) -> None:
    inputs = _attention_inputs()
    prepared = sima_prepare_qat_model(
        AttentionCore(), inputs, "cpu", bf16_rules=()
    )
    prepared(*inputs)
    sima_freeze_qat(prepared)
    finalized = sima_finalize_qat_model(prepared)

    output_path = tmp_path / "w8a8_attention.onnx"
    sima_export_onnx(finalized, inputs, str(output_path), device="cpu")
    exported = onnx.load(output_path)
    operator_types = {node.op_type for node in exported.graph.node}

    assert "MatMul" in operator_types
    assert "Softmax" in operator_types
    assert "QuantizeLinear" in operator_types
    assert "DequantizeLinear" in operator_types
    assert not any(node.domain == "ai.sima" for node in exported.graph.node)
