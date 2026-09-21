"""Target-specific correctness and rejection contracts for added operators."""

from __future__ import annotations

import onnx
import pytest
import torch
from torch import Tensor, nn
from torch.ao.quantization.fake_quantize import FakeQuantizeBase

from sima_qat import (
    sima_export_onnx,
    sima_finalize_qat_model,
    sima_freeze_qat,
    sima_prepare_qat_model,
)


pytestmark = pytest.mark.regression


class RepeatedInputConcat(nn.Module):
    def forward(self, inputs: Tensor) -> Tensor:
        channel = inputs[:, :1]
        return torch.cat((channel,) * 4, dim=1)


class IdentityPaddingConcat(nn.Module):
    def __init__(self, identity: float) -> None:
        super().__init__()
        self.identity = identity

    def forward(self, inputs: Tensor) -> Tensor:
        prefix_source = inputs[:1]
        prefix = (
            torch.zeros_like(prefix_source)
            if self.identity == 0.0
            else torch.ones_like(prefix_source)
        )
        return torch.cat((prefix, inputs[:-1]), dim=0)


class ApproximateGelu(nn.Module):
    def forward(self, inputs: Tensor) -> Tensor:
        return torch.nn.functional.gelu(inputs, approximate="tanh")


class AttentionCore(nn.Module):
    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        scores = torch.bmm(query, key.transpose(1, 2))
        probabilities = torch.softmax(scores, dim=-1)
        return torch.bmm(probabilities, value)


class IntegerMatMul(nn.Module):
    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        return torch.mm(left, right)


class ConvTransposeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.operation = nn.ConvTranspose2d(3, 4, 3)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.operation(inputs)


class EmbeddingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 4)

    def forward(self, indices: Tensor) -> Tensor:
        return self.embedding(indices)


class GridSampleModel(nn.Module):
    def forward(self, inputs: Tensor, grid: Tensor) -> Tensor:
        return torch.nn.functional.grid_sample(inputs, grid, align_corners=False)


class ReduceMinModel(nn.Module):
    def forward(self, inputs: Tensor) -> Tensor:
        return torch.amin(inputs, dim=(2, 3))


class CumSumModel(nn.Module):
    def forward(self, inputs: Tensor) -> Tensor:
        return torch.cumsum(inputs, dim=1)


def _converted_cat(model: nn.Module, inputs: Tensor) -> torch.fx.Node:
    prepared = sima_prepare_qat_model(model, (inputs,), "cpu")
    prepared(inputs)
    converted = sima_finalize_qat_model(prepared)
    return next(
        node
        for node in converted.graph.nodes
        if node.op == "call_function" and node.target == torch.ops.aten.cat.default
    )


def _qdq_qparams(node: torch.fx.Node) -> tuple[object, object]:
    return node.args[1], node.args[2]


def test_every_reused_conv_invocation_has_fake_quantized_edges() -> None:
    from .cases import IMAGE_INPUT, ReusedConv2dModel

    inputs = IMAGE_INPUT()
    prepared = sima_prepare_qat_model(ReusedConv2dModel(), inputs, "cpu")
    modules = dict(prepared.named_modules(remove_duplicate=False))
    convs = [
        node
        for node in prepared.graph.nodes
        if node.op == "call_function" and node.target == torch.ops.aten.conv2d.default
    ]

    assert len(convs) == 2
    for conv in convs:
        for argument in conv.args[:2]:
            assert argument.op == "call_module"
            assert isinstance(modules[str(argument.target)], FakeQuantizeBase)


def test_repeated_input_concat_reuses_the_payload_grid() -> None:
    cat = _converted_cat(RepeatedInputConcat(), torch.randn(1, 3, 8, 8))
    input_dequantizers = list(cat.args[0])
    output_quantizer = next(
        user
        for user in cat.users
        if user.target == torch.ops.quantized_decomposed.quantize_per_tensor.default
    )

    assert len(input_dequantizers) == 4
    assert len(set(input_dequantizers)) == 1
    assert _qdq_qparams(output_quantizer) == _qdq_qparams(input_dequantizers[0])


@pytest.mark.parametrize("identity", [0.0, 1.0])
def test_identity_padding_concat_reuses_the_payload_grid(identity: float) -> None:
    cat = _converted_cat(IdentityPaddingConcat(identity), torch.randn(4, 3, 8, 8))
    input_dequantizers = list(cat.args[0])
    output_quantizer = next(
        user
        for user in cat.users
        if user.target == torch.ops.quantized_decomposed.quantize_per_tensor.default
    )

    assert len(input_dequantizers) == 2
    assert _qdq_qparams(input_dequantizers[0]) == _qdq_qparams(input_dequantizers[1])
    assert _qdq_qparams(output_quantizer) == _qdq_qparams(input_dequantizers[1])


def test_tanh_gelu_is_rejected_instead_of_silently_claiming_int8_support() -> None:
    with pytest.raises(ValueError, match="exact GELU"):
        sima_prepare_qat_model(
            ApproximateGelu(),
            (torch.randn(1, 4, 8),),
            "cpu",
        )


def test_attention_core_finalizes_and_exports_all_inputs(tmp_path) -> None:
    inputs = (
        torch.randn(1, 4, 8),
        torch.randn(1, 6, 8),
        torch.randn(1, 6, 5),
    )
    prepared = sima_prepare_qat_model(AttentionCore(), inputs, "cpu")
    prepared(*inputs)
    attention_nodes = [
        node
        for node in prepared.graph.nodes
        if node.op == "call_function"
        and node.target
        in {
            torch.ops.aten.bmm.default,
            torch.ops.aten.softmax.int,
            torch.ops.aten._softmax.default,
        }
    ]
    assert len(attention_nodes) == 3
    assert all(
        getattr(node.meta.get("quantization_annotation"), "_annotated", False)
        or any(
            user.op == "call_module"
            and isinstance(prepared.get_submodule(user.target), FakeQuantizeBase)
            for user in node.users
        )
        for node in attention_nodes
    )

    sima_freeze_qat(prepared)
    finalized = sima_finalize_qat_model(prepared)
    expected = finalized(*inputs)
    output_path = tmp_path / "attention_core.onnx"
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
    operator_types = {node.op_type for node in exported.graph.node}
    assert len(exported.graph.input) == 3
    assert {"MatMul", "Softmax", "QuantizeLinear", "DequantizeLinear"} <= operator_types
    assert torch.isfinite(expected).all()


def test_integer_matmul_operands_are_not_qat_annotated() -> None:
    inputs = (
        torch.randint(-4, 4, (4, 8), dtype=torch.int32),
        torch.randint(-4, 4, (8, 3), dtype=torch.int32),
    )
    prepared = sima_prepare_qat_model(IntegerMatMul(), inputs, "cpu")
    mm = next(
        node
        for node in prepared.graph.nodes
        if node.op == "call_function" and node.target == torch.ops.aten.mm.default
    )

    assert not getattr(mm.meta.get("quantization_annotation"), "_annotated", False)
    assert not any(isinstance(module, FakeQuantizeBase) for module in prepared.modules())


@pytest.mark.parametrize(
    ("model", "inputs", "message"),
    [
        (ConvTransposeModel(), (torch.randn(1, 3, 8, 8),), "ConvTranspose2d"),
        (EmbeddingModel(), (torch.randint(0, 16, (2, 5)),), "Embedding/Gather"),
        (
            GridSampleModel(),
            (torch.randn(1, 3, 8, 8), torch.randn(1, 4, 4, 2)),
            "GridSample",
        ),
        (ReduceMinModel(), (torch.randn(1, 3, 8, 8),), "ReduceMin"),
        (CumSumModel(), (torch.randn(1, 3, 8, 8),), "CumSum"),
    ],
    ids=("conv_transpose", "embedding", "grid_sample", "reduce_min", "cumsum"),
)
def test_outside_w8a8_contract_is_rejected(model, inputs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        sima_prepare_qat_model(model, inputs, "cpu")
