"""Assertions shared by operator-level tests."""

from __future__ import annotations

import torch
from torch.ao.quantization.fake_quantize import FakeQuantizeBase
from torch.fx import GraphModule

from sima_qat import sima_prepare_qat_model
from sima_qat.qat_api import (
    _SHIFT_AWARE_OPS,
    _fake_quant_module,
    _find_output_fake_quant,
)

from .cases import OperatorCase


def prepare_case(case: OperatorCase) -> tuple[GraphModule, tuple[torch.Tensor, ...]]:
    torch.manual_seed(0)
    inputs = case.input_factory()
    prepared = sima_prepare_qat_model(case.model_factory(), inputs, "cpu")
    prepared(*inputs)
    return prepared, inputs


def fake_quantizers(model: GraphModule) -> list[FakeQuantizeBase]:
    return [module for module in model.modules() if isinstance(module, FakeQuantizeBase)]


def weighted_nodes(model: GraphModule) -> list[torch.fx.Node]:
    return [
        node
        for node in model.graph.nodes
        if node.op == "call_function" and node.target in _SHIFT_AWARE_OPS
    ]


def assert_shift_realizable(model: GraphModule) -> None:
    nodes = weighted_nodes(model)
    assert nodes, "weighted operator case did not contain a shift-aware operation"
    for node in nodes:
        input_fq = _fake_quant_module(model, node.args[0])
        weight_fq = _fake_quant_module(model, node.args[1])
        output_fq = _find_output_fake_quant(model, node)
        assert input_fq is not None
        assert weight_fq is not None
        assert output_fq is not None

        ratio = input_fq.scale * weight_fq.scale / output_fq.scale
        shifts = -torch.ceil(torch.log2(ratio))
        normalized = ratio * torch.pow(2.0, shifts)
        assert torch.all(normalized <= 1.0)
        assert torch.all(normalized > 0.99999)
