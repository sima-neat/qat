"""Integration tests for graph transformations surrounding quantized operators."""

import onnx
import pytest
import torch

from sima_qat import (
    sima_export_onnx,
    sima_finalize_qat_model,
    sima_freeze_qat,
    sima_prepare_qat_model,
)


pytestmark = pytest.mark.regression


class DropoutModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 1)
        self.dropout = torch.nn.Dropout()
        self.conv2 = torch.nn.Conv2d(4, 4, 1)

    def forward(self, inputs):
        return self.conv2(self.dropout(self.conv1(inputs)))


class FlattenModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 1)
        self.linear = torch.nn.Linear(4 * 4 * 4, 5)

    def forward(self, inputs):
        return self.linear(torch.flatten(self.conv(inputs), 1))


def test_dropout_is_removed_during_preparation() -> None:
    inputs = torch.randn(2, 3, 8, 8)

    prepared = sima_prepare_qat_model(DropoutModel(), (inputs,), "cpu")

    assert all(
        node.target not in (torch.ops.aten.dropout.default, torch.ops.aten.dropout_.default)
        for node in prepared.graph.nodes
    )


def test_flatten_does_not_create_consecutive_onnx_qdq(tmp_path) -> None:
    inputs = torch.randn(2, 3, 4, 4)
    prepared = sima_prepare_qat_model(FlattenModel(), (inputs,), "cpu")
    prepared(inputs)
    sima_freeze_qat(prepared)
    finalized = sima_finalize_qat_model(prepared)
    output_path = tmp_path / "flatten.onnx"
    sima_export_onnx(finalized, (inputs,), str(output_path), device="cpu")

    model = onnx.load(output_path)
    consumers = {
        input_name: node
        for node in model.graph.node
        for input_name in node.input
    }
    graph_outputs = {output.name for output in model.graph.output}
    for node in model.graph.node:
        if node.op_type != "DequantizeLinear" or node.output[0] in graph_outputs:
            continue
        assert consumers[node.output[0]].op_type != "QuantizeLinear"
