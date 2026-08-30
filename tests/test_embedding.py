import torch
import pytest

from sima_qat.qat_api import (
    sima_finalize_qat_model,
    sima_freeze_qat,
    sima_prepare_qat_model,
)


class EmbeddingProjection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 8)
        self.projection = torch.nn.Linear(8, 4)

    def forward(self, indices):
        return self.projection(self.embedding(indices))


@pytest.mark.regression
def test_embedding_table_is_w8_and_indices_remain_int64():
    indices = torch.tensor([[1, 2, 3]], dtype=torch.int64)
    prepared = sima_prepare_qat_model(
        EmbeddingProjection(),
        (indices,),
        "cpu",
        full_range_ste=True,
        learn_scales=True,
    )
    actual = prepared(indices)
    assert actual.shape == (1, 3, 4)

    embedding = next(
        node
        for node in prepared.graph.nodes
        if node.op == "call_function"
        and node.target == torch.ops.aten.embedding.default
    )
    weight_fake_quant = embedding.args[0]
    assert weight_fake_quant.op == "call_module"
    module = prepared.get_submodule(weight_fake_quant.target)
    assert module.dtype == torch.int8
    assert module.qscheme == torch.per_tensor_symmetric
    assert embedding.args[1].meta["val"].dtype == torch.int64

    sima_freeze_qat(prepared)
    finalized = sima_finalize_qat_model(prepared)
    assert finalized(indices).shape == actual.shape
