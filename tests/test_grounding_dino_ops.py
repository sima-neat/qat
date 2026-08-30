import pytest
import torch
import torch.nn.functional as F

from sima_qat.qat_api import (
    sima_qat_activation_sensitivity,
    sima_prepare_qat_model,
    sima_qat_activation_diagnostics,
)


class QuantizedCoordinateGridSample(torch.nn.Module):
    def forward(self, data, grid):
        valid = (
            (grid[..., 0] >= -1.0)
            & (grid[..., 0] <= 127.0 / 128.0)
            & (grid[..., 1] >= -1.0)
            & (grid[..., 1] <= 127.0 / 128.0)
        )
        sampled = F.grid_sample(
            data,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return sampled * valid.unsqueeze(1).to(sampled.dtype)


class TinyAttention(torch.nn.Module):
    def forward(self, query, key, value):
        score = torch.bmm(query, key.transpose(1, 2))
        probability = torch.softmax(score, dim=-1)
        return torch.bmm(probability, value)


@pytest.mark.regression
def test_attention_probability_domain_is_tagged_and_numerically_reported():
    query = torch.randn(1, 4, 8)
    key = torch.randn(1, 16, 8)
    value = torch.randn(1, 16, 8)
    prepared = sima_prepare_qat_model(
        TinyAttention(),
        (query, key, value),
        "cpu",
        full_range_ste=True,
        learn_scales=True,
    )
    prepared(query, key, value)

    probability_modules = [
        module
        for module in prepared.modules()
        if getattr(module, "sima_domain_kind", "") == "attention_probability"
    ]
    assert probability_modules
    state_before = [
        (
            module.fake_quant_enabled.detach().clone(),
            module.observer_enabled.detach().clone(),
        )
        for module in prepared.modules()
        if isinstance(module, torch.ao.quantization.FakeQuantizeBase)
    ]

    report = sima_qat_activation_diagnostics(
        prepared, (query, key, value)
    )
    probability_rows = [
        row for row in report
        if row["domain_kind"] == "attention_probability"
    ]
    assert probability_rows
    assert probability_rows[0]["step_over_rms"] > 0
    assert probability_rows[0]["relative_rmse"] >= 0
    state_after = [
        (module.fake_quant_enabled, module.observer_enabled)
        for module in prepared.modules()
        if isinstance(module, torch.ao.quantization.FakeQuantizeBase)
    ]
    for expected, actual in zip(state_before, state_after):
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])


@pytest.mark.regression
def test_task_sensitivity_ranks_activation_grids_without_changing_gradients():
    query = torch.randn(1, 4, 8, requires_grad=True)
    key = torch.randn(1, 16, 8, requires_grad=True)
    value = torch.randn(1, 16, 8, requires_grad=True)
    prepared = sima_prepare_qat_model(
        TinyAttention(),
        (query.detach(), key.detach(), value.detach()),
        "cpu",
        full_range_ste=True,
        learn_scales=True,
    )
    prepared(query.detach(), key.detach(), value.detach())
    candidates = [
        name
        for name, module in prepared.named_modules()
        if getattr(module, "sima_domain_kind", "")
        in {"attention_probability", "activation_matmul_output"}
    ]
    state_before = [
        (
            module.fake_quant_enabled.detach().clone(),
            module.observer_enabled.detach().clone(),
        )
        for module in prepared.modules()
        if isinstance(module, torch.ao.quantization.FakeQuantizeBase)
    ]
    rows = sima_qat_activation_sensitivity(
        prepared,
        (query, key, value),
        lambda output: output.float().square().mean(),
        candidate_names=candidates,
    )

    assert rows
    assert all(row["taylor_l1"] >= 0 for row in rows)
    assert all(row["quantization_rmse"] >= 0 for row in rows)
    assert rows == sorted(rows, key=lambda row: row["taylor_l1"], reverse=True)
    assert all(parameter.grad is None for parameter in prepared.parameters())
    state_after = [
        (module.fake_quant_enabled, module.observer_enabled)
        for module in prepared.modules()
        if isinstance(module, torch.ao.quantization.FakeQuantizeBase)
    ]
    for expected, actual in zip(state_before, state_after):
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])


@pytest.mark.regression
def test_grid_sample_uses_fixed_signed_coordinate_grid_and_bool_mask_is_not_quantized():
    data = torch.randn(1, 4, 8, 8)
    grid = torch.empty(1, 3, 5, 2).uniform_(-1.25, 1.25)
    prepared = sima_prepare_qat_model(
        QuantizedCoordinateGridSample(),
        (data, grid),
        "cpu",
        full_range_ste=True,
        learn_scales=True,
    )
    assert prepared(data, grid).shape == (1, 4, 3, 5)

    grid_sample = next(
        node
        for node in prepared.graph.nodes
        if node.op == "call_function"
        and node.target == torch.ops.aten.grid_sampler.default
    )
    coordinate_fake_quant = prepared.get_submodule(grid_sample.args[1].target)
    assert coordinate_fake_quant.sima_domain_kind == "deformable_grid"
    torch.testing.assert_close(
        coordinate_fake_quant.scale,
        torch.tensor([1.0 / 128.0]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        coordinate_fake_quant.zero_point,
        torch.tensor([0], dtype=coordinate_fake_quant.zero_point.dtype),
        rtol=0,
        atol=0,
    )

    for node in prepared.graph.nodes:
        if node.op != "call_module" or "activation_post_process" not in str(node.target):
            continue
        source = node.args[0]
        value = source.meta.get("val")
        assert not (
            isinstance(value, torch.Tensor) and value.dtype == torch.bool
        ), f"bool mask was fake-quantized at {node.target}"
