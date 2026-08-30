# **************************************************************************
#||                        SiMa.ai CONFIDENTIAL                          ||
#||   Unpublished Copyright (c) 2024 SiMa.ai, All Rights Reserved.       ||
# ***************************************************************************
import json

import pytest
import torch
from torch.ao.quantization import FakeQuantize, MinMaxObserver
from torch.fx import Graph, GraphModule

import sima_qat


class TinyProductSS2D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Conv2d(3, 2, 1)

    def forward(self, image):
        bounded = torch.sigmoid(image)
        product = bounded * bounded
        return self.projection(product + 0.0)


def _unit_product_graph() -> GraphModule:
    root = torch.nn.Module()
    root.product_grid = FakeQuantize(
        observer=MinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_affine,
    )
    with torch.no_grad():
        root.product_grid.activation_post_process.min_val.copy_(torch.tensor(0.0))
        root.product_grid.activation_post_process.max_val.copy_(torch.tensor(1.0))
        scale, zero_point = root.product_grid.activation_post_process.calculate_qparams()
        root.product_grid.scale.copy_(scale)
        root.product_grid.zero_point.copy_(zero_point)

    graph = Graph()
    value = graph.placeholder("value")
    product = graph.call_function(torch.ops.aten.mul.Tensor, (value, value))
    product.meta["nn_module_stack"] = {
        "block.ss2d": ("block.ss2d", "ExampleSS2D")
    }
    quantized = graph.call_module("product_grid", (product,))
    result = graph.call_function(torch.ops.aten.add.Tensor, (quantized, value))
    graph.output(result)
    model = GraphModule(root, graph)
    model.register_buffer("qat_frozen", torch.tensor([True], dtype=torch.bool))
    return model


def _two_group_product_graph() -> GraphModule:
    root = torch.nn.Module()
    for name in ("early_grid", "late_grid"):
        grid = FakeQuantize(
            observer=MinMaxObserver,
            quant_min=-128,
            quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_tensor_affine,
        )
        with torch.no_grad():
            grid.activation_post_process.min_val.copy_(torch.tensor(0.0))
            grid.activation_post_process.max_val.copy_(torch.tensor(1.0))
            scale, zero_point = grid.activation_post_process.calculate_qparams()
            grid.scale.copy_(scale)
            grid.zero_point.copy_(zero_point)
        root.add_module(name, grid)

    graph = Graph()
    value = graph.placeholder("value")
    early = graph.call_function(torch.ops.aten.mul.Tensor, (value, value))
    early.meta["nn_module_stack"] = {
        "early.ss2d": ("early.ss2d", "ExampleSS2D")
    }
    early_q = graph.call_module("early_grid", (early,))
    late = graph.call_function(torch.ops.aten.mul.Tensor, (early_q, value))
    late.meta["nn_module_stack"] = {
        "late.ss2d": ("late.ss2d", "ExampleSS2D")
    }
    late_q = graph.call_module("late_grid", (late,))
    graph.output(late_q)
    model = GraphModule(root, graph)
    model.register_buffer("qat_frozen", torch.tensor([True], dtype=torch.bool))
    return model


@pytest.mark.regression
def test_range_refinement_commits_only_improving_export_stable_grid():
    model = _unit_product_graph()
    graph_before = [(node.op, str(node.target)) for node in model.graph.nodes]
    original_scale = float(model.product_grid.scale)

    def evaluator(candidate, _data):
        # A deterministic stand-in for a task metric whose optimum is a 2x grid.
        scale = float(candidate.product_grid.scale)
        return {"quality": -abs(scale - 2.0 * original_scale)}

    report = sima_qat.refine_activation_ranges(
        model,
        [torch.ones(1)],
        evaluator,
        metric="quality",
        factors=(1.0, 2.0, 4.0),
    )

    assert report.committed
    assert report.best_factor == 2.0
    assert report.selected_fake_quantizers == 1
    assert float(model.product_grid.scale) == pytest.approx(2.0 * original_scale)
    observer_scale, observer_zero_point = (
        model.product_grid.activation_post_process.calculate_qparams()
    )
    torch.testing.assert_close(observer_scale, model.product_grid.scale)
    torch.testing.assert_close(observer_zero_point, model.product_grid.zero_point)
    assert [(node.op, str(node.target)) for node in model.graph.nodes] == graph_before


@pytest.mark.regression
def test_range_refinement_rolls_back_when_task_metric_does_not_improve():
    model = _unit_product_graph()
    original_scale = model.product_grid.scale.detach().clone()
    original_min = model.product_grid.activation_post_process.min_val.detach().clone()
    original_max = model.product_grid.activation_post_process.max_val.detach().clone()

    def evaluator(candidate, _data):
        return -abs(float(candidate.product_grid.scale) - float(original_scale))

    report = sima_qat.refine_activation_ranges(
        model,
        [torch.ones(1)],
        evaluator,
        factors=(1.0, 2.0),
    )

    assert not report.committed
    assert report.best_factor == 1.0
    torch.testing.assert_close(model.product_grid.scale, original_scale)
    torch.testing.assert_close(model.product_grid.activation_post_process.min_val, original_min)
    torch.testing.assert_close(model.product_grid.activation_post_process.max_val, original_max)


@pytest.mark.regression
def test_range_refinement_isolates_recurrent_modules_and_rejects_harmful_group():
    model = _two_group_product_graph()
    original = float(model.early_grid.scale)

    def evaluator(candidate, _data):
        early_error = abs(float(candidate.early_grid.scale) - original)
        late_error = abs(float(candidate.late_grid.scale) - 2.0 * original)
        return -(early_error + late_error)

    report = sima_qat.refine_activation_ranges(
        model,
        [torch.ones(1)],
        evaluator,
        factors=(1.0, 2.0),
    )

    assert [group["committed"] for group in report.groups] == [False, True]
    assert float(model.early_grid.scale) == pytest.approx(original)
    assert float(model.late_grid.scale) == pytest.approx(2.0 * original)


@pytest.mark.regression
def test_range_refinement_requires_frozen_model_and_reiterable_data():
    model = _unit_product_graph()
    model.qat_frozen.fill_(False)
    with pytest.raises(RuntimeError, match="freeze"):
        sima_qat.refine_activation_ranges(model, [torch.ones(1)], lambda *_: 1.0)

    model.qat_frozen.fill_(True)
    with pytest.raises(TypeError, match="re-iterable"):
        sima_qat.refine_activation_ranges(
            model,
            iter([torch.ones(1)]),
            lambda *_: 1.0,
        )


@pytest.mark.regression
def test_session_range_refinement_is_recorded_and_survives_export(tmp_path):
    image = torch.randn(1, 3, 4, 4)
    qat = sima_qat.prepare(TinyProductSS2D(), image, device="cpu")
    qat.calibrate([image] * 128)
    qat.freeze()

    product_grids = []
    for node in qat.model.graph.nodes:
        if node.op != "call_module" or not node.args:
            continue
        producer = node.args[0]
        if (
            isinstance(producer, torch.fx.Node)
            and producer.op == "call_function"
            and producer.target == torch.ops.aten.mul.Tensor
        ):
            module = qat.model.get_submodule(str(node.target))
            if id(module) not in {id(value) for value in product_grids}:
                product_grids.append(module)
    assert len(product_grids) == 1
    original_scale = float(product_grids[0].scale)

    selector = sima_qat.ActivationRangeSelector(
        producer_kinds=("multiply",),
        range_kinds=("any",),
        state_space_only=True,
    )
    report = qat.refine_ranges(
        [image],
        lambda _model, _data: -abs(
            float(product_grids[0].scale) - 2.0 * original_scale
        ),
        factors=(1.0, 2.0),
        policy=selector,
    )
    assert report.committed

    bundle = qat.export(tmp_path / "bundle")
    manifest = json.loads(bundle.manifest_path.read_text())
    assert manifest["range_refinements"][0]["committed"]
    assert bundle.quantize_linear_nodes > 0
