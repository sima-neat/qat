"""Device-kwarg rewriting across the QAT lifecycle."""

import pytest
import torch

from sima_qat import (
    sima_export_onnx,
    sima_finalize_qat_model,
    sima_freeze_qat,
    sima_prepare_qat_model,
)
from sima_qat.qat_api import check_graph_nodes, device_modifier_ops


pytestmark = pytest.mark.regression

CUDA = pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable"
))


def _export_model(device, *, with_state=True):
    root = torch.nn.Module()
    graph = torch.fx.Graph()
    inputs = graph.placeholder("inputs")
    output = inputs
    if with_state:
        root.register_parameter("weight", torch.nn.Parameter(torch.tensor(2.0, device=device)))
        root.register_buffer("bias", torch.tensor(0.5, device=device))
        output = graph.call_function(torch.mul, (output, graph.get_attr("weight")))
        output = graph.call_function(torch.add, (output, graph.get_attr("bias")))
    ramp = graph.call_function(torch.ops.aten.arange.default, (4,), {
        "dtype": torch.float32, "device": torch.device(device),
    })
    graph.output(graph.call_function(torch.add, (output, ramp)))
    return torch.fx.GraphModule(root, graph).eval()


def _assert_export_on_cpu(model, inputs):
    assert all(tensor.device.type == "cpu" for tensor in [*model.parameters(), *model.buffers()])
    if isinstance(model, torch.fx.GraphModule):
        assert all(torch.device(device).type == "cpu" for device in _device_kwargs(model))
    assert all(tensor.device.type == "cpu" for tensor in inputs)


@pytest.mark.parametrize("source_device", ["cpu", CUDA])
@pytest.mark.parametrize("return_device", [None, "cpu", CUDA])
def test_real_export_restores_device_or_applies_override(
    source_device, return_device, tmp_path, monkeypatch
):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    model = _export_model(source_device)
    inputs = torch.arange(4, dtype=torch.float32, device=source_device)
    snapshot = inputs.clone()
    expected = model(inputs).detach().cpu()
    original_kwargs = _device_kwargs(model)
    native_export = torch.onnx.export

    def export_on_cpu(model, export_inputs, *args, **kwargs):
        _assert_export_on_cpu(model, export_inputs)
        return native_export(model, export_inputs, *args, **kwargs)

    monkeypatch.setattr(torch.onnx, "export", export_on_cpu)
    output_path = tmp_path / "device_restore.onnx"
    options = {} if return_device is None else {"device": return_device}
    returned = sima_export_onnx(model, (inputs,), str(output_path), **options)
    expected_device = torch.device(source_device if return_device is None else return_device)
    assert returned is model
    assert all(tensor.device.type == expected_device.type for tensor in [*model.parameters(), *model.buffers()])
    assert all(torch.device(device).type == expected_device.type for device in _device_kwargs(model))
    if return_device is None:
        assert _device_kwargs(model) == original_kwargs
    torch.testing.assert_close(inputs, snapshot, rtol=0, atol=0)
    torch.testing.assert_close(model(inputs.to(expected_device)).detach().cpu(), expected)
    onnx.checker.check_model(onnx.load(output_path))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(output_path), options, providers=["CPUExecutionProvider"])
    actual = session.run(None, {session.get_inputs()[0].name: inputs.cpu().numpy()})[0]
    torch.testing.assert_close(torch.from_numpy(actual), expected)


@pytest.mark.parametrize("source_device", ["cpu", CUDA])
@pytest.mark.parametrize("return_device", [None, "cpu", CUDA])
def test_failed_export_restores_original_device(
    source_device, return_device, tmp_path, monkeypatch
):
    model = _export_model(source_device)
    inputs = torch.arange(4, dtype=torch.float32, device=source_device)
    snapshot = inputs.clone()
    original_kwargs = _device_kwargs(model)
    failure = RuntimeError("export failed")

    def fail_export(model, export_inputs, *_args, **_kwargs):
        _assert_export_on_cpu(model, export_inputs)
        export_inputs[0].zero_()
        raise failure

    monkeypatch.setattr(torch.onnx, "export", fail_export)
    options = {} if return_device is None else {"device": return_device}
    with pytest.raises(RuntimeError, match="export failed") as caught:
        sima_export_onnx(model, (inputs,), str(tmp_path / "failed.onnx"), **options)
    assert caught.value is failure
    assert all(tensor.device.type == source_device for tensor in [*model.parameters(), *model.buffers()])
    assert _device_kwargs(model) == original_kwargs
    torch.testing.assert_close(inputs, snapshot, rtol=0, atol=0)
    assert torch.isfinite(model(inputs)).all()


@pytest.mark.parametrize("source_device", ["cpu", CUDA])
def test_export_restores_tensor_free_graph_device(source_device, tmp_path, monkeypatch):
    model = _export_model(source_device, with_state=False)
    inputs = torch.ones(4, device=source_device)
    original_kwargs = _device_kwargs(model)
    monkeypatch.setattr(torch.onnx, "export", lambda model, inputs, *_args, **_kwargs: _assert_export_on_cpu(model, inputs))
    assert sima_export_onnx(model, (inputs,), str(tmp_path / "tensor_free.onnx")) is model
    assert _device_kwargs(model) == original_kwargs
    assert torch.isfinite(model(inputs)).all()


@pytest.mark.parametrize("source_device", ["cpu", CUDA])
def test_export_restores_eager_model_device(source_device, tmp_path, monkeypatch):
    model = torch.nn.Linear(4, 2).to(source_device).eval()
    inputs = torch.ones(1, 4, device=source_device)
    monkeypatch.setattr(torch.onnx, "export", lambda model, inputs, *_args, **_kwargs: _assert_export_on_cpu(model, inputs))
    assert sima_export_onnx(model, (inputs,), str(tmp_path / "eager.onnx")) is model
    assert all(parameter.device.type == source_device for parameter in model.parameters())
    assert torch.isfinite(model(inputs)).all()


class RandomMaskModel(torch.nn.Module):
    def forward(self, inputs):
        mask = torch.empty(
            [inputs.shape[0], 1, 1, 1],
            dtype=inputs.dtype,
            device=inputs.device,
        ).bernoulli_(0.95)
        return inputs * mask


def _device_kwargs(model):
    return [
        node.kwargs["device"]
        for node in model.graph.nodes
        if node.target in device_modifier_ops
    ]


def test_finalization_moves_the_model_and_device_kwargs_to_cpu() -> None:
    inputs = torch.randn(2, 3, 8, 8)
    prepared = sima_prepare_qat_model(RandomMaskModel(), (inputs,), "cpu")
    assert _device_kwargs(prepared)
    assert all(device == "cpu" for device in _device_kwargs(prepared))

    check_graph_nodes(prepared, "cuda")
    assert all(device == "cuda" for device in _device_kwargs(prepared))

    check_graph_nodes(prepared, "cpu")
    prepared(inputs)
    sima_freeze_qat(prepared)
    check_graph_nodes(prepared, "cuda")
    finalized = sima_finalize_qat_model(prepared)

    assert all(tensor.device.type == "cpu" for tensor in finalized.parameters())
    assert all(tensor.device.type == "cpu" for tensor in finalized.buffers())
    assert all(device == "cpu" for device in _device_kwargs(finalized))
    assert torch.isfinite(finalized(inputs)).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_training_model_finalizes_and_exports_on_cpu(
    tmp_path, monkeypatch
) -> None:
    inputs = torch.randn(2, 3, 8, 8, device="cuda")
    prepared = sima_prepare_qat_model(RandomMaskModel(), (inputs,), "cuda")
    prepared(inputs)
    sima_freeze_qat(prepared)

    finalized = sima_finalize_qat_model(prepared)
    assert all(tensor.device.type == "cpu" for tensor in finalized.buffers())
    assert all(device == "cpu" for device in _device_kwargs(finalized))

    observed = {}

    def record_export(model, export_inputs, *_args, **_kwargs):
        observed["model_devices"] = {
            tensor.device.type for tensor in model.buffers()
        }
        observed["input_devices"] = {
            tensor.device.type for tensor in export_inputs
        }

    monkeypatch.setattr(torch.onnx, "export", record_export)
    exported_model = sima_export_onnx(
        finalized, (inputs,), str(tmp_path / "random_mask.onnx")
    )

    assert observed == {
        "model_devices": {"cpu"},
        "input_devices": {"cpu"},
    }
    assert inputs.device.type == "cuda"
    assert all(tensor.device.type == "cpu" for tensor in exported_model.buffers())
