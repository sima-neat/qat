"""Model Compiler integration tests for float and QAT-exported ONNX."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import inspect
import json
import os
from pathlib import Path
import shutil
import tarfile
from typing import Any
import zipfile

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime
import pytest
import torch
from torch import nn

from sima_qat import (
    sima_export_onnx,
    sima_finalize_qat_model,
    sima_prepare_qat_model,
)


_RUN_COMPILER_TESTS = os.environ.get(
    "SIMA_QAT_RUN_MODEL_COMPILER_TESTS", "0"
).lower() in {"1", "true", "yes"}
_INPUT_SHAPE = (1, 3, 8, 8)
_TRANSPORT_OPS = {"QuantizeLinear", "DequantizeLinear"}

COMPILER_TEST_MARKS = [
    pytest.mark.model_compiler,
    pytest.mark.slow,
    pytest.mark.skipif(
        not _RUN_COMPILER_TESTS,
        reason=(
            "set SIMA_QAT_RUN_MODEL_COMPILER_TESTS=1 inside an activated "
            "Model Compiler environment"
        ),
    ),
]


class _CompileModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(inputs))


@dataclass(frozen=True)
class _QatContract:
    input_scale: float
    input_zero_point: int
    output_scale: float
    output_zero_point: int
    weight_scale: np.ndarray
    weight_zero_point: np.ndarray
    weight_codes: np.ndarray


@dataclass(frozen=True)
class _CompilerApi:
    ImporterParams: Any
    InputName: Any
    ModelFormat: Any
    ScalarType: Any
    default_quantization: Any
    load_model: Any
    target: Any


def _float_export(model: nn.Module, inputs: torch.Tensor, output: Path) -> None:
    kwargs = {
        "export_params": True,
        "opset_version": 17,
        "do_constant_folding": True,
        "input_names": ["input"],
        "output_names": ["output"],
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    torch.onnx.export(model, inputs, str(output), **kwargs)


def _axis(node: onnx.NodeProto) -> int | None:
    return next(
        (attribute.i for attribute in node.attribute if attribute.name == "axis"),
        None,
    )


def _initializer_values(
    model: onnx.ModelProto,
    name: str,
) -> np.ndarray:
    initializers = {item.name: item for item in model.graph.initializer}
    assert name in initializers, f"Expected initializer {name!r}"
    return numpy_helper.to_array(initializers[name])


def _scalar(values: np.ndarray) -> float | int:
    flat = np.asarray(values).reshape(-1)
    assert flat.size == 1
    return flat[0].item()


def _validate_fixture_operators(path: Path, *, qat: bool) -> None:
    model = onnx.load(path)
    onnx.checker.check_model(model, full_check=True)
    assert all(node.domain in ("", "ai.onnx") for node in model.graph.node)

    operators = {node.op_type for node in model.graph.node}
    compute_operators = operators - _TRANSPORT_OPS
    assert compute_operators == {"Conv", "Relu"}
    if qat:
        assert _TRANSPORT_OPS <= operators
    else:
        assert not (operators & _TRANSPORT_OPS)


def _qat_contract(path: Path) -> _QatContract:
    model = onnx.load(path)
    producers = {
        output: node
        for node in model.graph.node
        for output in node.output
    }

    input_quantizer = next(
        node
        for node in model.graph.node
        if node.op_type == "QuantizeLinear" and node.input[0] == "input"
    )
    output_dequantizer = producers[model.graph.output[0].name]
    assert output_dequantizer.op_type == "DequantizeLinear"
    output_quantizer = producers[output_dequantizer.input[0]]
    assert output_quantizer.op_type == "QuantizeLinear"

    convolution = next(
        node for node in model.graph.node if node.op_type == "Conv"
    )
    weight_dequantizer = producers[convolution.input[1]]
    assert weight_dequantizer.op_type == "DequantizeLinear"
    assert _axis(weight_dequantizer) == 0

    weight_codes = _initializer_values(model, weight_dequantizer.input[0])
    weight_scale = _initializer_values(model, weight_dequantizer.input[1])
    weight_zero_point = _initializer_values(model, weight_dequantizer.input[2])
    assert weight_codes.dtype == np.int8
    assert weight_scale.shape == weight_zero_point.shape
    assert weight_scale.shape == (weight_codes.shape[0],)

    return _QatContract(
        input_scale=float(
            _scalar(_initializer_values(model, input_quantizer.input[1]))
        ),
        input_zero_point=int(
            _scalar(_initializer_values(model, input_quantizer.input[2]))
        ),
        output_scale=float(
            _scalar(_initializer_values(model, output_quantizer.input[1]))
        ),
        output_zero_point=int(
            _scalar(_initializer_values(model, output_quantizer.input[2]))
        ),
        weight_scale=weight_scale.astype(np.float64),
        weight_zero_point=weight_zero_point.astype(np.int64),
        weight_codes=weight_codes,
    )


def _fixture_model() -> tuple[_CompileModel, torch.Tensor]:
    torch.manual_seed(7)
    inputs = torch.linspace(
        -1.5,
        1.5,
        steps=int(np.prod(_INPUT_SHAPE)),
        dtype=torch.float32,
    ).reshape(_INPUT_SHAPE)
    model = _CompileModel().eval()
    return model, inputs


@pytest.fixture(scope="session")
def compiler_target_name() -> str:
    """Return the explicit, validated qualification target name."""
    raw_target = os.environ.get("SIMA_QAT_MODEL_COMPILER_TARGET")
    if raw_target is None or not raw_target.strip():
        pytest.fail(
            "SIMA_QAT_MODEL_COMPILER_TARGET must be explicitly set to "
            "'modalix' or 'mlsoc' for compiler acceptance.",
            pytrace=False,
        )
    target_name = raw_target.strip().lower()
    if target_name not in {"modalix", "mlsoc"}:
        pytest.fail(
            "SIMA_QAT_MODEL_COMPILER_TARGET must be 'modalix' or 'mlsoc', "
            f"not {target_name!r}.",
            pytrace=False,
        )
    return target_name


@pytest.fixture(scope="session")
def compiler_artifact_root(
    tmp_path_factory: pytest.TempPathFactory,
    compiler_target_name: str,
) -> Path:
    """Return a basetemp whose final component matches the target."""
    artifact_root = tmp_path_factory.getbasetemp().resolve()
    if artifact_root.name != compiler_target_name:
        pytest.fail(
            "Compiler --basetemp must end with the selected target name; "
            f"target={compiler_target_name!r}, basetemp={artifact_root}.",
            pytrace=False,
        )
    return artifact_root


@pytest.fixture(scope="session")
def float_onnx(compiler_artifact_root: Path) -> Path:
    output_dir = compiler_artifact_root / "pre_qat" / "export"
    output_dir.mkdir(parents=True)
    output = output_dir / "pre_qat.onnx"
    model, inputs = _fixture_model()
    _float_export(model, inputs, output)
    _validate_fixture_operators(output, qat=False)
    return output


@pytest.fixture(scope="session")
def qat_onnx(compiler_artifact_root: Path) -> Path:
    output_dir = compiler_artifact_root / "post_qat" / "export"
    output_dir.mkdir(parents=True)
    output = output_dir / "post_qat.onnx"
    model, inputs = _fixture_model()

    qat_model = sima_prepare_qat_model(model, (inputs,), "cpu")
    qat_model.train()
    with torch.no_grad():
        for offset in (-0.25, 0.0, 0.25):
            qat_model(inputs + offset)
    qat_model = sima_finalize_qat_model(qat_model)
    sima_export_onnx(
        qat_model,
        (inputs,),
        str(output),
        input_names=["input"],
        output_names=["output"],
        device="cpu",
    )

    _validate_fixture_operators(output, qat=True)
    return output


@pytest.fixture(scope="session")
def compiler_api(compiler_target_name: str) -> _CompilerApi:
    try:
        from afe.apis.defines import (
            InputName,
            default_quantization,
            gen1_target,
            gen2_target,
        )
        from afe.apis.loaded_net import load_model
        from afe.ir.tensor_type import ScalarType
        from afe.load.importers.general_importer import (
            ImporterParams,
            ModelFormat,
        )
    except ImportError as error:
        pytest.fail(
            "Model Compiler Python packages are unavailable; run "
            "activate-model-compiler before this test. "
            f"Import error: {error}",
            pytrace=False,
        )

    if shutil.which("mla-masm") is None:
        pytest.fail(
            "Model Compiler assembler mla-masm is not on PATH; run "
            "activate-model-compiler before this test.",
            pytrace=False,
        )

    targets = {"modalix": gen2_target, "mlsoc": gen1_target}

    return _CompilerApi(
        ImporterParams=ImporterParams,
        InputName=InputName,
        ModelFormat=ModelFormat,
        ScalarType=ScalarType,
        default_quantization=default_quantization,
        load_model=load_model,
        target=targets[compiler_target_name],
    )


def _calibration_data(api: _CompilerApi) -> list[dict[Any, np.ndarray]]:
    nchw = _calibration_input()
    nhwc = np.ascontiguousarray(nchw.transpose(0, 2, 3, 1))
    return [{api.InputName("input"): nhwc}]


def _calibration_input() -> np.ndarray:
    return np.linspace(
        -1.0,
        1.0,
        num=int(np.prod(_INPUT_SHAPE)),
        dtype=np.float32,
    ).reshape(_INPUT_SHAPE)


def _lower_for_compile(
    api: _CompilerApi,
    model_path: Path,
    model_name: str,
) -> Any:
    importer = api.ImporterParams(
        format=api.ModelFormat.onnx,
        file_paths=[str(model_path)],
        input_names=["input"],
        input_shapes=[_INPUT_SHAPE],
        input_types=[api.ScalarType.float32],
        layout="NCHW",
        output_names=["output"],
    )
    loaded = api.load_model(importer, target=api.target)
    return loaded.quantize(
        calibration_data=_calibration_data(api),
        quantization_config=api.default_quantization,
        model_name=model_name,
    )


def _save_and_compile(model: Any, output_dir: Path, model_name: str) -> Path:
    output_dir.mkdir()
    model.save(model_name=model_name, output_directory=str(output_dir))
    model.compile(output_path=str(output_dir), batch_size=1)

    archive = output_dir / f"{model_name}_mpk.tar.gz"
    assert archive.is_file() and archive.stat().st_size > 0
    with tarfile.open(archive, "r:gz") as bundle:
        files = [member for member in bundle.getmembers() if member.isfile()]
        executables = [
            member
            for member in files
            if member.name.endswith((".elf", ".lm")) and member.size > 0
        ]
        assert executables
        assert any(
            member.name.endswith("_mla_stats.yaml") and member.size > 0
            for member in files
        )
        mpk_json = next(
            member for member in files if member.name.endswith("_mpk.json")
        )
        assert mpk_json.size > 0
        stream = bundle.extractfile(mpk_json)
        assert stream is not None
        manifest = json.load(stream)
        assert manifest["name"] == model_name
        mla_plugins = [
            plugin
            for plugin in manifest["plugins"]
            if plugin.get("processor") == "MLA"
        ]
        assert mla_plugins
        referenced_executables = {
            plugin["resources"]["executable"] for plugin in mla_plugins
        }
        assert any(
            executable.name in referenced_executables
            for executable in executables
        )
    return archive


def _serialized_quantized_net(
    sima_archive: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    try:
        import yaml
    except ImportError as error:
        pytest.fail(f"Model Compiler is missing PyYAML: {error}", pytrace=False)

    with zipfile.ZipFile(sima_archive) as archive:
        yaml_name = next(
            name for name in archive.namelist() if name.endswith(".yaml")
        )
        npz_name = next(
            name for name in archive.namelist() if name.endswith(".npz")
        )
        documents = yaml.safe_load(archive.read(yaml_name))
        root = next(
            item
            for item in documents
            if isinstance(item, dict) and "nodes" in item
        )
        with np.load(BytesIO(archive.read(npz_name))) as values:
            arrays = {name: values[name].copy() for name in values.files}
    return root, arrays


def _assert_qat_lowering_preserved(
    output_dir: Path,
    model_name: str,
    contract: _QatContract,
) -> None:
    root, arrays = _serialized_quantized_net(
        output_dir / f"{model_name}.sima"
    )
    quantize_node = next(
        node for node in root["nodes"] if node["ir"].get("op") == "quantize"
    )
    input_scale, input_zero_point = quantize_node["ir"]["attrs"][
        "channel_params"
    ][0]
    assert input_scale == pytest.approx(1.0 / contract.input_scale, rel=1e-6)
    assert input_zero_point == contract.input_zero_point

    mla = next(
        node for node in root["nodes"] if node["ir"].get("node_type") == "subgraph"
    )
    convolution = next(
        node
        for node in mla["ir"]["nodes"]
        if node["ir"].get("quant", {}).get("op") == "conv"
    )
    quantization = convolution["ir"]["quant"]
    assert quantization["scale"] == pytest.approx(
        1.0 / contract.output_scale,
        rel=1e-6,
    )
    assert quantization["zero_point"] == contract.output_zero_point
    assert quantization["per_channel"] is True

    shifts = arrays[quantization["requant"]["shift"]].astype(np.float64)
    actual_weights = arrays[quantization["weight_quant_data"]]
    factors = (
        contract.weight_scale
        * quantization["scale"]
        * np.power(2.0, shifts)
        / input_scale
    )
    centered = contract.weight_codes.astype(np.float64) - (
        contract.weight_zero_point[:, None, None, None]
    )
    folded = np.rint(centered * factors[:, None, None, None]).astype(np.int8)
    expected_weights = folded.transpose(2, 3, 1, 0)[:, :, :, None, :]
    np.testing.assert_array_equal(actual_weights, expected_weights)


def _assert_qat_execution_matches(
    model: Any,
    api: _CompilerApi,
    onnx_path: Path,
    output_scale: float,
) -> None:
    nchw = _calibration_input()
    sdk_outputs = list(model.execute(inputs=_calibration_data(api)[0]))
    assert len(sdk_outputs) == 1
    sdk_output = np.asarray(sdk_outputs[0])

    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    session = onnxruntime.InferenceSession(
        str(onnx_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    ort_output = session.run(None, {"input": nchw})[0]
    if sdk_output.shape == (
        ort_output.shape[0],
        ort_output.shape[2],
        ort_output.shape[3],
        ort_output.shape[1],
    ):
        sdk_output = sdk_output.transpose(0, 3, 1, 2)
    assert sdk_output.shape == ort_output.shape
    max_abs_error = float(np.max(np.abs(sdk_output - ort_output)))
    assert max_abs_error <= (2.0 * output_scale) + 1e-6
