#**************************************************************************
#||                        SiMa.ai CONFIDENTIAL                          ||
#||   Unpublished Copyright (c) 2024 SiMa.ai, All Rights Reserved.       ||
#**************************************************************************
# NOTICE: All information contained herein remains the property of SiMa.ai.
# The source code is confidential and proprietary.
#**************************************************************************
"""Public lifecycle API for the Dynamo-free SiMa FX QAT backend."""

from __future__ import annotations

import inspect
import operator
import re
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.ao.quantization import FakeQuantize
from torch.ao.quantization.fake_quantize import disable_observer, enable_fake_quant
from torch.ao.quantization.observer import PerChannelMinMaxObserver
from torch.ao.quantization.quantize_fx import prepare_qat_fx
from torch.fx import GraphModule, Node, symbolic_trace

import torch.ao.nn.intrinsic.qat as intrinsic_qat

from sima_qat.onnx_ops import normalize_qdq_model, prepare_export_copy
from sima_qat.sima_quantizer import (
    get_sima_backend_config,
    get_sima_qconfig_mapping,
)


__all__ = [
    "sima_prepare_qat_model",
    "sima_finalize_qat_model",
    "sima_export_onnx",
]

def _torch_major_minor(raw_version: str) -> Tuple[int, int]:
    """Extract Torch's numeric major/minor pair without third-party helpers."""
    match = re.match(r"^\s*(\d+)\.(\d+)(?=\D|$)", raw_version)
    if match is None:
        raise RuntimeError(f"Unable to parse torch version {raw_version!r}")
    return int(match.group(1)), int(match.group(2))


_TORCH_MAJOR_MINOR = _torch_major_minor(torch.__version__)
if not (2, 3) <= _TORCH_MAJOR_MINOR < (2, 9):
    raise RuntimeError(
        "Sima QAT only supports torch version 2.3.x through 2.8.x, "
        f"found {torch.__version__}"
    )


_QAT_SCHEMA_VERSION = 1
_DROPOUT_TYPES = tuple(
    dropout_type
    for dropout_type in (
        nn.Dropout,
        nn.Dropout1d,
        nn.Dropout2d,
        nn.Dropout3d,
        nn.AlphaDropout,
        nn.FeatureAlphaDropout,
    )
    if dropout_type is not None
)
_FUNCTIONAL_DROPOUT_TARGETS = {
    F.dropout,
    F.dropout1d,
    F.dropout2d,
    F.dropout3d,
    F.alpha_dropout,
    F.feature_alpha_dropout,
}
_CONV_BN_TYPES = tuple(
    module_type
    for module_type in (
        getattr(intrinsic_qat, "ConvBn1d", None),
        getattr(intrinsic_qat, "ConvBn2d", None),
        getattr(intrinsic_qat, "ConvBn3d", None),
        getattr(intrinsic_qat, "ConvBnReLU1d", None),
        getattr(intrinsic_qat, "ConvBnReLU2d", None),
        getattr(intrinsic_qat, "ConvBnReLU3d", None),
    )
    if module_type is not None
)

device_modifier_ops = [
    torch.ops.aten.empty.memory_format,
    torch.ops.aten.arange.default,
    torch.ops.aten.full.default,
]
_DEVICE_FACTORY_TARGETS = {
    torch.empty,
    torch.arange,
    torch.full,
    *device_modifier_ops,
}


def _get_module_device(module: nn.Module) -> torch.device:
    for parameter in module.parameters():
        return parameter.device
    for buffer in module.buffers():
        return buffer.device
    return torch.device("cpu")


def _move_value_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move_value_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_value_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {
            key: _move_value_to_device(item, device)
            for key, item in value.items()
        }
    return value


def _validate_device(device: Union[str, torch.device]) -> torch.device:
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for QAT, but CUDA is unavailable.")
    return requested


@contextmanager
def _dropout_modules_as_identity(module: nn.Module) -> Iterator[None]:
    replacements = []

    def replace(parent: nn.Module) -> None:
        for name, child in list(parent.named_children()):
            if isinstance(child, _DROPOUT_TYPES):
                replacements.append((parent, name, child))
                setattr(parent, name, nn.Identity())
            else:
                replace(child)

    replace(module)
    try:
        yield
    finally:
        for parent, name, child in reversed(replacements):
            setattr(parent, name, child)


def _remove_functional_dropout(graph_module: GraphModule) -> GraphModule:
    for node in list(graph_module.graph.nodes):
        if (
            node.op == "call_function"
            and node.target in _FUNCTIONAL_DROPOUT_TARGETS
            and node.args
        ):
            node.replace_all_uses_with(node.args[0])
            graph_module.graph.erase_node(node)
    graph_module.graph.lint()
    graph_module.recompile()
    return graph_module


def _normalize_fx_call_forms(graph_module: GraphModule) -> GraphModule:
    """Canonicalize public FX call forms used by quantization patterns."""

    for node in list(graph_module.graph.nodes):
        if node.op == "call_method" and node.target in ("add", "mul"):
            target = torch.add if node.target == "add" else torch.mul
            with graph_module.graph.inserting_before(node):
                replacement = graph_module.graph.call_function(
                    target, args=node.args, kwargs=node.kwargs
                )
            replacement.meta.update(node.meta)
            node.replace_all_uses_with(replacement)
            graph_module.graph.erase_node(node)
            continue

        unary_keyword_call = (
            node.op == "call_module"
            and isinstance(
                graph_module.get_submodule(node.target),
                (nn.Hardtanh, nn.SiLU),
            )
        ) or (
            node.op == "call_function"
            and node.target in (F.hardtanh, F.silu)
        )
        if unary_keyword_call and not node.args and "input" in node.kwargs:
            kwargs = dict(node.kwargs)
            input_value = kwargs.pop("input")
            node.args = (input_value,)
            node.kwargs = kwargs

    graph_module.graph.lint()
    graph_module.recompile()
    return graph_module


def _prepare_float_graph(input_graph: nn.Module) -> GraphModule:
    with _dropout_modules_as_identity(input_graph):
        traced = symbolic_trace(input_graph)
    traced = _remove_functional_dropout(traced)
    return _normalize_fx_call_forms(traced)


def _remove_alias_fake_quants_after_getitem(
    graph_module: GraphModule,
) -> GraphModule:
    """Remove duplicate calls to an observer already applied before indexing."""

    erased_targets = []
    for node in list(graph_module.graph.nodes):
        if (
            node.op != "call_function"
            or node.target is not operator.getitem
            or not node.args
        ):
            continue
        source = node.args[0]
        if getattr(source, "op", None) != "call_module":
            continue
        source_module = graph_module.get_submodule(source.target)
        for user in list(node.users):
            if user.op != "call_module":
                continue
            user_module = graph_module.get_submodule(user.target)
            if (
                isinstance(source_module, FakeQuantize)
                and user_module is source_module
            ):
                user.replace_all_uses_with(node)
                erased_targets.append(user.target)
                graph_module.graph.erase_node(user)

    live_targets = {
        node.target
        for node in graph_module.graph.nodes
        if node.op == "call_module"
    }
    for target in erased_targets:
        if target in live_targets:
            continue
        parent_name, _, child_name = target.rpartition(".")
        parent = (
            graph_module.get_submodule(parent_name)
            if parent_name
            else graph_module
        )
        delattr(parent, child_name)

    graph_module.graph.lint()
    graph_module.recompile()
    return graph_module


def _is_activation_fake_quant_node(
    graph_module: GraphModule, value: Any
) -> bool:
    if not isinstance(value, Node) or value.op != "call_module":
        return False
    module = graph_module.get_submodule(value.target)
    return isinstance(module, FakeQuantize) and not bool(module.is_per_channel)


def _unwrap_activation_fake_quant(
    graph_module: GraphModule, value: Any
) -> Tuple[Any, Optional[Node]]:
    if not _is_activation_fake_quant_node(graph_module, value):
        return value, None
    if not value.args:
        raise RuntimeError("Activation fake-quant node has no input.")
    return value.args[0], value


def _delete_call_module_if_unused(
    graph_module: GraphModule, target: str
) -> None:
    if any(
        node.op == "call_module" and node.target == target
        for node in graph_module.graph.nodes
    ):
        return
    parent_name, _, child_name = target.rpartition(".")
    parent = (
        graph_module.get_submodule(parent_name)
        if parent_name
        else graph_module
    )
    if hasattr(parent, child_name):
        delattr(parent, child_name)


def _bypass_activation_fake_quant(
    graph_module: GraphModule, consumer: Node, fake_quant: Node, source: Any
) -> None:
    consumer.replace_input_with(fake_quant, source)
    if fake_quant.users:
        return
    target = str(fake_quant.target)
    graph_module.graph.erase_node(fake_quant)
    _delete_call_module_if_unused(graph_module, target)


def _is_conv_node(graph_module: GraphModule, node: Any) -> bool:
    if not isinstance(node, Node):
        return False
    if node.op == "call_module":
        module = graph_module.get_submodule(node.target)
        conv_types = (nn.Conv1d, nn.Conv2d, nn.Conv3d) + _CONV_BN_TYPES
        return isinstance(module, conv_types)
    return node.op == "call_function" and node.target in (
        F.conv1d,
        F.conv2d,
        F.conv3d,
    )


def _is_add_node(node: Any) -> bool:
    return isinstance(node, Node) and (
        (node.op == "call_function" and node.target in (operator.add, torch.add))
        or (node.op == "call_method" and node.target in ("add", "add_"))
    )


def _is_hardtanh_node(graph_module: GraphModule, node: Node) -> bool:
    if node.op == "call_module":
        return isinstance(graph_module.get_submodule(node.target), nn.Hardtanh)
    if node.op == "call_function":
        return node.target is F.hardtanh
    return node.op == "call_method" and node.target in ("hardtanh", "hardtanh_")


def _primary_input(node: Node) -> Optional[Any]:
    if node.args:
        return node.args[0]
    return node.kwargs.get("input")


def _binary_operands(node: Node) -> Optional[Tuple[Any, Any]]:
    if len(node.args) >= 2:
        return node.args[0], node.args[1]

    left = node.args[0] if node.args else node.kwargs.get("input")
    right = node.kwargs.get("other")
    if left is None or right is None:
        return None
    return left, right


def _restore_legacy_quantization_regions(
    graph_module: GraphModule,
) -> GraphModule:
    """Keep legacy fused regions free of interior activation fake quantization."""

    cleanup_count = 0

    # Conv/Conv-BN -> Hardtanh and Add -> Hardtanh are one quantized region.
    for node in list(graph_module.graph.nodes):
        if not _is_hardtanh_node(graph_module, node):
            continue
        input_value = _primary_input(node)
        if input_value is None:
            continue
        source, fake_quant = _unwrap_activation_fake_quant(
            graph_module, input_value
        )
        if fake_quant is None or not (
            _is_conv_node(graph_module, source) or _is_add_node(source)
        ):
            continue
        _bypass_activation_fake_quant(
            graph_module, node, fake_quant, source
        )
        cleanup_count += 1

    # Conv -> Add/Mul(constant) is also one region. Constants stay floating
    # point; only the region output is activation fake-quantized.
    for node in list(graph_module.graph.nodes):
        is_arithmetic = (
            node.op == "call_function"
            and node.target in (operator.add, operator.mul, torch.add, torch.mul)
        ) or (
            node.op == "call_method"
            and node.target in ("add", "add_", "mul", "mul_")
        )
        if not is_arithmetic:
            continue

        operands = _binary_operands(node)
        if operands is None:
            continue
        unwrapped = [
            _unwrap_activation_fake_quant(graph_module, argument)
            for argument in operands
        ]
        conv_indices = [
            index
            for index, (source, _) in enumerate(unwrapped)
            if _is_conv_node(graph_module, source)
        ]
        if len(conv_indices) != 1:
            continue
        constant_index = 1 - conv_indices[0]
        constant_source = unwrapped[constant_index][0]
        if isinstance(constant_source, Node) and constant_source.op != "get_attr":
            continue

        for source, fake_quant in unwrapped:
            if fake_quant is None:
                continue
            _bypass_activation_fake_quant(
                graph_module, node, fake_quant, source
            )
            cleanup_count += 1

    graph_module.meta["qat_legacy_region_cleanup_count"] = cleanup_count
    graph_module.graph.lint()
    graph_module.recompile()
    return graph_module


def _observer_is_initialized(observer: PerChannelMinMaxObserver) -> bool:
    return (
        observer.min_val.numel() > 0
        and observer.max_val.numel() > 0
        and torch.isfinite(observer.min_val).all()
        and torch.isfinite(observer.max_val).all()
    )


def _freeze_weight_observer(
    observer: PerChannelMinMaxObserver,
    fallback_weight: Optional[Tensor] = None,
) -> FakeQuantize:
    if not _observer_is_initialized(observer):
        if fallback_weight is not None:
            observer(fallback_weight.detach())
        else:
            raise RuntimeError(
                "A weight observer has no statistics. Run at least one prepared "
                "model forward pass before finalization."
            )

    constructor = FakeQuantize.with_args(
        observer=type(observer),
        quant_min=observer.quant_min,
        quant_max=observer.quant_max,
        dtype=observer.dtype,
        qscheme=observer.qscheme,
        ch_axis=observer.ch_axis,
        reduce_range=False,
        eps=float(observer.eps),
    )
    frozen = constructor().to(observer.min_val.device)
    frozen.activation_post_process.load_state_dict(observer.state_dict())
    scale, zero_point = frozen.activation_post_process.calculate_qparams()
    with torch.no_grad():
        frozen.scale.resize_(scale.shape)
        frozen.scale.copy_(scale)
        frozen.zero_point.resize_(zero_point.shape)
        frozen.zero_point.copy_(zero_point)
    frozen.disable_observer()
    frozen.enable_fake_quant()
    frozen.eval()
    return frozen


class _FrozenConvBn(nn.Module):
    """BN-folded convolution retaining frozen per-channel fake quantization."""

    def __init__(
        self,
        conv: nn.Module,
        weight_observer: PerChannelMinMaxObserver,
        with_relu: bool,
    ) -> None:
        super().__init__()
        self.conv = conv
        self.weight_fake_quant = _freeze_weight_observer(
            weight_observer,
            fallback_weight=conv.weight,
        )
        self.with_relu = with_relu

    def forward(self, inputs: Tensor) -> Tensor:
        weight = self.weight_fake_quant(self.conv.weight)
        output = self.conv._conv_forward(inputs, weight, self.conv.bias)
        if self.with_relu:
            output = torch.relu(output)
        return output


def _fold_conv_bn_modules(module: nn.Module) -> int:
    folded_count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, _CONV_BN_TYPES):
            child.eval()
            with_relu = "ReLU" in type(child).__name__
            weight_observer = child.weight_fake_quant
            if not isinstance(weight_observer, PerChannelMinMaxObserver):
                raise RuntimeError(
                    "Conv-BN fusion expected a PerChannelMinMaxObserver weight state."
                )
            float_module = child.to_float()
            conv = float_module[0] if with_relu else float_module
            replacement = _FrozenConvBn(conv, weight_observer, with_relu)
            replacement.eval()
            setattr(module, name, replacement)
            folded_count += 1
        else:
            folded_count += _fold_conv_bn_modules(child)
    return folded_count


def _freeze_remaining_weight_observers(module: nn.Module) -> int:
    if isinstance(module, FakeQuantize):
        return 0

    frozen_count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, PerChannelMinMaxObserver):
            fallback_weight = getattr(module, "weight", None)
            setattr(
                module,
                name,
                _freeze_weight_observer(child, fallback_weight=fallback_weight),
            )
            frozen_count += 1
        else:
            frozen_count += _freeze_remaining_weight_observers(child)
    return frozen_count


class SimaQatWrapper(GraphModule):
    """FX GraphModule with explicit QAT lifecycle and checkpoint state."""

    _tag_to_id: Dict[str, int] = {
        "scaffold": 0,
        "fq": 1,
    }

    def __init__(self, source: GraphModule, label: str):
        if not isinstance(source, GraphModule):
            raise RuntimeError(f"SiMa QAT requires an FX GraphModule, found {type(source)}.")
        if label not in self._tag_to_id:
            raise RuntimeError(f"QAT state must be one of {tuple(self._tag_to_id)}.")

        super().__init__(source, source.graph, source._get_name())
        self.meta.update(getattr(source, "meta", {}))
        self.meta["qat_state"] = label
        self.meta["qat_backend"] = "fx_graph_mode"
        self.register_buffer(
            "qat_state",
            torch.tensor([self._tag_to_id[label]], dtype=torch.int8),
        )
        self.register_buffer(
            "qat_backend_version",
            torch.tensor([_QAT_SCHEMA_VERSION], dtype=torch.int16),
        )
        nn.Module.train(self, label == "scaffold")

    @property
    def qat_stage(self) -> str:
        state_id = int(self.qat_state.item())
        for label, candidate in self._tag_to_id.items():
            if candidate == state_id:
                return label
        raise RuntimeError(f"Unknown QAT state id {state_id}.")

    def _set_qat_stage(self, label: str) -> None:
        self.qat_state.fill_(self._tag_to_id[label])
        self.meta["qat_state"] = label

    def train(self, mode: bool = True) -> "SimaQatWrapper":
        if mode and self.qat_stage == "fq":
            raise RuntimeError(
                "The QAT model is finalized in fake-quant inference mode; "
                "training mode is disallowed."
            )
        nn.Module.train(self, mode)
        return self

    def eval(self) -> "SimaQatWrapper":
        return self.train(False)

    @staticmethod
    def _checkpoint_scalar(
        state_dict: Mapping[str, Any], key: str, description: str
    ) -> int:
        try:
            value = torch.as_tensor(state_dict[key])
        except Exception as error:
            raise RuntimeError(f"Invalid {description} value at {key!r}.") from error
        if value.numel() != 1:
            raise RuntimeError(f"{description} at {key!r} must contain one value.")
        return int(value.reshape(-1)[0].item())

    def _validate_checkpoint_schema(
        self, state_dict: Mapping[str, Any], prefix: str
    ) -> None:
        backend_key = f"{prefix}qat_backend_version"
        state_key = f"{prefix}qat_state"

        if backend_key not in state_dict:
            raise RuntimeError(
                "Checkpoint has no FX QAT backend version. Legacy PT2E checkpoints "
                "are not directly compatible with this backend."
            )
        checkpoint_version = self._checkpoint_scalar(
            state_dict, backend_key, "QAT backend version"
        )
        if checkpoint_version != _QAT_SCHEMA_VERSION:
            raise RuntimeError(
                f"Unsupported QAT checkpoint schema {checkpoint_version}; "
                f"expected {_QAT_SCHEMA_VERSION}."
            )

        if state_key not in state_dict:
            raise RuntimeError("State dictionary does not represent a QAT model.")
        checkpoint_state = self._checkpoint_scalar(
            state_dict, state_key, "QAT state"
        )
        expected_state = self._tag_to_id[self.qat_stage]
        if checkpoint_state != expected_state:
            raise RuntimeError(
                f"Model QAT state {expected_state} does not match checkpoint "
                f"QAT state {checkpoint_state}."
            )

    def _load_from_state_dict(
        self,
        state_dict: Mapping[str, Any],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        self._validate_checkpoint_schema(state_dict, prefix)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,

            unexpected_keys,
            error_msgs,
        )

def sima_prepare_qat_model(
    input_graph: nn.Module,
    inputs: Tuple[Any, ...],
    device: Union[str, torch.device],
) -> GraphModule:
    """Prepare a symbolically traceable model for signed-int8 FX QAT.

    This path uses FX graph-mode quantization only. It does not import or call
    Dynamo, torch.export, PT2E prepare, or PT2E conversion APIs.
    """

    if not isinstance(input_graph, nn.Module):
        raise RuntimeError(
            "Input graph to prepare must be an nn.Module, "
            f"found {type(input_graph)}."
        )
    if not isinstance(inputs, tuple):
        raise RuntimeError("Example inputs must be supplied as a tuple.")
    if isinstance(input_graph, SimaQatWrapper):
        if input_graph.qat_stage == "scaffold":
            return input_graph
        raise RuntimeError("A finalized QAT model cannot be prepared again.")

    requested_device = _validate_device(device)
    cpu = torch.device("cpu")
    input_graph.to(cpu)
    input_graph.train(True)
    cpu_inputs = _move_value_to_device(inputs, cpu)

    print(f"Preparing {input_graph._get_name()} with the SiMa FX QAT backend...")
    try:
        float_graph = _prepare_float_graph(input_graph)
        prepared = prepare_qat_fx(
            float_graph,
            get_sima_qconfig_mapping(),
            cpu_inputs,
            backend_config=get_sima_backend_config(),
        )
        prepared = _remove_alias_fake_quants_after_getitem(prepared)
        prepared = _restore_legacy_quantization_regions(prepared)
    except Exception as error:
        raise RuntimeError(
            "SiMa FX QAT preparation failed. The model forward must be "
            "symbolically traceable with static Python control flow. No Dynamo "
            "fallback is used by this backend."
        ) from error

    wrapped = SimaQatWrapper(prepared, label="scaffold")
    wrapped.to(requested_device)
    wrapped.train(True)
    return check_graph_nodes(wrapped, requested_device)


def sima_finalize_qat_model(qat_model: GraphModule) -> GraphModule:
    """Freeze observers and produce an inference-only fake-quantized model."""

    if not isinstance(qat_model, SimaQatWrapper):
        raise RuntimeError(
            "Finalize expects a model returned by sima_prepare_qat_model, "
            f"found {type(qat_model)}."
        )
    if qat_model.qat_stage == "fq":
        return qat_model

    print("Freezing QAT observers, weights, and Conv-BN fusions...")
    device = _get_module_device(qat_model)
    qat_model.eval()
    qat_model.apply(disable_observer)
    qat_model.apply(enable_fake_quant)
    folded_count = _fold_conv_bn_modules(qat_model)
    weight_count = _freeze_remaining_weight_observers(qat_model)
    if folded_count + weight_count == 0:
        qat_model.meta["qat_weight_fake_quant_count"] = 0
    else:
        qat_model.meta["qat_weight_fake_quant_count"] = folded_count + weight_count

    for parameter in qat_model.parameters():
        parameter.requires_grad_(False)
    qat_model._set_qat_stage("fq")
    qat_model.to(device)
    qat_model.eval()
    qat_model.graph.lint()
    qat_model.recompile()
    return qat_model


def sima_export_onnx(
    qat_model: nn.Module,
    inputs: Tuple[Any, ...],
    output_file: str,
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> GraphModule:
    """Export a finalized QAT model as a standard opset-17 ONNX Q/DQ graph."""

    if not isinstance(qat_model, SimaQatWrapper):
        raise RuntimeError(
            "Export expects a model returned by sima_prepare_qat_model, "
            f"found {type(qat_model)}."
        )
    if qat_model.qat_stage != "fq":
        raise RuntimeError("The QAT model must be finalized before ONNX export.")
    if not isinstance(inputs, tuple):
        raise RuntimeError("Export inputs must be supplied as a tuple.")

    original_device = _get_module_device(qat_model)
    restore_device = (
        _validate_device(device)
        if device is not None
        else original_device
    )
    export_inputs = _move_value_to_device(inputs, torch.device("cpu"))
    shadow, shadowed_weights = prepare_export_copy(qat_model)
    shadow = check_graph_nodes(shadow, torch.device("cpu"))

    export_kwargs = {
        "export_params": True,
        "opset_version": 17,
        "do_constant_folding": True,
        "input_names": input_names,
        "output_names": output_names,
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    with torch.no_grad():
        torch.onnx.export(
            shadow,
            export_inputs,
            output_file,
            **export_kwargs,
        )
    normalize_qdq_model(output_file, shadowed_weights)

    qat_model.to(restore_device)
    return check_graph_nodes(qat_model, restore_device)


def check_graph_nodes(
    prepared_mod: GraphModule,
    device: Union[str, torch.device],
) -> GraphModule:
    """Update explicit tensor-factory device kwargs in an FX graph."""

    if not isinstance(prepared_mod, GraphModule):
        return prepared_mod

    target_device = torch.device(device)
    changed = False
    for node in prepared_mod.graph.nodes:
        if node.target in _DEVICE_FACTORY_TARGETS:
            new_kwargs = dict(node.kwargs)
            new_kwargs["device"] = target_device
            node.kwargs = new_kwargs
            changed = True
    if changed:
        prepared_mod.graph.lint()
        prepared_mod.recompile()
    return prepared_mod
