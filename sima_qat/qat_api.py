#**************************************************************************
#||                        SiMa.ai CONFIDENTIAL                          ||
#||   Unpublished Copyright (c) 2024 SiMa.ai, All Rights Reserved.       ||
#**************************************************************************
# NOTICE:  All information contained herein is, and remains the property of
# SiMa.ai. The intellectual and technical concepts contained herein are
# proprietary to SiMa and may be covered by U.S. and Foreign Patents,
# patents in process, and are protected by trade secret or copyright law.
#
# Dissemination of this information or reproduction of this material is
# strictly forbidden unless prior written permission is obtained from
# SiMa.ai.  Access to the source code contained herein is hereby forbidden
# to anyone except current SiMa.ai employees, managers or contractors who
# have executed Confidentiality and Non-disclosure agreements explicitly
# covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes information
# that is confidential and/or proprietary, and is a trade secret, of SiMa.ai.
#
# ANY REPRODUCTION, MODIFICATION, DISTRIBUTION, PUBLIC PERFORMANCE, OR PUBLIC
# DISPLAY OF OR THROUGH USE OF THIS SOURCE CODE WITHOUT THE EXPRESS WRITTEN
# CONSENT OF SiMa.ai IS STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE
# LAWS AND INTERNATIONAL TREATIES. THE RECEIPT OR POSSESSION OF THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS TO
# REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE, USE, OR
# SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
#**************************************************************************
import math
import warnings
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import torch
from packaging import version

if (version.parse(torch.__version__) < version.parse("2.3.0") or
    version.parse(torch.__version__) >= version.parse("2.9.0")):
    raise RuntimeError(f"Sima QAT only supports torch version 2.3.x through 2.8.x, found {torch.__version__}")

from torch import optim, nn, utils, Tensor

try:
    # torch <= 2.4: pre-autograd capture lives here.
    from torch._export import capture_pre_autograd_graph as _capture_pre_autograd_graph

    def _export_training_graph(mod, inputs):
        return _capture_pre_autograd_graph(mod, inputs)
except ImportError:
    # torch >= 2.5: capture_pre_autograd_graph was removed in favor of export_for_training.
    from torch.export import export_for_training as _export_for_training

    def _export_training_graph(mod, inputs):
        return _export_for_training(mod, inputs).module()
from torch.ao.quantization.quantize_pt2e import (
  prepare_qat_pt2e,
  convert_pt2e,
)
from torch.ao.quantization import (
    disable_observer,
    move_exported_model_to_eval,
    move_exported_model_to_train,
)
from torch.ao.quantization.fake_quantize import FakeQuantizeBase
from torch.fx.graph_module import GraphModule


from sima_qat import onnx_ops
from sima_qat.sima_quantizer import SimaQuantizer, get_sima_quantization_config


device_modifier_ops = [
    torch.ops.aten.empty.memory_format,
    torch.ops.aten.arange.default,
    torch.ops.aten.full.default
]


def _get_module_device(module: nn.Module) -> torch.device:
    for tensor in list(module.parameters()) + list(module.buffers()):
        return tensor.device
    return torch.device("cpu")


def _move_value_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move_value_to_device(v, device) for v in value)
    if isinstance(value, list):
        return [_move_value_to_device(v, device) for v in value]
    if isinstance(value, dict):
        return {k: _move_value_to_device(v, device) for k, v in value.items()}
    return value


def sima_prepare_qat_model(
    input_graph: nn.Module,
    inputs: Tuple,
    device: torch.device,
    shift_aware: bool = True,
    activation_observer: Optional[str] = None,
    full_range_ste: Optional[bool] = None,
    learn_scales: Optional[bool] = None,
) -> GraphModule:
    """This function is the first transformation needed to perform QAT on a Pytorch model. It takes an
    eager-mode reference to the ML model and produces an FX version of the graph with special annotations
    needed for QAT. Internally, it will scaffold the graph using observers and fakequant nodes needed 
    during the training process.

    Note:
        The Pytorch graph on which QAT is performed may be a full model, or a subsection of a model.
        QAT optimization will be limited to the graph given by the `input_graph` argument. This region
        must always be contained to the level of hierarchy as described by a single nn.Module.

    Args:
        input_graph: an eager-mode `nn.Module` representing the model on which QAT is to be performed.
            This may be a full model, or may be a sub-section of an ML model.
        inputs: a `Tuple` of tensor inputs, sized to the correct shape as the input to the given 
            `input_graph`. This data can be randomly generated. It is used during the preparation 
            process to build the compiled FX representation.
        device: a Pytorch `device` identifier. This will be the device on which the prepared model will
            be located after the preparation step is complete.
        shift_aware: when ``True`` (the default), fake-quantize weights during training and prepare
            them for SiMa's power-of-two requantization. Set this to ``False`` to retain the legacy
            observer-only weight behavior.
        activation_observer: activation range estimator: ``moving_average``,
            ``minmax``, or ``histogram``. The default preserves the installed
            environment's policy.
        full_range_ste: use a full-range straight-through activation fake
            quantizer. This keeps gradients outside the observed INT8 range.
        learn_scales: make activation scales trainable when ``full_range_ste``
            is enabled.

    Returns:
        GraphModule: a compiled version of the given graph with QAT annotations, ready to begin training.
    """
    if not isinstance(input_graph, nn.Module):
        raise RuntimeError(f"Input graph to prepare function must be of type nn.Module, found {type(input_graph)}")
    
    if isinstance(input_graph, GraphModule):
        return input_graph

    print(f"Making QAT annotations on model {input_graph._get_name()}...")
    # We have to move things to the CPU to do the scaffolding. We will return the model to the proper
    # device when we are done.
    input_graph.to("cpu")
    m = _export_training_graph(input_graph, inputs)
    m = replace_dropout(m)

    cfg = get_sima_quantization_config(
        is_qat=True,
        shift_aware=shift_aware,
        activation_observer=activation_observer,
        full_range_ste=full_range_ste,
        learn_scales=learn_scales,
    )
    quantizer = SimaQuantizer().set_global(cfg)
    gm = prepare_qat_pt2e(m, quantizer)
    sima_mod = SimaQatWrapper(source=gm, label='scaffold', shift_aware=shift_aware)
    sima_mod.to(device)
    sima_mod.train()
    sima_mod = check_graph_nodes(sima_mod, device)

    return sima_mod


_SHIFT_AWARE_OPS = {
    torch.ops.aten.conv1d.default,
    torch.ops.aten.conv2d.default,
    torch.ops.aten.linear.default,
}
_MIN_REQUANT_SHIFT = 0
_MAX_REQUANT_SHIFT = 31


def _resolve_attr(module: nn.Module, target: str) -> Any:
    value: Any = module
    for atom in target.split("."):
        value = getattr(value, atom)
    return value


def _fake_quant_module(module: GraphModule, node: Any) -> Optional[FakeQuantizeBase]:
    if getattr(node, "op", None) != "call_module":
        return None
    candidate = module.get_submodule(node.target)
    return candidate if isinstance(candidate, FakeQuantizeBase) else None


def _find_output_fake_quant(module: GraphModule, op_node: Any) -> Optional[FakeQuantizeBase]:
    """Find the nearest per-tensor fake quantizer following an annotated op.

    PT2E can place the output observer after a fused activation such as ReLU,
    rather than directly after the convolution. Stop at another weighted op so
    that an unrelated downstream quantizer cannot be selected accidentally.
    """
    pending = list(op_node.users)
    visited = set()
    while pending:
        node = pending.pop(0)
        if node in visited:
            continue
        visited.add(node)
        fake_quant = _fake_quant_module(module, node)
        if fake_quant is not None and fake_quant.qscheme in (
            torch.per_tensor_affine,
            torch.per_tensor_symmetric,
        ):
            return fake_quant
        if node.op == "call_function" and node.target in _SHIFT_AWARE_OPS:
            continue
        pending.extend(node.users)
    return None


def _safe_power_of_two_weight_scale(
    input_scale: Tensor,
    output_scale: Tensor,
    weight: Tensor,
) -> Tensor:
    """Return per-channel scales satisfying sx * sw / sy ~= 2**-shift.

    Scales are rounded one float32 ULP toward zero when necessary. This keeps
    the normalized multiplier on the safe side of the Model Compiler's
    power-of-two boundary, preventing a value infinitesimally above the
    boundary from selecting the next shift and a 0.5 correction factor.
    """
    if weight.ndim < 1:
        raise RuntimeError("Shift-aware weight tensors must have an output-channel dimension")

    sx = float(input_scale.reshape(-1)[0].detach().cpu())
    sy = float(output_scale.reshape(-1)[0].detach().cpu())
    if not math.isfinite(sx) or not math.isfinite(sy) or sx <= 0.0 or sy <= 0.0:
        raise RuntimeError(f"Observed activation scales must be positive, found input={sx}, output={sy}")

    reduce_dims = tuple(range(1, weight.ndim))
    max_abs = weight.detach().abs().amax(dim=reduce_dims).to(torch.float64).cpu()
    if not bool(torch.isfinite(max_abs).all()):
        raise RuntimeError("Shift-aware weights must be finite")
    required_scale = torch.clamp(max_abs / 127.0, min=torch.finfo(torch.float32).tiny)
    minimum_ratio = (sx / sy) * required_scale
    if bool((minimum_ratio > 1.0).any()):
        raise RuntimeError(
            "Shift-aware weights cannot fit the Model Compiler requantization range at shift 0"
        )
    unclamped_shift = torch.floor(-torch.log2(minimum_ratio))
    shifts = unclamped_shift.clamp(_MIN_REQUANT_SHIFT, _MAX_REQUANT_SHIFT).to(torch.int32)

    target_ratio = torch.pow(torch.tensor(2.0, dtype=torch.float64), -shifts.to(torch.float64))
    scales = ((sy / sx) * target_ratio).to(torch.float32)

    # Work with the exact float32 values AFE will ingest. One or two ULP steps
    # are normally sufficient. Always start one ULP below the exact boundary so
    # alternate float32 multiplication order cannot move it to the unsafe side.
    zero = torch.zeros_like(scales)
    scales = torch.nextafter(scales, zero)
    for _ in range(4):
        imported_ratio = (float(torch.tensor(sx, dtype=torch.float32)) * scales.to(torch.float64)
                          / float(torch.tensor(sy, dtype=torch.float32)))
        too_high = imported_ratio > target_ratio
        if not bool(too_high.any()):
            break
        scales = torch.where(too_high, torch.nextafter(scales, zero), scales)
    imported_ratio = (float(torch.tensor(sx, dtype=torch.float32)) * scales.to(torch.float64)
                      / float(torch.tensor(sy, dtype=torch.float32)))
    if bool((imported_ratio > target_ratio).any()):
        raise RuntimeError("Unable to represent shift-aware weight scale safely in float32")

    return scales.to(weight.device)


def sima_freeze_qat(qat_model: GraphModule) -> GraphModule:
    """Freeze QAT observers and lock Model Compiler-compatible power-of-two weight scales.

    Call this after observer warm-up, then continue fine-tuning with fake quantization
    enabled. For a model prepared with ``shift_aware=False``, this only freezes the
    existing activation observers and therefore retains the legacy behavior.
    """
    if not isinstance(qat_model, GraphModule):
        raise RuntimeError(f"Input graph to freeze function must be a GraphModule, found {type(qat_model)}")
    if bool(getattr(qat_model, "qat_frozen", torch.tensor([0])).item()):
        return qat_model

    shift_aware = bool(getattr(qat_model, "shift_aware_qat", torch.tensor([0])).item())
    # PT2E conversion serializes qparams recalculated from observers, not a
    # learnable fake-quantizer's live ``exp(log_scale)`` buffer.  Synchronize
    # activation grids to those exact export qparams before deriving coupled
    # weight scales.  Otherwise a one-ULP activation mismatch can change the
    # compiler's power-of-two shift after ONNX import.
    activation_qparams = []
    for fake_quant in qat_model.modules():
        if (
            "FakeQuant" not in type(fake_quant).__name__
            or getattr(fake_quant, "is_per_channel", False)
        ):
            continue
        scale, zero_point = fake_quant.activation_post_process.calculate_qparams()
        activation_qparams.append(
            (
                fake_quant,
                scale.to(device=fake_quant.scale.device, dtype=fake_quant.scale.dtype),
                zero_point.to(
                    device=fake_quant.zero_point.device,
                    dtype=fake_quant.zero_point.dtype,
                ),
            )
        )
    activation_qparams_by_id = {
        id(fake_quant): (scale, zero_point)
        for fake_quant, scale, zero_point in activation_qparams
    }
    locked_scales = []
    skipped = []
    if shift_aware:
        for node in qat_model.graph.nodes:
            if node.op != "call_function" or node.target not in _SHIFT_AWARE_OPS:
                continue
            if len(node.args) < 2:
                skipped.append(node.name)
                continue

            input_fq = _fake_quant_module(qat_model, node.args[0])
            weight_fq = _fake_quant_module(qat_model, node.args[1])
            output_fq = _find_output_fake_quant(qat_model, node)
            weight_node = node.args[1].args[0] if getattr(node.args[1], "args", ()) else None
            if (
                input_fq is None
                or weight_fq is None
                or output_fq is None
                or getattr(weight_node, "op", None) != "get_attr"
                or weight_fq.qscheme not in (torch.per_channel_affine, torch.per_channel_symmetric)
            ):
                skipped.append(node.name)
                continue

            weight = _resolve_attr(qat_model, weight_node.target)
            input_scale = activation_qparams_by_id.get(
                id(input_fq), (input_fq.scale, input_fq.zero_point)
            )[0]
            output_scale = activation_qparams_by_id.get(
                id(output_fq), (output_fq.scale, output_fq.zero_point)
            )[0]
            scales = _safe_power_of_two_weight_scale(
                input_scale,
                output_scale,
                weight,
            )
            locked_scales.append((weight_fq, scales))

        if skipped:
            raise RuntimeError(
                "Shift-aware QAT could not determine complete input/weight/output quantization "
                f"parameters for {len(skipped)} op(s): {', '.join(skipped)}"
            )

    # Do not mutate observer state until every shift-aware layer has validated.
    qat_model.apply(disable_observer)
    for fake_quant, scale, zero_point in activation_qparams:
        fake_quant.scale.resize_(scale.shape).copy_(scale)
        fake_quant.zero_point.resize_(zero_point.shape).copy_(zero_point)
    for weight_fq, scales in locked_scales:
        # Keep observer state consistent with the locked qparams as well. This
        # makes checkpoints self-describing and avoids restoring an old scale
        # if observers are explicitly re-enabled by a training framework.
        weight_observer = weight_fq.activation_post_process
        locked_range = scales * 127.0
        # Observer multiplication/division can move a nominal scale by one
        # float32 ULP. Move the persisted range toward zero until the observer
        # returns a scale no larger than the already-safe target boundary,
        # then use that exact export scale for the live fake quantizer.
        zero_range = torch.zeros_like(locked_range)
        for _ in range(8):
            weight_observer.min_val.resize_(scales.shape).copy_(-locked_range)
            weight_observer.max_val.resize_(scales.shape).copy_(locked_range)
            export_scale, export_zero_point = weight_observer.calculate_qparams()
            export_scale = export_scale.to(
                device=scales.device, dtype=scales.dtype
            )
            too_high = export_scale > scales
            if not bool(too_high.any()):
                break
            locked_range = torch.where(
                too_high,
                torch.nextafter(locked_range, zero_range),
                locked_range,
            )
        if bool((export_scale > scales).any()):
            raise RuntimeError(
                "Unable to persist a safe shift-aware weight scale through "
                "the PT2E observer qparam contract"
            )
        weight_fq.scale.resize_(export_scale.shape).copy_(export_scale)
        weight_fq.zero_point.resize_(export_zero_point.shape).copy_(
            export_zero_point.to(
                device=weight_fq.zero_point.device,
                dtype=weight_fq.zero_point.dtype,
            )
        )

    # Frozen activation scales are now the observer/export qparams. Disable
    # the learned-scale forward so it cannot restore exp(log_scale) after the
    # exact synchronization above. The parameter remains in the state dict for
    # topology-compatible checkpoint loading but is no longer consulted.
    for fake_quant, _, _ in activation_qparams:
        if hasattr(fake_quant, "learn_scale"):
            fake_quant.learn_scale = False

    qat_model.qat_frozen.fill_(1)
    return qat_model


def _ensure_bn_tracking_meta(gm: GraphModule) -> None:
    """torch >= 2.8 `convert_pt2e` QAT bn-folding reads `node.meta["source_fn_stack"]`
    on the BatchNorm `num_batches_tracked += 1` in-place add nodes, but graphs produced
    by `export_for_training` don't always populate it -> KeyError. Those nodes have the
    shape `aten.add_.Tensor(get_attr, 1)`; tag them so torch's loop erases them (its
    intent for BN tracking nodes)."""
    if not hasattr(gm, "graph"):
        return
    bn_tag = [("bn_num_batches_tracked", torch.nn.modules.batchnorm.BatchNorm2d)]
    for node in gm.graph.nodes:
        if (
            node.op == "call_function"
            and node.target == torch.ops.aten.add_.Tensor
            and len(node.args) >= 2
            and getattr(node.args[0], "op", None) == "get_attr"
            and node.args[1] == 1
            and "source_fn_stack" not in node.meta
        ):
            node.meta["source_fn_stack"] = bn_tag


def sima_finalize_qat_model(qat_model: GraphModule) -> GraphModule:
    """This function takes a QAT scaffolded model which has completed the training regimen and 
    converts it to an inference-only (via fakequant) form. Once this process is complete, the model 
    can no longer be trained, and is intended for export via ONNX.

    Args:
        qat_model: a trained QAT model to be converted into inference-only form.

    Returns:
        GraphModule: an inference-only version of the QAT model, which can be run in Pytorch 
            `eval(True)` mode, or exported via ONNX.
    """
    if not isinstance(qat_model, nn.Module):
        raise RuntimeError(f"Input graph to finalize function must be of type nn.Module, found {type(qat_model)}")
    
    if not isinstance(qat_model, GraphModule):
        return qat_model
    shift_aware = bool(getattr(qat_model, "shift_aware_qat", torch.tensor([0])).item())
    if shift_aware and not bool(getattr(qat_model, "qat_frozen", torch.tensor([0])).item()):
        warnings.warn(
            "Finalizing a shift-aware model before sima_freeze_qat(); scales will be locked now. "
            "For best accuracy, freeze earlier and fine-tune with the locked scales.",
            stacklevel=2,
        )
        sima_freeze_qat(qat_model)
    print(f"Removing QAT scaffold and quantizing network ...")
    device = _get_module_device(qat_model)
    _ensure_bn_tracking_meta(qat_model)
    m = convert_pt2e(qat_model, use_reference_representation=False)
    sima_mod = SimaQatWrapper(source=m, label='fq', shift_aware=shift_aware)
    sima_mod.to(device)
    # We must call eval() to invoke internal functions to put the GraphModule in eval state. Once we are
    # in FQ mode, we always remain in eval mode.
    sima_mod.eval()
    sima_mod = replace_batchnorm(sima_mod)
    # These two buffers are training/checkpoint control-plane state, not model
    # tensors. Keeping them on the inference-only wrapper perturbs legacy ONNX
    # export's generated initializer numbering even though neither buffer is
    # reachable from the graph. Finalized models cannot resume training, so
    # remove the dead metadata before export and keep the qualified Q/DQ graph
    # byte-stable across package revisions.
    del sima_mod.shift_aware_qat
    del sima_mod.qat_frozen
    return sima_mod


def sima_export_onnx(
    qat_model: nn.Module,
    inputs: Tuple[Tensor],
    output_file: str,
    input_names: Optional[List[str]] = None,
    output_names: Optional[List[str]] = None,
    device: Optional[Union[str, torch.device]] = None,
    export_device: Optional[Union[str, torch.device]] = None,
) -> GraphModule:
    """This function exports a finalized QAT model to ONNX format.

    Args:
        qat_model: The finalized ML model to export to ONNX.
        inputs: a `Tuple` of tensor inputs used to infer the proper shapes of all internal tensors.
            This is used by the Pytorch ONNX exporter.
        output_file: the path name of the .onnx file to generate.
        input_names: a list of tensor names used to label the ONNX model inputs.
        output_names: a list of tensor names used to label the ONNX model outputs.
        device: optional device to restore the returned model to after CPU ONNX export.
            If unset, the model returns to its original device.
        export_device: optional device on which to trace and constant-fold the
            ONNX graph. CPU is the portable default. Set this explicitly when
            reproducing a device-qualified export whose constant-folding
            contract was established on an accelerator.
    """
    if not isinstance(qat_model, nn.Module):
        raise RuntimeError(f"Input graph to export function must be of type nn.Module, found {type(qat_model)}")

    original_device = _get_module_device(qat_model)
    restore_device = torch.device(device) if device is not None else original_device
    selected_export_device = torch.device(export_device or "cpu")
    if selected_export_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Cannot export the QAT model on CUDA because CUDA is unavailable."
        )

    qat_model.to(selected_export_device)
    export_inputs = _move_value_to_device(inputs, selected_export_device)
    qat_model = check_graph_nodes(qat_model, device=selected_export_device)
    torch.onnx.export(
        qat_model,
        export_inputs[0],
        output_file,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names = input_names,
        output_names = output_names,
    )
    onnx_ops.canonicalize_repeated_input_concat_qdq(output_file)

    if restore_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Cannot restore exported QAT model to CUDA because CUDA is unavailable.")
    qat_model.to(restore_device)
    qat_model = check_graph_nodes(qat_model, device=str(restore_device))
    return qat_model

class SimaQatWrapper(GraphModule):
    """This is a Sima-defined wrapper which allows Pytorch GraphModule objects to behave 
    like `nn.Module`s at training time. It is used so that commonly called Pytorch functions
    work correctly when QAT is invoked.

    Note:
        This wrapper can only be created from an existing GraphModule. The source GraphModules 
        are created by Pytorch at each control point during QAT runtime.
    """
    _tag_to_id: Dict = {
        'scaffold': 0,
        'fq': 1,
    }

    def __init__(self, source: GraphModule, label: str, shift_aware: bool = True):
        """This constructor creates a wrapper from a GraphModule. We can only create this object
        from an existing GraphModule class. Every time we create a wrapper, we also need to
        specify which phase of QAT we are representing, since each phase has different 
        restrictions as to what is permissible.

        Args:
            source: A `GraphModule` produced by Pytorch call to some PT2E initialization. Must be
                a compiled FX graph.
            label: One of the legal enumerated labels matching the phase of the QAT process.
            shift_aware: whether this model uses SiMa power-of-two-aware weight quantization.
        """
        if not isinstance(source, GraphModule):
            raise RuntimeError(f"Sima supports only compiled graphs, found {type(source)}")
        
        if label not in self._tag_to_id:
            raise RuntimeError(f"Error: label must be one of: {self._tag_to_id.keys()}")
        
        d = source.__dict__
        for k in SimaQatWrapper.__dict__.keys():
            if k in d:
                del d[k]
        self.__dict__.update(d)
        self.meta['qat_state'] = label
        # We use a buffer to store which phase of QAT the current model is in. The phase is
        # set whenever the Sima QAT API is invoked incrementally.
        state_id = self._tag_to_id[label]
        device = _get_module_device(self)
        self.register_buffer(
            "qat_state",
            torch.tensor([state_id], dtype=torch.int8, device=device),
        )
        self.register_buffer(
            "shift_aware_qat",
            torch.tensor([shift_aware], dtype=torch.bool, device=device),
        )
        self.register_buffer(
            "qat_frozen",
            torch.tensor([label == 'fq'], dtype=torch.bool, device=device),
        )

    def train(self, use_train: bool = True) -> 'SimaQatWrapper':
        """This function emulates the behavior of train() on nn.Module.

        Args:
            use_train: set to `True` is training mode is desired, `False` if evaluation mode.

        Returns:
            SimaQatWrapper: a copy of the `self` variable. This return value provides 
                consistency with the behavior of `nn.Module.train()`.
        """
        if use_train:
            # Once we reach the fakequant state, we are always in inference mode.
            if self.meta['qat_state'] == 'fq':
                raise RuntimeError("Error: model is in fakequant mode; training mode is disallowed.")

        if use_train != self.training:
            mtext = {True: "train", False: "eval"}
            print(f"Switching mode to: {mtext[use_train]}")

        if use_train:
            move_exported_model_to_train(self)
            self.training = True
        else:
            move_exported_model_to_eval(self)
            self.training = False
        return self

    def eval(self, use_eval: bool = True) -> 'SimaQatWrapper':
        """This function sets the QAT module to evaluate mode. It is equivalent to 
        `nn.Module.eval()`.

        Args:
            use_eval: set to `True` if evaluation mode is desired, `False` if training mode.

        Returns:
            SimaQatWrapper: a copy of the `self` variable. This return value provides 
                consistency with the behavior of `nn.Module.eval()`.
        """
        return self.train(not use_eval)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        """This specialized load function adds an additional check for QAT models. This check ensures that
        a loaded state corresponds to the same phase of QAT training as the model skeleton in memory.

        Args:
            state_dict: a mapping of string keys to learned state (dense tensor data).
            strict: if `True`, all keys in the `state_dict` must exactly match the contents of this
                Module. If `False`, keys are allowed to mismatch state elements within this Module.
            assign: When ``False``, the properties of the tensors in the current module are preserved 
                while when ``True``, the properties of the Tensors in the state dict are preserved. The only
                exception is the ``requires_grad`` field of :class:`~torch.nn.Parameter`s for which the 
                value from the module is preserved.
        """
        if 'qat_state' not in state_dict:
            raise RuntimeError("Error: state_dict does not represent a QAT model")

        state_id = self._tag_to_id[self.meta['qat_state']]
        checkpoint_state_id = int(torch.as_tensor(state_dict['qat_state']).reshape(-1)[0].item())
        if checkpoint_state_id != state_id:
            raise RuntimeError(
                f"Error: model QAT state {state_id} doesn't match checkpoint QAT state {checkpoint_state_id}"
            )

        model_shift_aware = bool(self.shift_aware_qat.item())
        checkpoint_mode = state_dict.get('shift_aware_qat')
        if checkpoint_mode is None:
            if model_shift_aware:
                raise RuntimeError(
                    "This checkpoint predates shift-aware QAT. Prepare the model with "
                    "shift_aware=False before loading it."
                )
        else:
            checkpoint_shift_aware = bool(torch.as_tensor(checkpoint_mode).reshape(-1)[0].item())
            if checkpoint_shift_aware != model_shift_aware:
                raise RuntimeError(
                    "Checkpoint shift-aware mode does not match the prepared model: "
                    f"checkpoint={checkpoint_shift_aware}, model={model_shift_aware}"
                )

        compatible_state = OrderedDict(state_dict)
        if hasattr(state_dict, '_metadata'):
            compatible_state._metadata = state_dict._metadata
        compatible_state.setdefault('shift_aware_qat', self.shift_aware_qat.detach().clone())
        compatible_state.setdefault('qat_frozen', torch.zeros_like(self.qat_frozen))

        result = super().load_state_dict(compatible_state, strict, assign)
        if bool(self.qat_frozen.item()):
            for fake_quant in self.modules():
                if (
                    "FakeQuant" in type(fake_quant).__name__
                    and not getattr(fake_quant, "is_per_channel", False)
                    and hasattr(fake_quant, "learn_scale")
                ):
                    fake_quant.learn_scale = False
        return result


def check_graph_nodes(prepared_mod : GraphModule, device: torch.device) -> GraphModule:
    """ Checks the prepared model for inconsistent device paramterers and 
        also for setting the dropout layers to inactive mode
    """
    for n in prepared_mod.graph.nodes:
        #check for parameters not being in the same device as the model
        if n.target in device_modifier_ops:
            new_kwargs = dict(n.kwargs)
            new_kwargs['device'] = device
            n.kwargs = new_kwargs

    prepared_mod.recompile()
    return prepared_mod


def replace_dropout(m: GraphModule) -> GraphModule:
    def pattern(x, y, z):
        return torch.ops.aten.dropout.default(x, y, z)

    def replacement(x, y, z):
        return x

    torch.fx.replace_pattern(m, pattern, replacement)
    return m


def replace_batchnorm(m: GraphModule) -> GraphModule:
    """ FX Graph rewriter to replace a flavor of batchnorm with one that can be exported
    """
    def pattern(x, bn_weight, bn_bias, bn_running_mean, bn_running_var, momentum, eps):
        x = torch.ops.aten._native_batch_norm_legit_no_training.default(x, bn_weight, bn_bias, bn_running_mean, bn_running_var, momentum, eps)
        x = x[0]
        return x

    def replacement(x, bn_weight, bn_bias, bn_running_mean, bn_running_var, momentum, eps):
        return torch.nn.functional.batch_norm(x, bn_running_mean, bn_running_var, bn_weight, bn_bias, False, momentum, eps)

    torch.fx.replace_pattern(m, pattern, replacement)
    return m
