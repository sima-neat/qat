import copy
import math
import operator
import warnings
from collections import OrderedDict
from itertools import chain
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch

if tuple(int(part) for part in torch.__version__.split("+", 1)[0].split(".")[:2]) != (2, 8):
    raise RuntimeError(f"Sima QAT requires torch 2.8.x, found {torch.__version__}")

from torch import nn, Tensor
from torch.export import Dim, export_for_training
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
from torch.fx.node import Node
from torch.utils._pytree import tree_flatten, tree_map


from sima_qat import onnx_ops
from sima_qat.sima_quantizer import (
    SimaFakeQuantize,
    SimaQuantizer,
    get_sima_quantization_config,
)


device_modifier_ops = [
    torch.ops.aten.empty.memory_format,
    torch.ops.aten.arange.default,
    torch.ops.aten.full.default
]


def sima_prepare_qat_model(
    input_graph: nn.Module,
    inputs: Tuple,
    device: torch.device,
    *,
    dynamic_batch: bool = False,
) -> GraphModule:
    """This function is the first transformation needed to perform QAT on a Pytorch model. It takes an
    eager-mode reference to the ML model and produces an FX version of the graph with special annotations
    needed for QAT. Internally, it will scaffold the graph using observers and fakequant nodes needed 
    during the training process.

    Note:
        The Pytorch graph on which QAT is performed may be a full model, or a subsection of a model.
        QAT optimization will be limited to the graph given by the `input_graph` argument. This region
        must always be contained to the level of hierarchy as described by a single nn.Module.

        Capture preserves the exact example shapes by default. This is required
        for models that fold the batch dimension into recurrence, direction, or
        channel geometry. Set ``dynamic_batch=True`` only when the model is
        genuinely batch-polymorphic. Dynamic capture validates that the
        resulting graph still executes the caller's original example before
        QAT annotations are inserted.

    Args:
        input_graph: an eager-mode `nn.Module` representing the model on which QAT is to be performed.
            This may be a full model, or may be a sub-section of an ML model.
        inputs: a `Tuple` of tensor inputs, sized to the correct shape as the input to the given 
            `input_graph`. This data can be randomly generated. It is used during the preparation 
            process to build the compiled FX representation.
        device: a Pytorch `device` identifier. This will be the device on which the prepared model will
            be located after the preparation step is complete.
        dynamic_batch: explicitly opt into a symbolic leading batch dimension
            for tensor inputs sharing the first tensor's leading size. The
            default is deliberately static. Existing three-argument calls
            remain source-compatible and preserve the supplied example shape.
    Returns:
        GraphModule: a compiled version of the given graph with QAT annotations, ready to begin training.
    """
    if not isinstance(input_graph, nn.Module):
        raise RuntimeError(f"Input graph to prepare function must be of type nn.Module, found {type(input_graph)}")
    
    if isinstance(input_graph, GraphModule):
        return input_graph

    print(f"Making QAT annotations on model {input_graph._get_name()}...")
    # Capture from an isolated CPU copy. Preparation is a transformation, not
    # an in-place device/state mutation of the caller's eager model. This is
    # especially important for fail-closed dynamic capture: a rejected opt-in
    # must leave parameters, buffers, BatchNorm state, mode, and device intact.
    capture_graph = copy.deepcopy(input_graph).to("cpu")
    capture_example_inputs = _capture_inputs_to_cpu(inputs)
    if dynamic_batch:
        try:
            capture_inputs, dynamic_shapes = _dynamic_batch_capture(
                capture_example_inputs
            )
            m = export_for_training(
                capture_graph,
                capture_inputs,
                dynamic_shapes=dynamic_shapes,
            ).module()
            # Some networks fold batch into an internal recurrence or direction
            # dimension. A duplicated batch-one capture can export successfully
            # while hard-wiring the wrong internal geometry. Validate the graph on
            # the exact example the caller supplied and fail before scaffolding.
            validation_model = copy.deepcopy(m)
            with torch.no_grad():
                validation_model(*capture_example_inputs)
        except Exception as error:
            raise RuntimeError(
                "Dynamic-batch QAT capture does not execute the original "
                "example. Keep dynamic_batch=False for models whose batch "
                "dimension participates in folded recurrence or layout math."
            ) from error
    else:
        m = export_for_training(capture_graph, capture_example_inputs).module()
    m = replace_dropout(m)

    cfg = get_sima_quantization_config(is_qat=True)
    quantizer = SimaQuantizer().set_global(cfg)
    gm = prepare_qat_pt2e(m, quantizer)
    sima_mod = SimaQatWrapper(source=gm, label='scaffold')
    sima_mod.to(device)
    sima_mod.train()
    sima_mod = check_graph_nodes(sima_mod, device)

    return sima_mod


def _capture_inputs_to_cpu(inputs: Tuple) -> Tuple:
    """Return an isolated CPU example pytree without mutating caller values."""

    captured_tensors: Dict[int, Tensor] = {}

    def capture_leaf(value: Any) -> Any:
        if not isinstance(value, Tensor):
            return copy.deepcopy(value)
        # Preserve repeated-object aliasing in multi-input call signatures
        # while severing storage and autograd ties to the caller's tensor.
        key = id(value)
        if key not in captured_tensors:
            captured_tensors[key] = value.detach().to("cpu").clone()
        return captured_tensors[key]

    return tree_map(capture_leaf, inputs)


def _dynamic_batch_capture(inputs: Tuple) -> Tuple[Tuple, Optional[Any]]:
    """Build explicit dynamic-batch capture inputs and Torch shape hints.

    Torch specializes dimensions whose example value is zero or one. When the
    caller supplies batch one, capture uses an equivalent duplicated
    example because Torch specializes dimensions whose example value is one.
    The caller must explicitly request this behavior through
    :func:`sima_prepare_qat_model`; the exported graph is then validated on the
    original inputs before being returned.
    """
    leaves, _ = tree_flatten(inputs)
    tensor_inputs = [
        value for value in leaves if isinstance(value, Tensor) and value.ndim > 0
    ]
    if not tensor_inputs:
        return inputs, None

    batch_size = tensor_inputs[0].shape[0]
    if batch_size < 1:
        return inputs, None

    def is_batched(tensor: Tensor) -> bool:
        return tensor.ndim > 0 and tensor.shape[0] == batch_size

    capture_inputs = tree_map(
        lambda tensor: (
            torch.cat((tensor, tensor), dim=0)
            if isinstance(tensor, Tensor) and batch_size == 1 and is_batched(tensor)
            else tensor
        ),
        inputs,
    )
    dynamic_shapes = tree_map(
        lambda tensor: (
            {0: Dim.AUTO}
            if isinstance(tensor, Tensor) and is_batched(tensor)
            else None
        ),
        inputs,
    )
    return capture_inputs, dynamic_shapes


_SHIFT_AWARE_OPS = {
    torch.ops.aten.conv1d.default,
    torch.ops.aten.conv2d.default,
    torch.ops.aten.linear.default,
}
_MIN_REQUANT_SHIFT = 0
_MAX_REQUANT_SHIFT = 31


class _ShiftZeroOverflow(RuntimeError):
    """Weight range cannot fit the largest target-realizable requant grid."""


_STATIC_WEIGHT_FUNCTIONS = {
    operator.getitem,
    torch.ops.aten.add.Tensor,
    torch.ops.aten.cat.default,
    torch.ops.aten.chunk.default,
    torch.ops.aten.clone.default,
    torch.ops.aten.detach.default,
    torch.ops.aten.div.Tensor,
    torch.ops.aten.mul.Tensor,
    torch.ops.aten.neg.default,
    torch.ops.aten.permute.default,
    torch.ops.aten.reshape.default,
    torch.ops.aten.rsqrt.default,
    torch.ops.aten.select.int,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.sqrt.default,
    torch.ops.aten.squeeze.dim,
    torch.ops.aten.stack.default,
    torch.ops.aten.sub.Tensor,
    torch.ops.aten.t.default,
    torch.ops.aten.transpose.int,
    torch.ops.aten.unsqueeze.default,
    torch.ops.aten.view.default,
    torch.ops.aten._to_copy.default,
    torch.ops.aten._unsafe_view.default,
}


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


def _evaluate_static_weight_arg(
    module: GraphModule,
    value: Any,
    memo: Dict[Node, Any],
) -> Any:
    """Evaluate a parameter-only FX value without executing the model graph.

    Only the small set of pure tensor operations emitted by PT2E Conv-BN
    folding is accepted. Runtime inputs, modules, and unknown functions fail
    closed so dynamic weights cannot be mistaken for compile-time constants.
    """
    if isinstance(value, Node):
        if value in memo:
            return memo[value]
        if value.op == "placeholder":
            raise RuntimeError(
                f"static weight expression depends on runtime input {value.name!r}"
            )
        if value.op == "get_attr":
            result = _resolve_attr(module, value.target)
        elif value.op == "call_function":
            if value.target not in _STATIC_WEIGHT_FUNCTIONS:
                raise RuntimeError(
                    f"static weight expression contains unsupported operation {value.target}"
                )
            args = _evaluate_static_weight_arg(module, value.args, memo)
            kwargs = _evaluate_static_weight_arg(module, value.kwargs, memo)
            with torch.no_grad():
                result = value.target(*args, **kwargs)
        else:
            raise RuntimeError(
                f"static weight expression contains unsupported FX node {value.op!r}"
            )
        memo[value] = result
        return result
    if isinstance(value, tuple):
        return tuple(_evaluate_static_weight_arg(module, item, memo) for item in value)
    if isinstance(value, list):
        return [_evaluate_static_weight_arg(module, item, memo) for item in value]
    if isinstance(value, dict):
        return {
            key: _evaluate_static_weight_arg(module, item, memo)
            for key, item in value.items()
        }
    return value


def _resolve_static_weight_tensor(module: GraphModule, weight_source: Any) -> Tensor:
    weight = _evaluate_static_weight_arg(module, weight_source, {})
    if not isinstance(weight, Tensor):
        raise RuntimeError(
            f"static weight expression produced {type(weight).__name__}, expected Tensor"
        )
    return weight


def _minimum_weight_scale(
    module: GraphModule,
    weight_fq_node: Any,
) -> Tensor:
    """Return the smallest scale that should be used when locking a weight.

    The fake quantizer may consume either a direct Conv/Linear parameter or a
    parameter-only expression produced by PT2E Conv-BN folding. Evaluate that
    expression from current parameters and buffers so an optimizer step after
    the last observer update cannot introduce clipping.
    """
    weight_source = weight_fq_node.args[0] if getattr(weight_fq_node, "args", ()) else None
    weight = _resolve_static_weight_tensor(module, weight_source)
    if weight.ndim < 1:
        raise RuntimeError("Shift-aware weight tensors must have an output-channel dimension")
    reduce_dims = tuple(range(1, weight.ndim))
    max_abs = weight.detach().abs().amax(dim=reduce_dims)
    return torch.clamp(max_abs / 127.0, min=torch.finfo(torch.float32).tiny)


def _safe_power_of_two_weight_scale(
    input_scale: Tensor,
    output_scale: Tensor,
    minimum_weight_scale: Tensor,
) -> Tensor:
    """Return per-channel scales satisfying sx * sw / sy ~= 2**-shift.

    ``minimum_weight_scale`` is derived from the current parameter range for a
    direct weight, or from the learned fake-quant scale for an effective weight
    computed by a PT2E QAT pattern such as Conv-BatchNorm folding.

    Scales are rounded one float32 ULP toward zero when necessary. This keeps
    the normalized multiplier on the safe side of its power-of-two boundary,
    preventing a value infinitesimally above the boundary from selecting the
    next shift and a 0.5 correction factor.
    """
    sx = float(input_scale.reshape(-1)[0].detach().cpu())
    sy = float(output_scale.reshape(-1)[0].detach().cpu())
    if not math.isfinite(sx) or not math.isfinite(sy) or sx <= 0.0 or sy <= 0.0:
        raise RuntimeError(
            "Observed activation scales must be finite and positive, "
            f"found input={sx}, output={sy}"
        )

    required_scale = minimum_weight_scale.detach().reshape(-1).to(torch.float64).cpu()
    if (
        required_scale.numel() == 0
        or not bool(torch.isfinite(required_scale).all())
        or bool((required_scale <= 0).any())
    ):
        raise RuntimeError("Observed per-channel weight scales must be finite and positive")
    required_scale = torch.clamp(required_scale, min=torch.finfo(torch.float32).tiny)
    minimum_ratio = (sx / sy) * required_scale
    unclamped_shift = torch.floor(-torch.log2(minimum_ratio))
    shifts = unclamped_shift.clamp(_MIN_REQUANT_SHIFT, _MAX_REQUANT_SHIFT).to(torch.int32)

    sx_float32 = float(torch.tensor(sx, dtype=torch.float32))
    sy_float32 = float(torch.tensor(sy, dtype=torch.float32))
    for _ in range(_MAX_REQUANT_SHIFT + 2):
        target_ratio = torch.pow(
            torch.tensor(2.0, dtype=torch.float64),
            -shifts.to(torch.float64),
        )
        scales = ((sy / sx) * target_ratio).to(torch.float32)

        # Work with the exact float32 values persisted in the QDQ graph. Start one
        # ULP below the exact boundary so alternate float32 multiplication
        # order cannot move the imported ratio to the unsafe side.
        zero = torch.zeros_like(scales)
        scales = torch.nextafter(scales, zero)
        for _ in range(4):
            imported_ratio = sx_float32 * scales.to(torch.float64) / sy_float32
            too_high = imported_ratio > target_ratio
            if not bool(too_high.any()):
                break
            scales = torch.where(too_high, torch.nextafter(scales, zero), scales)

        too_small = scales.to(torch.float64) < required_scale
        if not bool(too_small.any()):
            break
        if bool(((shifts == _MIN_REQUANT_SHIFT) & too_small).any()):
            raise _ShiftZeroOverflow(
                "Required weight scale exceeds the largest shift-realizable grid at shift 0"
            )
        shifts = torch.where(too_small, shifts - 1, shifts)
    else:
        raise RuntimeError("Unable to find a non-clipping shift-aware weight scale")

    imported_ratio = sx_float32 * scales.to(torch.float64) / sy_float32
    if bool((imported_ratio > target_ratio).any()):
        raise RuntimeError("Unable to represent shift-aware weight scale safely in float32")

    return scales.to(minimum_weight_scale.device)


def _stage_activation_grid(
    fake_quant: FakeQuantizeBase,
    scale: Tensor,
    zero_point: Tensor,
    requested_scale: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Stage an observer-stable activation grid without mutating the model."""
    if scale.numel() != 1 or zero_point.numel() != 1 or requested_scale.numel() != 1:
        raise RuntimeError(
            "Shift-aware activation retargeting requires per-tensor qparams"
        )
    requested_scale = requested_scale.to(device=scale.device, dtype=scale.dtype)
    if (
        not bool(torch.isfinite(requested_scale).all())
        or bool((requested_scale <= 0).any())
    ):
        raise RuntimeError("Requested activation grid is non-finite or non-positive")
    observer = copy.deepcopy(fake_quant.activation_post_process)
    if not hasattr(observer, "min_val") or not hasattr(observer, "max_val"):
        raise RuntimeError(
            f"Activation observer {type(observer).__name__} cannot persist a retargeted grid"
        )

    # Keep the affine grid's integer origin fixed so real zero stays exact.
    requested_zero_point = zero_point.detach().clone()
    lower = (
        (observer.quant_min - requested_zero_point.to(requested_scale.dtype))
        * requested_scale
    ).to(device=observer.min_val.device, dtype=observer.min_val.dtype)
    upper = (
        (observer.quant_max - requested_zero_point.to(requested_scale.dtype))
        * requested_scale
    ).to(device=observer.max_val.device, dtype=observer.max_val.dtype)
    observer.min_val.resize_(lower.shape).copy_(lower)
    observer.max_val.resize_(upper.shape).copy_(upper)
    persisted_scale, persisted_zero_point = observer.calculate_qparams()
    persisted_scale = persisted_scale.to(device=scale.device, dtype=scale.dtype)
    persisted_zero_point = persisted_zero_point.to(
        device=zero_point.device, dtype=zero_point.dtype
    )
    if (
        not bool(torch.isfinite(persisted_scale).all())
        or bool((persisted_scale <= 0).any())
    ):
        raise RuntimeError("Retargeted activation grid is non-finite or non-positive")
    if not torch.equal(persisted_zero_point, requested_zero_point):
        raise RuntimeError(
            "Activation observer could not preserve the asymmetric zero point "
            "while persisting a retargeted grid"
        )
    return persisted_scale, persisted_zero_point, lower, upper


def _stage_coarser_activation_grid(
    fake_quant: FakeQuantizeBase,
    scale: Tensor,
    zero_point: Tensor,
    multiplier: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Stage a power-of-two coarser activation grid without mutation.

    Shift-aware QAT uses non-negative right shifts. When a weight needs a shift
    smaller than zero, coarsening the weighted operation's output grid by a
    power of two makes the contract realizable. Work on an observer copy so a
    later failure leaves the prepared model completely unchanged.
    """
    if multiplier < 2 or multiplier & (multiplier - 1):
        raise ValueError(
            "activation-grid multiplier must be a power of two >= 2, "
            f"found {multiplier}"
        )
    requested_scale = scale.detach() * multiplier
    return _stage_activation_grid(fake_quant, scale, zero_point, requested_scale)


def _shared_weight_output_scale(
    input_scale: Tensor,
    weight_scales: Tensor,
    minimum_output_scale: Tensor,
) -> Tensor:
    """Choose a non-clipping output grid for an already locked tied weight.

    Scales produced by :func:`_safe_power_of_two_weight_scale` differ across
    channels only by powers of two relative to its anchor invocation. A later
    invocation can therefore keep that exact weight tensor and select one
    scalar output scale that changes only the per-channel right shifts.
    """
    if input_scale.numel() != 1 or minimum_output_scale.numel() != 1:
        raise RuntimeError("Shared-weight retargeting requires per-tensor activations")
    sx = float(input_scale.reshape(-1)[0].detach().cpu())
    old_sy = float(minimum_output_scale.reshape(-1)[0].detach().cpu())
    products = sx * weight_scales.detach().reshape(-1).to(torch.float64).cpu()
    if not bool(torch.isfinite(products).all()) or bool((products <= 0).any()):
        raise RuntimeError("Shared-weight products must be finite and positive")

    reference = float(products.max())
    required_ratio = old_sy / reference
    shift = max(0, math.ceil(math.log2(required_ratio)))
    if shift > _MAX_REQUANT_SHIFT:
        raise RuntimeError(
            "A tied weight would require a requantization shift greater than "
            f"{_MAX_REQUANT_SHIFT}"
        )
    requested = torch.tensor(
        [reference * (2.0 ** shift)],
        device=minimum_output_scale.device,
        dtype=minimum_output_scale.dtype,
    )
    while float(requested.item()) < old_sy and shift < _MAX_REQUANT_SHIFT:
        shift += 1
        requested.mul_(2.0)
    if float(requested.item()) < old_sy:
        raise RuntimeError("Unable to retain the observed range for a tied weight")
    return requested


def _shift_ratios_are_realizable(
    input_scale: Tensor,
    weight_scales: Tensor,
    output_scale: Tensor,
) -> bool:
    ratios = (
        input_scale.detach().reshape(-1)[0].to(torch.float64).cpu()
        * weight_scales.detach().reshape(-1).to(torch.float64).cpu()
        / output_scale.detach().reshape(-1)[0].to(torch.float64).cpu()
    )
    if not bool(torch.isfinite(ratios).all()) or bool((ratios <= 0).any()):
        return False
    shifts = -torch.ceil(torch.log2(ratios))
    normalized = ratios * torch.pow(2.0, shifts)
    return bool(
        (shifts >= _MIN_REQUANT_SHIFT).all()
        and (shifts <= _MAX_REQUANT_SHIFT).all()
        and (normalized <= 1.0).all()
        and (normalized > 0.99999).all()
    )


def _freeze_batchnorm_stats(module: GraphModule) -> None:
    """Keep exported BatchNorm nodes in inference-statistics mode during recovery."""
    tracking_nodes = []
    for node in module.graph.nodes:
        if node.op == "call_function" and node.target == torch.ops.aten.batch_norm.default:
            if len(node.args) > 5 and node.args[5] is True:
                args = list(node.args)
                args[5] = False
                node.args = tuple(args)
        elif (
            node.op == "call_function"
            and node.target == torch.ops.aten.add_.Tensor
            and len(node.args) >= 2
            and getattr(node.args[0], "op", None) == "get_attr"
            and str(node.args[0].target).endswith("num_batches_tracked")
            and node.args[1] == 1
        ):
            tracking_nodes.append(node)

    for node in tracking_nodes:
        module.graph.erase_node(node)
    module.recompile()


def sima_freeze_qat(qat_model: GraphModule) -> GraphModule:
    """Freeze QAT observers and lock SiMa-compatible power-of-two weight scales.

    Call this after observer warm-up, then continue fine-tuning with fake quantization
    enabled.
    """
    if not isinstance(qat_model, GraphModule):
        raise RuntimeError(f"Input graph to freeze function must be a GraphModule, found {type(qat_model)}")
    if bool(getattr(qat_model, "qat_frozen", torch.tensor([0])).item()):
        return qat_model

    # Resolve the exact observer/export activation grids first. Keep these in
    # staged rows until every graph constraint validates so freeze remains
    # atomic even when output-grid coarsening is required.
    activation_qparams = []
    for fake_quant in qat_model.modules():
        if (
            not isinstance(fake_quant, FakeQuantizeBase)
            or getattr(fake_quant, "is_per_channel", False)
        ):
            continue
        try:
            scale, zero_point = fake_quant.activation_post_process.calculate_qparams()
        except (AssertionError, RuntimeError, ValueError) as error:
            raise RuntimeError(
                "Shift-aware QAT could not calculate valid activation qparams "
                f"from {type(fake_quant.activation_post_process).__name__}: {error}"
            ) from error
        activation_qparams.append([
            fake_quant,
            scale.to(device=fake_quant.scale.device, dtype=fake_quant.scale.dtype),
            zero_point.to(
                device=fake_quant.zero_point.device,
                dtype=fake_quant.zero_point.dtype,
            ),
            None,
            None,
        ])
    activation_qparams_by_id = {id(row[0]): row for row in activation_qparams}

    contracts = []
    skipped = []
    for node in qat_model.graph.nodes:
        if node.op != "call_function" or node.target not in _SHIFT_AWARE_OPS:
            continue
        if len(node.args) < 2:
            skipped.append(node.name)
            continue

        input_fq = _fake_quant_module(qat_model, node.args[0])
        weight_fq = _fake_quant_module(qat_model, node.args[1])
        output_fq = _find_output_fake_quant(qat_model, node)
        if (
            input_fq is None
            or weight_fq is None
            or output_fq is None
            or weight_fq.qscheme not in (torch.per_channel_affine, torch.per_channel_symmetric)
        ):
            skipped.append(node.name)
            continue

        try:
            minimum_weight_scale = _minimum_weight_scale(qat_model, node.args[1])
        except RuntimeError as error:
            skipped.append(f"{node.name} ({error})")
            continue

        contracts.append(
            (node, input_fq, weight_fq, output_fq, minimum_weight_scale)
        )

    if skipped:
        raise RuntimeError(
            "Shift-aware QAT could not determine complete input/weight/output quantization "
            f"parameters for {len(skipped)} op(s): {', '.join(skipped)}"
        )

    # Close shift-realizability over the graph. A coarsened output may be a
    # downstream input or a shared grid, so repeat until no contract changes.
    activation_retargets = []
    for _ in range(len(contracts) + 1):
        changed = False
        for node, input_fq, _, output_fq, minimum_weight_scale in contracts:
            input_row = activation_qparams_by_id[id(input_fq)]
            output_row = activation_qparams_by_id[id(output_fq)]
            try:
                _safe_power_of_two_weight_scale(
                    input_row[1],
                    output_row[1],
                    minimum_weight_scale,
                )
                continue
            except _ShiftZeroOverflow:
                # Output-grid coarsening is the only legal recovery from a
                # typed shift-0 overflow.
                pass
            except RuntimeError as error:
                # Structural and non-finite failures are not feasibility
                # results. Fail atomically rather than inferring control flow
                # from their diagnostic strings.
                raise RuntimeError(
                    f"Shift-aware QAT could not inspect weight grid for {node.name}: {error}"
                ) from error

            sx = float(input_row[1].reshape(-1)[0].detach().cpu())
            sy = float(output_row[1].reshape(-1)[0].detach().cpu())
            required_scale = minimum_weight_scale.detach().reshape(-1).to(
                torch.float64
            ).cpu()
            maximum_ratio = float(((sx / sy) * required_scale).max())
            exponent = max(1, math.ceil(math.log2(maximum_ratio)))
            multiplier = 1 << exponent
            old_scale = float(output_row[1].reshape(-1)[0].detach().cpu())
            old_zero_point = int(output_row[2].reshape(-1)[0].detach().cpu())
            scale, zero_point, lower, upper = _stage_coarser_activation_grid(
                output_fq,
                output_row[1],
                output_row[2],
                multiplier,
            )
            new_scale = float(scale.reshape(-1)[0].detach().cpu())
            if new_scale <= old_scale:
                raise RuntimeError(
                    f"Activation observer for {node.name} could not persist a coarser grid"
                )
            output_row[1] = scale
            output_row[2] = zero_point
            output_row[3] = lower
            output_row[4] = upper
            activation_retargets.append({
                "op": node.name,
                "scale_before": old_scale,
                "scale_after": new_scale,
                "zero_point_before": old_zero_point,
                "zero_point_after": int(
                    zero_point.reshape(-1)[0].detach().cpu()
                ),
                "power_of_two_multiplier": multiplier,
            })
            changed = True
        if not changed:
            break
    else:
        raise RuntimeError(
            "Shift-aware activation-grid constraint solver did not converge"
        )

    # Lock tied weights once, in graph order. Later invocations retain that
    # exact weight grid and coarsen their output activation grid just enough
    # to select a legal right shift. Computing and writing one scale per call
    # would make the last call silently invalidate every earlier call that
    # shares the fake-quantized weight.
    locked_scales_by_weight = {}
    for node, input_fq, weight_fq, output_fq, minimum_weight_scale in contracts:
        input_row = activation_qparams_by_id[id(input_fq)]
        output_row = activation_qparams_by_id[id(output_fq)]
        weight_id = id(weight_fq)
        if weight_id not in locked_scales_by_weight:
            try:
                scales = _safe_power_of_two_weight_scale(
                    input_row[1],
                    output_row[1],
                    minimum_weight_scale,
                )
            except RuntimeError as error:
                raise RuntimeError(
                    f"Shift-aware QAT could not solve weight grid for {node.name}: {error}"
                ) from error
            locked_scales_by_weight[weight_id] = (weight_fq, scales)
            continue

        _, scales = locked_scales_by_weight[weight_id]
        if _shift_ratios_are_realizable(input_row[1], scales, output_row[1]):
            continue
        old_scale = float(output_row[1].reshape(-1)[0].detach().cpu())
        old_zero_point = int(output_row[2].reshape(-1)[0].detach().cpu())
        try:
            requested_scale = _shared_weight_output_scale(
                input_row[1], scales, output_row[1]
            )
            scale, zero_point, lower, upper = _stage_activation_grid(
                output_fq,
                output_row[1],
                output_row[2],
                requested_scale,
            )
        except RuntimeError as error:
            raise RuntimeError(
                f"Shift-aware QAT could not align tied weight grid for {node.name}: {error}"
            ) from error
        if not _shift_ratios_are_realizable(input_row[1], scales, scale):
            raise RuntimeError(
                f"Shift-aware QAT could not represent tied weight grid for {node.name}"
            )
        output_row[1] = scale
        output_row[2] = zero_point
        output_row[3] = lower
        output_row[4] = upper
        new_scale = float(scale.reshape(-1)[0].detach().cpu())
        activation_retargets.append({
            "op": node.name,
            "scale_before": old_scale,
            "scale_after": new_scale,
            "zero_point_before": old_zero_point,
            "zero_point_after": int(zero_point.reshape(-1)[0].detach().cpu()),
            "shared_weight_alignment": True,
        })

    # A retargeted shared activation can participate in more than one graph
    # contract. Validate the complete fixed point before mutating any module.
    for node, input_fq, weight_fq, output_fq, _ in contracts:
        _, scales = locked_scales_by_weight[id(weight_fq)]
        if not _shift_ratios_are_realizable(
            activation_qparams_by_id[id(input_fq)][1],
            scales,
            activation_qparams_by_id[id(output_fq)][1],
        ):
            raise RuntimeError(
                "Shift-aware QAT tied-weight alignment invalidated another "
                f"contract at {node.name}"
            )

    locked_scales = list(locked_scales_by_weight.values())

    fake_quantizers = [
        module for module in qat_model.modules() if isinstance(module, FakeQuantizeBase)
    ]
    unsupported = [
        type(module).__name__
        for module in fake_quantizers
        if not isinstance(module, SimaFakeQuantize)
    ]
    if unsupported:
        raise RuntimeError(
            "Sima QAT cannot freeze qparams for unsupported fake quantizer(s): "
            + ", ".join(sorted(set(unsupported)))
        )

    # Do not mutate observer or fake-quantizer state until every activation and
    # weight contract has validated.
    qat_model.apply(disable_observer)
    _freeze_batchnorm_stats(qat_model)
    for fake_quant, scale, zero_point, lower, upper in activation_qparams:
        if lower is not None:
            observer = fake_quant.activation_post_process
            observer.min_val.resize_(lower.shape).copy_(lower)
            observer.max_val.resize_(upper.shape).copy_(upper)
        fake_quant.scale.resize_(scale.shape).copy_(scale)
        fake_quant.zero_point.resize_(zero_point.shape).copy_(zero_point)
    for weight_fq, scales in locked_scales:
        weight_fq.scale.resize_(scales.shape).copy_(scales)
        weight_fq.zero_point.resize_(scales.shape).zero_()

    for fake_quantizer in fake_quantizers:
        fake_quantizer.freeze_qparams()

    qat_model.meta["qat_activation_retargets"] = activation_retargets
    qat_model.qat_frozen.fill_(1)
    return qat_model


def _remove_batchnorm_tracking_updates(gm: GraphModule) -> None:
    """Remove dead BatchNorm batch-counter updates before PT2E conversion.

    Frozen BatchNorm statistics no longer consume ``num_batches_tracked``.
    Removing its dead in-place increment here also prevents PT2E's Conv-BN
    folding passes from attempting to erase the same node more than once.
    """
    if not hasattr(gm, "graph"):
        return
    for node in list(gm.graph.nodes):
        if (
            node.op == "call_function"
            and node.target == torch.ops.aten.add_.Tensor
            and len(node.args) >= 2
            and getattr(node.args[0], "op", None) == "get_attr"
            and str(getattr(node.args[0], "target", "")).endswith(
                "num_batches_tracked"
            )
            and node.args[1] == 1
            and not node.users
        ):
            gm.graph.erase_node(node)
    gm.graph.eliminate_dead_code()
    gm.recompile()


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
    if qat_model.meta.get("qat_state") == "fq":
        return qat_model
    if not bool(getattr(qat_model, "qat_frozen", torch.tensor([0])).item()):
        warnings.warn(
            "Finalizing a shift-aware model before sima_freeze_qat(); scales will be locked now. "
            "For best accuracy, freeze earlier and fine-tune with the locked scales.",
            stacklevel=2,
        )
        sima_freeze_qat(qat_model)
    print(f"Removing QAT scaffold and quantizing network ...")
    _remove_batchnorm_tracking_updates(qat_model)
    with warnings.catch_warnings():
        # Torch 2.8 can report a second erase for an overlapping Conv-BN
        # pattern after the node was already removed successfully. Output and
        # graph validation below cover the resulting folded graph.
        warnings.filterwarnings(
            "ignore",
            message=r"erase_node\(batch_norm_\d+\) on an already erased node",
            category=UserWarning,
        )
        m = convert_pt2e(qat_model, use_reference_representation=False)
    sima_mod = SimaQatWrapper(source=m, label='fq')
    # We must call eval() to invoke internal functions to put the GraphModule in eval state. Once we are
    # in FQ mode, we always remain in eval mode.
    sima_mod.eval()
    sima_mod = replace_batchnorm(sima_mod)
    return sima_mod


def sima_export_onnx(qat_model: nn.Module, inputs: Tuple[Tensor], output_file: str, input_names: Optional[List[str]] = None, 
                     output_names: Optional[List[str]] = None, device: torch.device = 'cuda') -> GraphModule:
    """This function exports a finalized QAT model to ONNX format.

    Args:
        qat_model: The finalized ML model to export to ONNX.
        inputs: a `Tuple` of tensor inputs used to infer the proper shapes of all internal tensors.
            This is used by the Pytorch ONNX exporter.
        output_file: the path name of the .onnx file to generate.
        input_names: a list of tensor names used to label the ONNX model inputs.
        output_names: a list of tensor names used to label the ONNX model outputs.
    """
    if not isinstance(qat_model, nn.Module):
        raise RuntimeError(f"Input graph to export function must be of type nn.Module, found {type(qat_model)}")
    qat_model = check_graph_nodes(qat_model, device='cpu')
    with warnings.catch_warnings():
        # ONNX InstanceNormalization always uses input statistics, matching
        # PyTorch InstanceNorm with track_running_stats=False. The legacy
        # exporter labels that valid behavior as train=True and warns solely
        # because the surrounding export is in evaluation mode.
        warnings.filterwarnings(
            "ignore",
            message=(
                r"ONNX export mode is set to TrainingMode\.EVAL, but operator "
                r"'instance_norm' is set to train=True\..*"
            ),
            category=UserWarning,
        )
        torch.onnx.export(
            qat_model,
            inputs,
            output_file,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
        )
    qat_model = check_graph_nodes(qat_model, device=device)
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

    def __init__(self, source: GraphModule, label: str):
        """This constructor creates a wrapper from a GraphModule. We can only create this object
        from an existing GraphModule class. Every time we create a wrapper, we also need to
        specify which phase of QAT we are representing, since each phase has different 
        restrictions as to what is permissible.

        Args:
            source: A `GraphModule` produced by Pytorch call to some PT2E initialization. Must be
                a compiled FX graph.
            label: One of the legal enumerated labels matching the phase of the QAT process.
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
        # GraphModule.__dict__ was adopted above, so new wrapper-owned state must
        # follow the graph's device.  Registering these markers on the CPU makes
        # a CUDA graph mixed-device and causes PT2E's BatchNorm train/eval
        # rewriting to reject it during finalization.
        graph_tensors = chain(source.parameters(), source.buffers())
        first_graph_tensor = next(graph_tensors, None)
        state_device = (
            first_graph_tensor.device
            if first_graph_tensor is not None
            else torch.device("cpu")
        )
        self.register_buffer(
            "qat_state",
            torch.tensor([state_id], dtype=torch.int8, device=state_device),
        )
        self.register_buffer(
            "qat_frozen",
            torch.tensor([label == 'fq'], dtype=torch.bool, device=state_device),
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
            if bool(getattr(self, "qat_frozen", torch.tensor([0])).item()):
                _freeze_batchnorm_stats(self)
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

        compatible_state = OrderedDict(state_dict)
        if hasattr(state_dict, '_metadata'):
            compatible_state._metadata = state_dict._metadata
        # The first shift-aware release stored an always-true mode marker. The
        # mode is now unconditional, so discard that redundant checkpoint key.
        mode_marker = compatible_state.pop('shift_aware_qat', None)
        if mode_marker is not None and not bool(torch.as_tensor(mode_marker).reshape(-1)[0].item()):
            raise RuntimeError("Only shift-aware QAT checkpoints are supported")
        compatible_state.setdefault('qat_frozen', torch.zeros_like(self.qat_frozen))

        return super().load_state_dict(compatible_state, strict, assign)


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
