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
from __future__ import annotations

import copy
import operator
import functools
import itertools
import os
import math

from typing import Any, Callable, Dict, List, Optional, Set

import torch
import torch._dynamo as torchdynamo
import torch.nn.functional as F
from torch.ao.quantization.fake_quantize import (
    FakeQuantize,
    FusedMovingAvgObsFakeQuantize,
)
from torch.ao.quantization.observer import (
    HistogramObserver,
    MinMaxObserver,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
    PerChannelMinMaxObserver,
    PlaceholderObserver,
)

from torch.ao.quantization.qconfig import _ObserverOrFakeQuantizeConstructor

from torch.ao.quantization.quantizer import (
    FixedQParamsQuantizationSpec,
    QuantizationSpec, 
    Quantizer,
    QuantizationAnnotation,
    SharedQuantizationSpec,
)

from torch.ao.quantization.quantizer.xnnpack_quantizer_utils import (
    _convert_scalars_to_attrs,
    OP_TO_ANNOTATOR,
    OperatorConfig,
    OperatorPatternType,
    propagate_annotation,
    QuantizationConfig,
    _is_annotated,
    get_input_act_qspec,
    get_output_act_qspec,
    register_annotator,
    _is_input_non_float_tensor,
    _is_input_large_scalar,
    get_weight_qspec,
    get_bias_qspec,
    _mark_nodes_as_annotated,
    _WrapperModule
)
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    _get_module_type_filter,
    _get_dynamo_graph,
    _get_linear_patterns,
    _get_module_name_filter,
    _get_module_type_filter,
    _get_not_module_type_or_name_filter,
)
try:
    from torch.ao.quantization.pt2e.utils import (
        _conv1d_bn_example_inputs,
        _conv2d_bn_example_inputs,
        get_aten_graph_module,
    )
except ImportError:
    # torch 2.8 renamed the pattern exporter and stopped publishing the two
    # small example tuples.  Their concrete values are irrelevant; only the
    # ranks/shapes are used to capture Conv+BN patterns.
    from torch.ao.quantization.pt2e.utils import (
        _get_aten_graph_module_for_pattern as get_aten_graph_module,
    )

    _conv1d_bn_example_inputs = (
        torch.ones(1, 1, 3),
        torch.ones(1, 1, 1),
        torch.ones(1),
        torch.ones(1),
        torch.ones(1),
        torch.ones(1),
        torch.ones(1),
    )
    _conv2d_bn_example_inputs = (
        torch.ones(1, 1, 3, 3),
        torch.ones(1, 1, 1, 1),
        torch.ones(1),
        torch.ones(1),
        torch.ones(1),
        torch.ones(1),
        torch.ones(1),
    )
from torch.fx.passes.utils.matcher_with_name_node_map_utils import (
    SubgraphMatcherWithNameNodeMap,
)

from torch.fx import Node
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions
from torch.ao.quantization.pt2e.graph_utils import find_sequential_partitions


# from torch.ops.quantized_decomposed import quantize_per_tensor


__all__ = [
    "SimaQuantizer",
    "get_sima_quantization_config",
]


class _LearnedScaleSTE(torch.autograd.Function):
    """Memory-bounded LSQ surrogate for large exported activation graphs.

    Building ``scale_error`` with ordinary autograd keeps another full-sized
    tensor alive at every fake-quant boundary.  Attention models have thousands
    of those boundaries and can OOM before the first backward.  Recompute the
    inexpensive code error in backward while retaining only references to the
    input and scalar qparams.
    """

    @staticmethod
    def forward(
        ctx,
        value: Tensor,
        scale: Tensor,
        zero_point: Tensor,
        quant_min: int,
        quant_max: int,
        quant_strength: float,
        grad_factor: float,
    ) -> Tensor:
        detached_scale = scale.detach()
        detached_zero_point = zero_point.detach().to(value.dtype)
        normalized = value.detach() / detached_scale + detached_zero_point
        code = torch.round(normalized).clamp(quant_min, quant_max)
        # Use the same native kernel as the fixed-scale strict-INT8 forward.
        # Reconstructing Q/DQ as round/divide/multiply is mathematically
        # equivalent over reals but not bit-equivalent in float32; one-ULP
        # differences are observable in long recurrent graphs.
        dequantized = torch.fake_quantize_per_tensor_affine(
            value,
            detached_scale,
            zero_point.detach().to(torch.int32),
            quant_min,
            quant_max,
        )
        # The LSQ error is the only activation-dependent value needed for the
        # scalar scale gradient.  Saving it in FP16 is substantially smaller
        # than retaining the FP32 activation (or both activation and code) at
        # every boundary; the reduction is accumulated in FP32 in backward.
        # LSQ's scale surrogate is piecewise. Inside the representable range,
        # dQ/ds = round(x/s + z) - (x/s + z). At either saturation rail it is
        # the *rail code relative to zero*, not that expression continued past
        # the rail. Continuing ``q - x/s`` outside the range makes a single
        # outlier contribute an unbounded, wrong-sign scale gradient.
        scale_error = torch.where(
            normalized < quant_min,
            torch.as_tensor(
                quant_min, device=value.device, dtype=value.dtype
            ) - detached_zero_point,
            torch.where(
                normalized > quant_max,
                torch.as_tensor(
                    quant_max, device=value.device, dtype=value.dtype
                ) - detached_zero_point,
                code - normalized,
            ),
        ).to(torch.float16)
        ctx.save_for_backward(scale_error)
        ctx.grad_factor = grad_factor
        ctx.quant_strength = float(quant_strength)
        return value + float(quant_strength) * (dequantized - value)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (scale_error,) = ctx.saved_tensors
        grad_scale = (
            grad_output.float()
            * scale_error.float()
            * ctx.grad_factor
            * ctx.quant_strength
        ).sum().reshape(1)
        return grad_output, grad_scale, None, None, None, None, None


class SimaMovingAverageMinMaxObserver(MovingAverageMinMaxObserver):
    r"""
        We override the vanilla Pytorch MinMaxObserver with a specialized version. This
        specialized version averages min/max over each sample in the batch. This gives
        more consistency across samples, and makes training behavior more independent
        from the batch size setting.
    """
    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()  # avoid keeping autograd tape
        x = x.to(self.min_val.dtype)
        min_val = self.min_val
        max_val = self.max_val
        if min_val == float("inf") and max_val == float("-inf"):
            min_val, max_val = torch.aminmax(x)
        else:
            if len(x.shape) < 2:
                min_val_cur, max_val_cur = torch.aminmax(x)
            else:
                # min_val_cur, max_val_cur = torch.aminmax(x)
                if x.is_contiguous():
                    x_flat = x.view(x.shape[0], -1)
                else:
                    x_flat = x.reshape(x.shape[0], -1)
                min_val_batch, max_val_batch = torch.aminmax(x_flat, dim=1)
                min_val_cur = min_val_batch.mean()
                max_val_cur = max_val_batch.mean()

            min_val = min_val + self.averaging_constant * (min_val_cur - min_val)
            max_val = max_val + self.averaging_constant * (max_val_cur - max_val)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig


class FullRangeSTEFakeQuantize(FakeQuantize):
    """Fake INT8 forward with an identity backward, including clipped values.

    PyTorch's native fake-quant backward zeroes gradients outside the observed
    range. DepthART's recurrent state can temporarily exceed that range early
    in QAT, which otherwise blocks the very gradient needed to pull it back.
    The forward remains bit-identical to native fake quantization; only the
    training surrogate gradient changes.
    """

    def __init__(self, *args, learn_scale: Optional[bool] = None, **kwargs):
        super().__init__(*args, **kwargs)
        # QAT on a deeply recurrent graph can fail if every activation is
        # switched from float to INT8 in one step.  Keep the deploy default at
        # one, while allowing the training driver to introduce the exact INT8
        # forward progressively.  This is deliberately non-persistent: a
        # reloaded/exported checkpoint always executes the strict INT8
        # simulation unless its training session explicitly changes it.
        self.quant_strength = 1.0
        # QDrop-style activation bypass is a training-only regularizer.  A
        # probability of zero is the strict deployment behavior.  Keep this
        # as ordinary (non-persistent) Python state so loading or exporting a
        # checkpoint can never accidentally retain a relaxed forward.
        self.quantization_dropout_probability = 0.0
        # Integer MLA kernels can differ from the float-QDQ reference by one
        # output code because accumulation and requantization happen in the
        # integer domain. QAT may explicitly inject that measured error as a
        # robustness augmentation. Reloaded/exported checkpoints default to
        # zero and remain deterministic.
        self.target_code_noise_probability = 0.0
        self.learn_scale = (
            os.environ.get("SIMA_QAT_LEARN_SCALES", "0") == "1"
            if learn_scale is None
            else bool(learn_scale)
        )
        # Optional training-only reparameterization used when reopening a
        # qualified frozen grid.  Keeping the exact float32 anchor avoids the
        # one-ULP ``exp(log(scale))`` round trip at delta zero; recurrent INT8
        # graphs can amplify that otherwise harmless-looking perturbation.
        # This is intentionally non-persistent: callers must synchronize or
        # freeze the selected scale before serializing a deploy checkpoint.
        self._relative_scale_anchor = None
        if self.learn_scale:
            self.log_scale = torch.nn.Parameter(self.scale.detach().clamp_min(1e-12).log())
            # Training-only anchor for an exactly projected float32 scale.
            # ``exp(log(scale))`` is not generally bit-identical to ``scale``;
            # retaining the anchor avoids a one-ULP grid change between target
            # projection and final export while preserving LSQ gradients.
            self._learned_scale_anchor = None
            self._learned_log_scale_anchor = None

    @torch.no_grad()
    def initialize_learned_scale(self):
        if self.learn_scale:
            self._relative_scale_anchor = None
            self.log_scale.copy_(self.scale.detach().clamp_min(1e-12).log())
            self._learned_scale_anchor = self.scale.detach().clone()
            self._learned_log_scale_anchor = self.log_scale.detach().clone()

    def current_learned_scale(self):
        if self._learned_scale_anchor is None:
            return self.log_scale.exp().clamp_min(1e-12)
        anchor_scale = self._learned_scale_anchor.to(
            device=self.log_scale.device, dtype=self.log_scale.dtype
        )
        anchor_log = self._learned_log_scale_anchor.to(
            device=self.log_scale.device, dtype=self.log_scale.dtype
        )
        return (anchor_scale * (self.log_scale - anchor_log).exp()).clamp_min(1e-12)

    @torch.no_grad()
    def set_projected_learned_scale(self, scale):
        scale = scale.detach().to(
            device=self.log_scale.device, dtype=self.log_scale.dtype
        ).clamp_min(1e-12)
        self.log_scale.copy_(scale.log())
        self._learned_scale_anchor = scale.clone()
        self._learned_log_scale_anchor = self.log_scale.detach().clone()

    @torch.no_grad()
    def enable_relative_scale_learning(self):
        """Learn a log multiplier around the exact current activation scale.

        At initialization the effective scale is ``anchor * exp(0)`` and is
        therefore bit-identical to ``anchor`` in float32.  This is preferable
        to the absolute ``exp(log(anchor))`` parameterization when resuming a
        frozen, accuracy-qualified recurrent graph.  The anchor is
        training-only; call :meth:`sync_learned_scale` and freeze before
        saving a deployment checkpoint.
        """

        if self.is_per_channel:
            raise RuntimeError("relative scale learning supports activations only")
        if not hasattr(self, "log_scale"):
            raise RuntimeError("relative scale learning requires learn_scale=True")
        self._relative_scale_anchor = self.scale.detach().clone()
        self.log_scale.zero_()
        self.learn_scale = True

    def current_learned_scale(self):
        """Return the differentiable scale used by the learned-scale forward."""

        if self._relative_scale_anchor is not None:
            anchor = self._relative_scale_anchor.to(
                device=self.log_scale.device, dtype=self.log_scale.dtype
            )
            return anchor * self.log_scale.exp()
        return self.log_scale.exp()

    @torch.no_grad()
    def sync_learned_scale(self):
        if self.learn_scale:
            self.scale.copy_(self.current_learned_scale().detach())

    def set_quant_strength(self, value: float) -> None:
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"quantization strength must be in [0, 1], found {value}")
        self.quant_strength = value

    def set_quantization_dropout_probability(self, value: float) -> None:
        """Set the probability that an activation element bypasses Q/DQ.

        Dropout is active only in training mode and only for per-tensor
        activations.  The strict inference graph is unchanged.  In training,
        for a Bernoulli keep mask ``m`` the forward is

        ``x + m * (Q(x) - x)``.

        Its input surrogate gradient is exactly one, while stochastic float
        bypasses prevent every layer from fitting the same deterministic
        activation-rounding error on every update.
        """

        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                "quantization-dropout probability must be in [0, 1], "
                f"found {value}"
            )
        self.quantization_dropout_probability = value

    def _apply_quantization_dropout(
        self,
        value: torch.Tensor,
        quantized: torch.Tensor,
        *,
        detach_residual: bool,
    ) -> torch.Tensor:
        probability = self.quantization_dropout_probability
        if (
            not self.training
            or probability <= 0.0
            or self.is_per_channel
        ):
            return quantized
        if probability >= 1.0:
            return value
        keep = torch.rand_like(value) >= probability
        residual = quantized - value
        if detach_residual:
            residual = residual.detach()
        return value + keep.to(value.dtype) * residual

    def set_target_code_noise_probability(self, value: float) -> None:
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                "target code-noise probability must be in [0, 1], "
                f"found {value}"
            )
        self.target_code_noise_probability = value

    def _apply_target_code_noise(self, quantized: torch.Tensor) -> torch.Tensor:
        """Randomly perturb activation codes by +/-1 without changing the STE.

        This training-only augmentation preserves a valid signed-INT8 code at
        every forward boundary and never applies to per-channel weights.
        """
        probability = self.target_code_noise_probability
        if probability <= 0.0 or self.is_per_channel:
            return quantized
        scale = self.scale.to(device=quantized.device, dtype=quantized.dtype)
        zero_point = self.zero_point.to(device=quantized.device, dtype=quantized.dtype)
        code = torch.round(quantized.detach() / scale + zero_point)
        draw = torch.rand_like(quantized)
        delta = torch.where(
            draw < probability * 0.5,
            -torch.ones_like(draw),
            torch.where(draw < probability, torch.ones_like(draw), torch.zeros_like(draw)),
        )
        noisy_code = (code + delta).clamp(
            self.activation_post_process.quant_min,
            self.activation_post_process.quant_max,
        )
        noisy = (noisy_code - zero_point) * scale
        return quantized + (noisy - quantized).detach()

    def forward(self, x):
        # Structural masks and indices may inherit an annotation through PT2E
        # propagation.  They are not activation tensors and cannot legally be
        # fake-quantized (subtraction/rounding are undefined for bool).  Keep
        # them byte-for-byte unchanged.
        if not x.is_floating_point():
            return x
        if self.observer_enabled[0] == 1:
            self.activation_post_process(x.detach())
            # Observer-only calibration intentionally bypasses fake quant and
            # defers the potentially expensive qparam search until all samples
            # have been seen. During ordinary QAT, fake quant is enabled and
            # qparams continue to track the observer as usual.
            if self.fake_quant_enabled[0] == 1:
                scale, zero_point = self.calculate_qparams()
                scale = scale.to(self.scale.device)
                zero_point = zero_point.to(self.zero_point.device)
                if self.scale.shape != scale.shape:
                    self.scale.resize_(scale.shape)
                    self.zero_point.resize_(zero_point.shape)
                self.scale.copy_(scale)
                self.zero_point.copy_(zero_point)
                self.initialize_learned_scale()
        if self.fake_quant_enabled[0] != 1:
            return x
        if self.learn_scale and not self.is_per_channel:
            scale = self.current_learned_scale()
            with torch.no_grad():
                self.scale.copy_(scale.detach())
            if self.log_scale.requires_grad:
                zero_point = self.zero_point.to(x.dtype)
                grad_factor = 1.0 / math.sqrt(
                    max(1, x.numel()) * max(1, self.quant_max)
                )
                quantized = _LearnedScaleSTE.apply(
                    x,
                    scale,
                    zero_point,
                    self.activation_post_process.quant_min,
                    self.activation_post_process.quant_max,
                    self.quant_strength,
                    grad_factor,
                )
                quantized = self._apply_target_code_noise(quantized)
                # _LearnedScaleSTE already supplies the identity input
                # gradient. Do not detach here: the mask must also gate the
                # learned-scale gradient for bypassed elements.
                return self._apply_quantization_dropout(
                    x, quantized, detach_residual=False
                )
            # A frozen log_scale does not need LSQ's elementwise saved tensor.
            # Fall through to the ordinary full-range STE using the exact same
            # live scale. This cuts activation memory from O(all QAT boundary
            # elements) to O(only boundaries whose scale is trainable).
        if self.is_per_channel:
            quantized = torch.fake_quantize_per_channel_affine(
                x, self.scale, self.zero_point, self.ch_axis,
                self.activation_post_process.quant_min,
                self.activation_post_process.quant_max,
            )
        else:
            quantized = torch.fake_quantize_per_tensor_affine(
                x, self.scale, self.zero_point,
                self.activation_post_process.quant_min,
                self.activation_post_process.quant_max,
            )
        quantized = self._apply_target_code_noise(quantized)
        quantized = self._apply_quantization_dropout(
            x, quantized, detach_residual=True
        )
        return x + self.quant_strength * (quantized - x).detach()


def _supported_symmetric_quantized_operators() -> Dict[str, List[OperatorPatternType]]:
    supported_operators: Dict[str, List[OperatorPatternType]] = {
        # Both conv and linear should be able to handle relu + hardtanh fusion since
        # those are clamp ops
        "conv2d": [
            [torch.nn.Conv2d, torch.nn.ReLU],
            [torch.nn.Conv2d, F.relu],
            [F.conv2d, torch.nn.ReLU],
            [F.conv2d, F.relu],
        ],
        "linear": [[torch.nn.Linear], [F.linear]],
        "matmul": [[torch.matmul], [operator.matmul]],
        "softmax": [[torch.nn.Softmax], [F.softmax]],
        "add": [[torch.add]],
        "max_pool2d": [[torch.nn.MaxPool2d], [F.max_pool2d]],
        "adaptive_avg_pool2d": [
            [torch.nn.AdaptiveAvgPool2d],
            [F.adaptive_avg_pool2d],
        ],
    }
    return copy.deepcopy(supported_operators)


def _get_supported_symmetric_config_and_operators() -> List[OperatorConfig]:
    supported_config_and_operators: List[OperatorConfig] = []
    for quantization_config in [
        get_sima_quantization_config(),
        get_sima_quantization_config(is_qat=True),
        get_sima_quantization_config(is_qat=True, shift_aware=False),
    ]:
        ops = _supported_symmetric_quantized_operators()
        for pattern_list in ops.values():
            supported_config_and_operators.append(
                OperatorConfig(quantization_config, pattern_list)
            )
    return copy.deepcopy(supported_config_and_operators)


def get_sima_quantization_config(
    is_qat: bool = False,
    shift_aware: bool = True,
    activation_observer: Optional[str] = None,
    full_range_ste: Optional[bool] = None,
    learn_scales: Optional[bool] = None,
):
    # This configuration function only has one parameter (use QAT or not).
    # Sima has a preferred encoding for activation and weight tensors that give 
    # best possible results. Since QAT is a high-effort activity, we only use the
    # best quantization settings possible here.
    #

    # Activations
    # ---------------------------------------------------
    act_extra_args: Dict[str, Any] = {"eps": 2**-12}
    if is_qat:
        use_full_range_ste = (
            os.environ.get("SIMA_QAT_FULL_RANGE_STE", "0") == "1"
            if full_range_ste is None
            else bool(full_range_ste)
        )
        act_observer_or_fake_quant_ctr = (
            FullRangeSTEFakeQuantize if use_full_range_ste else FakeQuantize
        )
        observer_mode = (
            activation_observer
            or os.environ.get("SIMA_QAT_ACTIVATION_OBSERVER", "moving_average")
        ).lower()
        if use_full_range_ste and learn_scales is not None:
            act_extra_args["learn_scale"] = bool(learn_scales)
        if observer_mode == "minmax":
            # State-space recurrences can contain rare but legitimate channel
            # outliers.  An EMA of per-sample extrema clips those values by
            # orders of magnitude after calibration; a global MinMaxObserver
            # gives the strict INT8 graph a conservative, non-saturating range.
            act_extra_args["observer"] = MinMaxObserver
        elif observer_mode == "histogram":
            # HistogramObserver searches a clipped range that minimizes the
            # reconstruction L2 error, closely matching AFE's MSE calibration.
            act_extra_args["observer"] = HistogramObserver
        elif observer_mode == "moving_average":
            act_extra_args["observer"] = SimaMovingAverageMinMaxObserver
        else:
            raise ValueError(
                "SIMA_QAT_ACTIVATION_OBSERVER must be moving_average, minmax, or histogram, "
                f"found {observer_mode!r}"
            )
    else:
        # If QAT is disabled, we can add histogram observers to collect data.
        act_observer_or_fake_quant_ctr = HistogramObserver  # type: ignore[assignment]

    # Activations have a specific set of params that we don't need to change.
    # This is always per-tensor, signed integer encoding.
    # 
    act_quantization_spec = QuantizationSpec(
        dtype = torch.int8,
        quant_min = -128,
        quant_max = 127,
        qscheme = torch.per_tensor_affine,
        is_dynamic = False,
        observer_or_fake_quant_ctr = act_observer_or_fake_quant_ctr.with_args(
            **act_extra_args,
        ),
    )

    # Weights
    # ---------------------------------------------------
    # Weights will always be captured as per-channel symmetric.
    # Weight scales are often below the activation-scale floor.  Keeping the
    # old 2**-12 observer epsilon silently coarsened small Linear/Conv weights
    # and could make sx*sw/sy exceed the compiler's shift-0 range.  Float32
    # epsilon is the shared observer/shift-solver floor; activations retain
    # their independent 2**-12 policy above.
    wt_extra_args: Dict[str, Any] = {"eps": torch.finfo(torch.float32).eps}
    if is_qat and shift_aware:
        # Shift-aware QAT must expose quantized weights during the forward
        # pass. A bare observer postpones weight rounding until conversion and
        # cannot recover accuracy after the target grids are locked.
        weight_observer_or_fake_quant_ctr = FakeQuantize
        wt_extra_args["observer"] = MovingAveragePerChannelMinMaxObserver
    elif is_qat:
        # Explicit compatibility path for checkpoints created before
        # shift-aware weight fake quantization became the default.
        weight_observer_or_fake_quant_ctr = PerChannelMinMaxObserver
    else:
        weight_observer_or_fake_quant_ctr = PlaceholderObserver

    weight_quantization_spec = QuantizationSpec(
        dtype = torch.int8,
        quant_min = -127,     # use -128 ??
        quant_max = 127,
        qscheme = torch.per_channel_symmetric,
        ch_axis = 0,          # Weights are always channel-first
        is_dynamic = False,
        observer_or_fake_quant_ctr = weight_observer_or_fake_quant_ctr.with_args(
            **wt_extra_args
        ),
    )

    bias_quantization_spec = None
    quantization_config = QuantizationConfig(
        input_activation = act_quantization_spec,
        output_activation = act_quantization_spec,
        weight = weight_quantization_spec,
        bias = bias_quantization_spec,
        is_qat = is_qat,
    )
    return quantization_config


def _get_supported_config_and_operators() -> List[OperatorConfig]:
    return _get_supported_symmetric_config_and_operators()


class SimaQuantizer(Quantizer):
    """ This quantizer definition uses XNNPACK implementation for the majority of ops. This is because
        the XNNPACK code simply looks for appropriate patterns, and applies the QuantizationAnnotation
        attributes accordingly. The quantization rules are defined separately, and specified here using
        Sima properties.
    """
    supported_config_and_operators = _get_supported_config_and_operators()
    STATIC_QAT_ONLY_OPS = [
        "sima_conv_bn_hardtanh",
        "conv_bn_relu",
        "conv_bn",
    ]

    # static quantization ops (both PTQ and QAT)
    # Preserve the order that fusions come before singular ops
    STATIC_OPS = [
        "linear_relu",
        "linear",
        "sima_embedding",
        "sima_matmul",
        "sima_softmax",
        "sima_grid_sample",
        "sima_conv_add_or_mul_const",
        "sima_conv_hardtanh",
        "sima_conv_transpose2d",
        "conv_relu",
        "conv",
        # XNNPACK's source-partition Conv annotator treats repeated calls to
        # one module as a single partition because they share the same weight
        # get_attr node.  It consequently annotates only the first call.  A
        # strict W8A8 graph must cover every invocation, so finish any Conv2d
        # nodes left behind by the source-pattern pass.
        "sima_unannotated_conv2d",
        "adaptive_avg_pool2d",
        "max_pool2d",
        "sima_add_hardtanh",
        "add_relu",
        "add",
        "mul_relu",
        "mul",
        "sima_cat",
        "sima_sigmoid",
        "sima_erf",
        "sima_silu",
        "sima_slice_select_unsqueeze",
        "sima_batchnorm"
    ]

    def __init__(self):
        super().__init__()
        self.global_config: Optional[QuantizationConfig] = None
        self.operator_type_config: Dict[
            torch._ops.OpOverloadPacket, Optional[QuantizationConfig]
        ] = {}
        self.module_type_config: Dict[Callable, Optional[QuantizationConfig]] = {}
        self.module_name_config: Dict[str, Optional[QuantizationConfig]] = {}

    @classmethod
    def get_supported_quantization_configs(cls) -> List[QuantizationConfig]:
        op_configs: Set[QuantizationConfig] = set({})
        for spec, _ in cls.supported_config_and_operators:
            op_configs.add(spec)
        return list(op_configs)

    @classmethod
    def get_supported_operator_for_quantization_config(
        cls, quantization_config: Optional[QuantizationConfig]
    ) -> List[OperatorPatternType]:
        if quantization_config is None:
            all_ops = []
            for _, ops in cls.supported_config_and_operators:
                all_ops.extend(ops)
            return all_ops

        for config, ops in cls.supported_config_and_operators:
            # note: this assumes each entry in cls.supported_spec_and_operators
            # corresponds to one spec, e.g. we don't have
            # [(spec1, op_list1), (spec1, op_list2), (spec2, op_list3)]
            # where the first and second entry have the same spec but did not
            # merge the op list
            if config == quantization_config:
                return ops
        return []

    def set_global(self, quantization_config: QuantizationConfig) -> SimaQuantizer:
        self.global_config = quantization_config
        return self

    def set_operator_type(
        self,
        operator_type: torch._ops.OpOverloadPacket,
        quantization_config: QuantizationConfig,
    ) -> SimaQuantizer:
        self.operator_type_config[operator_type] = quantization_config
        return self

    def set_module_type(
        self, module_type: Callable, quantization_config: QuantizationConfig
    ):
        """Set quantization_config for a submodule with type: `module_type`, for example:
        quantizer.set_module_name(Sub) or quantizer.set_module_name(nn.Linear), it will quantize all supported operator/operator
        patterns in the submodule with this module type with the given `quantization_config`
        """
        self.module_type_config[module_type] = quantization_config
        return self

    def set_module_name(
        self, module_name: str, quantization_config: Optional[QuantizationConfig]
    ):
        """Set quantization_config for a submodule with name: `module_name`, for example:
        quantizer.set_module_name("blocks.sub"), it will quantize all supported operator/operator
        patterns in the submodule with this module name with the given `quantization_config`
        """
        assert (
            quantization_config is not None
        ), " quantization_config == None is not supported yet"
        self.module_name_config[module_name] = quantization_config
        return self

    def transform_for_annotation(
        self, model: torch.fx.GraphModule
    ) -> torch.fx.GraphModule:
        """Transforms scalar values to tensor attributes"""
        return _convert_scalars_to_attrs(model)

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        """just handling global spec for now"""
        # Dynamic is unsupported.
        if self.global_config and self.global_config.input_activation.is_dynamic:  # type: ignore[union-attr]
            assert False, "Error: dynamic quantization is unsupported on Sima models."
        model = self._annotate_for_static_quantization_config(model)
        self._remove_non_float_qspecs(model)
        self._fuse_attention_score_softmax_boundaries(model)
        self._fuse_transformer_residual_norm_boundaries(model)
        self._fuse_deformable_weighted_reduction(model)
        propagate_annotation(model)
        return model

    @staticmethod
    def _remove_non_float_qspecs(model: torch.fx.GraphModule) -> None:
        """Never insert activation Q/DQ on masks, indices, or shape tensors."""
        def is_non_float(node: Node) -> bool:
            value = node.meta.get("val")
            return isinstance(value, torch.Tensor) and not value.is_floating_point()

        for node in model.graph.nodes:
            annotation = node.meta.get("quantization_annotation")
            if annotation is None or not annotation._annotated:
                continue
            annotation.input_qspec_map = {
                input_node: qspec
                for input_node, qspec in annotation.input_qspec_map.items()
                if not is_non_float(input_node)
            }
            if is_non_float(node):
                annotation.output_qspec = None

    @staticmethod
    def _fuse_attention_score_softmax_boundaries(model: torch.fx.GraphModule) -> None:
        """Keep QK score accumulation and masking inside fused attention.

        Q/K/V operands and the Softmax result remain A8.  The score tensor is
        not rounded once before applying the mask and again at Softmax input.
        """
        score_targets = {
            torch.ops.aten.baddbmm.default,
            torch.ops.aten.bmm.default,
            torch.ops.aten.matmul.default,
            torch.ops.aten.mm.default,
        }
        softmax_targets = {
            torch.ops.aten._softmax.default,
            torch.ops.aten.softmax.int,
        }
        for softmax in model.graph.nodes:
            if softmax.op != "call_function" or softmax.target not in softmax_targets:
                continue
            if not softmax.args or not isinstance(softmax.args[0], Node):
                continue
            score = softmax.args[0]
            # MultiheadAttention commonly inserts an Add mask between BMM and
            # Softmax.  Only walk that single arithmetic boundary.
            score_input = score
            if (
                score.op == "call_function"
                and score.target == torch.ops.aten.add.Tensor
                and score.args
                and isinstance(score.args[0], Node)
            ):
                score_input = score.args[0]
            if score_input.op != "call_function" or score_input.target not in score_targets:
                continue
            softmax_annotation = softmax.meta.get("quantization_annotation")
            if softmax_annotation is not None and softmax_annotation._annotated:
                softmax_annotation.input_qspec_map = {}
            score_annotation = score_input.meta.get("quantization_annotation")
            if score_annotation is not None and score_annotation._annotated:
                score_annotation.output_qspec = None
            if score is not score_input:
                add_annotation = score.meta.get("quantization_annotation")
                if add_annotation is not None and add_annotation._annotated:
                    add_annotation.input_qspec_map = {}
                    add_annotation.output_qspec = None

    @staticmethod
    def _fuse_transformer_residual_norm_boundaries(model: torch.fx.GraphModule) -> None:
        """Keep projection accumulator and residual Add inside LayerNorm."""
        pass_through = {
            torch.ops.aten.view.default,
            torch.ops.aten.reshape.default,
            torch.ops.aten.transpose.int,
            torch.ops.aten.permute.default,
            torch.ops.aten.clone.default,
            torch.ops.aten.contiguous.default,
        }

        def upstream(node: Node) -> Node:
            while (
                node.op == "call_function"
                and node.target in pass_through
                and node.args
                and isinstance(node.args[0], Node)
            ):
                node = node.args[0]
            return node

        for norm in model.graph.nodes:
            if norm.op != "call_function" or norm.target != torch.ops.aten.layer_norm.default:
                continue
            add = upstream(norm.args[0]) if norm.args and isinstance(norm.args[0], Node) else None
            if add is None or add.op != "call_function" or add.target != torch.ops.aten.add.Tensor:
                continue
            add_annotation = add.meta.get("quantization_annotation")
            if add_annotation is None or not add_annotation._annotated:
                continue
            add_annotation.input_qspec_map = {}
            add_annotation.output_qspec = None
            for value in add.args[:2]:
                if not isinstance(value, Node):
                    continue
                producer = upstream(value)
                if producer.op != "call_function" or producer.target != torch.ops.aten.linear.default:
                    continue
                producer_annotation = producer.meta.get("quantization_annotation")
                if producer_annotation is not None and producer_annotation._annotated:
                    producer_annotation.output_qspec = None

    @staticmethod
    def _fuse_deformable_weighted_reduction(model: torch.fx.GraphModule) -> None:
        """Keep point weighting in the INT32 accumulator until ReduceSum."""
        sum_targets = {
            torch.ops.aten.sum.dim_IntList,
            torch.ops.aten.sum.default,
        }

        def reaches_grid_sample(value: Node) -> bool:
            pending = [value]
            visited = set()
            while pending and len(visited) < 32:
                current = pending.pop()
                if current in visited:
                    continue
                visited.add(current)
                if current.target == torch.ops.aten.grid_sampler.default:
                    return True
                if current.op == "call_function" and current.target in {
                    torch.ops.aten.cat.default,
                    torch.ops.aten.mul.Tensor,
                }:
                    pending.extend(current.all_input_nodes)
            return False

        # Multiplication by the pre-quantization validity mask only clears
        # sampled codes. It must retain the GridSample data/output grid and
        # must not quantize the exact 0/1 mask on an independent affine grid.
        for product in model.graph.nodes:
            if product.op != "call_function" or product.target != torch.ops.aten.mul.Tensor:
                continue
            grid = next(
                (
                    value for value in product.all_input_nodes
                    if value.target == torch.ops.aten.grid_sampler.default
                ),
                None,
            )
            if grid is None:
                continue
            annotation = product.meta.get("quantization_annotation")
            if annotation is None or not annotation._annotated:
                continue
            data = grid.args[0]
            if not isinstance(data, Node):
                continue
            shared = SharedQuantizationSpec((data, grid))
            annotation.input_qspec_map = {grid: shared}
            annotation.output_qspec = shared

        for reduce_sum in model.graph.nodes:
            if reduce_sum.op != "call_function" or reduce_sum.target not in sum_targets:
                continue
            if not reduce_sum.args or not isinstance(reduce_sum.args[0], Node):
                continue
            product = reduce_sum.args[0]
            if product.op != "call_function" or product.target != torch.ops.aten.mul.Tensor:
                continue
            if not any(reaches_grid_sample(value) for value in product.all_input_nodes):
                continue
            annotation = product.meta.get("quantization_annotation")
            if annotation is not None and annotation._annotated:
                # Both operands remain A8. The product is accumulated at wide
                # precision and narrowed only at the reduction's consumer.
                annotation.output_qspec = None

    def _annotate_all_static_patterns(
        self,
        model: torch.fx.GraphModule,
        quantization_config: Optional[QuantizationConfig],
        filter_fn: Optional[Callable[[Node], bool]] = None,
    ) -> torch.fx.GraphModule:
        # TODO: implement the support for None to be canceling out previous annotations
        if quantization_config is None:
            return model

        if quantization_config.is_qat:
            for op in self.STATIC_QAT_ONLY_OPS:
                OP_TO_ANNOTATOR[op](model, quantization_config, filter_fn)
        for op in self.STATIC_OPS:
            annotator = OP_TO_ANNOTATOR.get(op)
            if annotator is None and op == "max_pool2d":
                # torch 2.8 propagates the producer quantization grid through
                # MaxPool and no longer exposes a standalone XNNPACK annotator.
                continue
            if annotator is None:
                raise KeyError(f"missing PT2E annotator for {op!r}")
            annotator(model, quantization_config, filter_fn)
        return model

    def _annotate_for_static_quantization_config(
        self, model: torch.fx.GraphModule
    ) -> torch.fx.GraphModule:
        module_name_list = list(self.module_name_config.keys())
        for module_name, config in self.module_name_config.items():
            self._annotate_all_static_patterns(
                model, config, _get_module_name_filter(module_name)
            )

        tp_list = list(self.module_type_config.keys())
        for module_type, config in self.module_type_config.items():
            self._annotate_all_static_patterns(
                model, config, _get_module_type_filter(module_type)
            )

        self._annotate_all_static_patterns(
            model,
            self.global_config,
            _get_not_module_type_or_name_filter(tp_list, module_name_list),
        )
        return model

    def validate(self, model: torch.fx.GraphModule) -> None:
        pass

    @classmethod
    def get_supported_operators(cls) -> List[OperatorConfig]:
        return cls.supported_config_and_operators


@register_annotator("sima_conv_transpose2d")
def _sima_annotate_conv_transpose2d(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    """Annotate ConvTranspose2d with target-exact symmetric INT8 weights.

    PyTorch stores ConvTranspose2d weights as ``[C_in, C_out/groups, kH, kW]``.
    PT2E 2.3 rewrites a requested axis-1 quantizer to axis 0 for this operator,
    which would silently quantize input rather than output channels.  Use a
    symmetric per-tensor grid for this single layer instead.  It remains strict
    W8A8, is faithfully simulated during training, and exports without an
    ambiguous channel-axis contract.
    """
    if quantization_config is None:
        return []
    annotated_partitions: List[List[Node]] = []
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target != torch.ops.aten.conv_transpose2d.input:
            continue
        partition = [node]
        weight = node.args[1]
        if not isinstance(weight, Node):
            raise RuntimeError("ConvTranspose2d weight must be an FX node")
        partition.append(weight)
        bias = node.args[2] if len(node.args) > 2 else None
        if isinstance(bias, Node):
            partition.append(bias)
        if _is_annotated(partition):
            continue
        if filter_fn and any(not filter_fn(member) for member in partition):
            continue
        base_weight_qspec = get_weight_qspec(quantization_config)
        weight_qspec = QuantizationSpec(
            dtype=base_weight_qspec.dtype,
            quant_min=base_weight_qspec.quant_min,
            quant_max=base_weight_qspec.quant_max,
            qscheme=torch.per_tensor_symmetric,
            is_dynamic=False,
            observer_or_fake_quant_ctr=FakeQuantize.with_args(
                observer=MovingAverageMinMaxObserver,
                eps=2**-12,
            ),
        )
        input_qspec_map = {
            node.args[0]: get_input_act_qspec(quantization_config),
            weight: weight_qspec,
        }
        if isinstance(bias, Node):
            input_qspec_map[bias] = get_bias_qspec(quantization_config)
        node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=get_output_act_qspec(quantization_config),
            _annotated=True,
        )
        _mark_nodes_as_annotated(partition)
        annotated_partitions.append(partition)
    return annotated_partitions


@register_annotator("sima_unannotated_conv2d")
def _sima_annotate_unannotated_conv2d(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    """Annotate Conv2d invocations missed when a module is called repeatedly.

    PT2E/XNNPACK source partitions include the shared weight node when they
    decide whether a Conv partition is already annotated.  After annotating
    the first invocation, that shared node makes all later invocations appear
    annotated even though the Conv nodes have no qspec.  Check the operation
    itself instead and attach the normal SiMa activation/weight/bias specs to
    every remaining call.  This is not a DepthART special case: weight tying
    and recurrent/module-reuse patterns can trigger it in any PyTorch model.
    """
    if quantization_config is None:
        return []
    annotated_partitions: List[List[Node]] = []
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target != torch.ops.aten.conv2d.default:
            continue
        annotation = node.meta.get("quantization_annotation")
        if annotation is not None and annotation._annotated:
            continue
        if filter_fn and not filter_fn(node):
            continue
        if len(node.args) < 2 or not isinstance(node.args[0], Node):
            raise RuntimeError("Conv2d activation must be an FX node")
        weight = node.args[1]
        if not isinstance(weight, Node):
            raise RuntimeError("Conv2d weight must be an FX node")
        bias = node.args[2] if len(node.args) > 2 else None
        input_qspec_map = {
            node.args[0]: get_input_act_qspec(quantization_config),
            weight: get_weight_qspec(quantization_config),
        }
        if isinstance(bias, Node):
            input_qspec_map[bias] = get_bias_qspec(quantization_config)
        node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=get_output_act_qspec(quantization_config),
            _annotated=True,
        )
        # Do not mark the shared weight get_attr as an operation partition:
        # each Conv invocation owns its own annotation while reusing that
        # weight's resulting fake-quant module is valid and intentional.
        _mark_nodes_as_annotated([node])
        annotated_partitions.append([node])
    return annotated_partitions


def _annotate_single_op(
    quantization_config: QuantizationConfig,
    op_partitions: List[object],
    op_check: Callable,
) -> List[List[Node]]:
    """ This is a helper function which annotates a single operation in a graph with Fakequant
        observers. This function assumes single-input operators, and is not suitable for multi-input
        ops which may have constant inputs (e.g. Conv2D).
    """
    annotated_partitions = []
    for op_partition in op_partitions:
        op_node = op_partition.output_nodes[0]
        if _is_annotated([op_node]):
            continue

        if not op_check(op_node):
            continue

        annotated_partitions.append(op_partition.nodes)

        input_act_qspec = get_input_act_qspec(quantization_config)
        input_act0 = op_node.args[0]

        input_qspec_map = {}
        if isinstance(input_act0, Node):
            input_qspec_map[input_act0] = input_act_qspec

        output_act_qspec = get_output_act_qspec(quantization_config)

        op_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=output_act_qspec,
            _annotated=True,
        )
    return annotated_partitions


@register_annotator("sima_matmul")
def _sima_annotate_matmul(
    model: torch.fx.GraphModule,
    quantization_config: QuantizationConfig,
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> List[List[Node]]:
    """Annotate activation-by-activation matrix multiplication."""
    targets = {
        torch.ops.aten.bmm.default,
        torch.ops.aten.matmul.default,
        torch.ops.aten.mm.default,
    }
    annotated_partitions = []
    for node in model.graph.nodes:
        if node.op != "call_function" or node.target not in targets:
            continue
        if filter_fn is not None and not filter_fn(node):
            continue
        if _is_annotated([node]):
            continue

        input_qspec = get_input_act_qspec(quantization_config)
        input_nodes = []
        for input_node in node.args[:2]:
            if not isinstance(input_node, Node):
                break
            if _is_input_large_scalar(input_node, model) or _is_input_non_float_tensor(input_node):
                break
            input_nodes.append(input_node)
        if len(input_nodes) != 2:
            continue
        # A self-product can use the same activation for both operands. One
        # map entry correctly shares its fake-quantized value across both
        # edges; requiring two unique nodes would silently skip that MatMul.
        input_qspec_map = {input_node: input_qspec for input_node in input_nodes}

        node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=get_output_act_qspec(quantization_config),
            _annotated=True,
        )
        annotated_partitions.append([node])
    return annotated_partitions


@register_annotator("sima_embedding")
def _sima_annotate_embedding(
    model: torch.fx.GraphModule,
    quantization_config: QuantizationConfig,
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> List[List[Node]]:
    """Store embedding tables and their lookup result on one signed-W8 grid.

    Embedding indices are structural INT64 values, not activations.  A single
    per-tensor table scale lets MLA implement the lookup as an exact INT8 copy;
    per-row scales would require a data-dependent requantization after Gather.
    """
    base_weight_qspec = get_weight_qspec(quantization_config)
    table_qspec = QuantizationSpec(
        dtype=base_weight_qspec.dtype,
        quant_min=base_weight_qspec.quant_min,
        quant_max=base_weight_qspec.quant_max,
        qscheme=torch.per_tensor_symmetric,
        is_dynamic=False,
        observer_or_fake_quant_ctr=FakeQuantize.with_args(
            observer=MovingAverageMinMaxObserver,
            eps=2**-12,
        ),
    )
    annotated_partitions: List[List[Node]] = []
    for node in model.graph.nodes:
        if node.op != "call_function" or node.target != torch.ops.aten.embedding.default:
            continue
        if filter_fn is not None and not filter_fn(node):
            continue
        if len(node.args) < 2:
            continue
        weight = node.args[0]
        indices = node.args[1]
        if not isinstance(weight, Node) or not isinstance(indices, Node):
            continue
        partition = [node, weight]
        if _is_annotated(partition):
            continue
        node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map={weight: table_qspec},
            output_qspec=SharedQuantizationSpec((weight, node)),
            _annotated=True,
        )
        _mark_nodes_as_annotated(partition)
        annotated_partitions.append(partition)
    return annotated_partitions


@register_annotator("sima_softmax")
def _sima_annotate_softmax(
    model: torch.fx.GraphModule,
    quantization_config: QuantizationConfig,
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> List[List[Node]]:
    """Annotate Softmax input and output activations."""
    targets = {
        torch.ops.aten._softmax.default,
        torch.ops.aten.softmax.int,
    }
    annotated_partitions = []
    for node in model.graph.nodes:
        if node.op != "call_function" or node.target not in targets:
            continue
        if filter_fn is not None and not filter_fn(node):
            continue
        if _is_annotated([node]):
            continue

        input_node = node.args[0]
        if not isinstance(input_node, Node):
            continue
        if _is_input_non_float_tensor(input_node):
            continue
        node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map={input_node: get_input_act_qspec(quantization_config)},
            output_qspec=get_output_act_qspec(quantization_config),
            _annotated=True,
        )
        annotated_partitions.append([node])
    return annotated_partitions


@register_annotator("sima_grid_sample")
def _sima_annotate_grid_sample(
    model: torch.fx.GraphModule,
    quantization_config: QuantizationConfig,
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> List[List[Node]]:
    """Annotate the strict MLA full-2D GridSample W8A8 contract.

    MLA consumes signed INT8 normalized coordinates on the exact 1/128 grid.
    Grounding DINO's explicit validity mask preserves zeros-padding semantics
    for coordinates that would otherwise saturate onto the image border.
    Its result uses the data input's activation grid, so the output shares that
    edge qspec rather than introducing an independently observed requantizer.
    """
    grid_qspec = FixedQParamsQuantizationSpec(
        dtype=torch.int8,
        scale=1.0 / 128.0,
        zero_point=0,
        quant_min=-128,
        quant_max=127,
        qscheme=torch.per_tensor_affine,
        is_dynamic=False,
    )
    annotated_partitions: List[List[Node]] = []
    for node in model.graph.nodes:
        if node.op != "call_function" or node.target != torch.ops.aten.grid_sampler.default:
            continue
        if filter_fn is not None and not filter_fn(node):
            continue
        if _is_annotated([node]):
            continue
        if len(node.args) < 2:
            continue
        data, grid = node.args[:2]
        if not isinstance(data, Node) or not isinstance(grid, Node):
            continue
        data_qspec = get_input_act_qspec(quantization_config)
        node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map={data: data_qspec, grid: grid_qspec},
            output_qspec=SharedQuantizationSpec((data, node)),
            _annotated=True,
        )
        annotated_partitions.append([node])
    return annotated_partitions


@register_annotator("sima_sigmoid")
def _sima_annotate_sigmoid(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    sig_partitions = get_source_partitions(gm.graph, [torch.sigmoid, F.sigmoid, torch.nn.Sigmoid], filter_fn)
    sig_partitions = list(itertools.chain.from_iterable(sig_partitions.values()))

    def _sig_target_check(sig_node: Node) -> bool:
        if sig_node.target != torch.ops.aten.sigmoid.default:
            # TODO: change this to AnnotationException
            raise Exception(
                f"Expected sigmoid node: torch.ops.aten.sigmoid.default, but found {sig_node.target}"
                " please check if you are calling the correct capture API"
            )
        return True

    return _annotate_single_op(
        quantization_config = quantization_config,
        op_partitions = sig_partitions,
        op_check = _sig_target_check,
    )


@register_annotator("sima_erf")
def _sima_annotate_erf(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    """Quantize explicit Erf inputs/outputs used by target-realizable GELU.

    AFE lowers GELU into primitive INT8 stages and therefore materializes an
    INT8 grid at Erf.  PT2E has no stock Erf annotator, so without this rule
    QAT cannot see the target boundary even when GELU is explicitly
    decomposed in PyTorch.
    """
    partitions = get_source_partitions(gm.graph, [torch.erf], filter_fn)
    partitions = list(itertools.chain.from_iterable(partitions.values()))

    def _erf_target_check(erf_node: Node) -> bool:
        if erf_node.target != torch.ops.aten.erf.default:
            raise Exception(
                f"Expected erf node torch.ops.aten.erf.default, found {erf_node.target}"
            )
        return True

    return _annotate_single_op(
        quantization_config=quantization_config,
        op_partitions=partitions,
        op_check=_erf_target_check,
    )


@register_annotator("sima_silu")
def _sima_annotate_silu(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    silu_partitions = get_source_partitions(gm.graph, [F.silu, torch.nn.SiLU], filter_fn)
    silu_partitions = list(itertools.chain.from_iterable(silu_partitions.values()))

    def _silu_target_check(silu_node: Node) -> bool:
        if silu_node.target not in [torch.ops.aten.silu_.default, torch.ops.aten.silu.default]:
            # TODO: change this to AnnotationException
            raise Exception(
                f"Expected SiLU node: torch.ops.aten.silu_.default, but found {silu_node.target}"
                " please check if you are calling the correct capture API"
            )
        return True

    return _annotate_single_op(
        quantization_config = quantization_config,
        op_partitions = silu_partitions,
        op_check = _silu_target_check,
    )


@register_annotator("sima_add_hardtanh")
def _sima_annotate_add_hardtanh(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    fused_partitions = find_sequential_partitions(
        gm, [torch.add, torch.nn.Hardtanh], filter_fn=filter_fn
    )
    annotated_partitions = []
    for fused_partition in fused_partitions:
        add_partition, hardtanh_partition = fused_partition
        annotated_partitions.append(add_partition.nodes + hardtanh_partition.nodes)
        if len(hardtanh_partition.output_nodes) > 1:
            raise ValueError("hardtanh partition has more than one output node")
        hardtanh_node = hardtanh_partition.output_nodes[0]
        if len(add_partition.output_nodes) > 1:
            raise ValueError("add partition has more than one output node")
        add_node = add_partition.output_nodes[0]

        if _is_annotated([hardtanh_node, add_node]):
            continue

        input_act_qspec = get_input_act_qspec(quantization_config)
        output_act_qspec = get_output_act_qspec(quantization_config)

        input_qspec_map = {}
        input_act0 = add_node.args[0]
        if isinstance(input_act0, Node):
            if _is_input_large_scalar(input_act0, gm):
                continue
            if _is_input_non_float_tensor(input_act0):
                continue
            input_qspec_map[input_act0] = input_act_qspec

        input_act1 = add_node.args[1]
        if isinstance(input_act1, Node):
            if _is_input_large_scalar(input_act1, gm):
                continue
            if _is_input_non_float_tensor(input_act1):
                continue
            input_qspec_map[input_act1] = input_act_qspec

        add_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            _annotated=True,
        )
        hardtanh_node.meta["quantization_annotation"] = QuantizationAnnotation(
            output_qspec=output_act_qspec,
            _annotated=True,
        )
    return annotated_partitions


@register_annotator("sima_conv_hardtanh")
def _sima_annotate_conv_hardtanh(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    annotated_partitions = []
    for n in gm.graph.nodes:
        if n.op != "call_function" or n.target not in [
            torch.ops.aten.hardtanh.default,
            torch.ops.aten.hardtanh_.default,
        ]:
            continue
        hardtanh_node = n
        maybe_conv_node = n.args[0]
        if (
            not isinstance(maybe_conv_node, Node)
            or maybe_conv_node.op != "call_function"
            or maybe_conv_node.target
            not in [
                torch.ops.aten.conv1d.default,
                torch.ops.aten.conv2d.default,
            ]
        ):
            continue
        conv_node = maybe_conv_node

        input_qspec_map = {}
        input_act = conv_node.args[0]
        assert isinstance(input_act, Node)
        input_qspec_map[input_act] = get_input_act_qspec(quantization_config)

        weight = conv_node.args[1]
        assert isinstance(weight, Node)
        input_qspec_map[weight] = get_weight_qspec(quantization_config)

        # adding weight node to the partition as well
        partition = [hardtanh_node, conv_node, conv_node.args[1]]
        bias = conv_node.args[2] if len(conv_node.args) > 2 else None
        if isinstance(bias, Node):
            input_qspec_map[bias] = get_bias_qspec(quantization_config)
            partition.append(bias)

        if _is_annotated(partition):
            continue

        if filter_fn and any(not filter_fn(n) for n in partition):
            continue

        conv_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map, _annotated=True
        )
        hardtanh_node.meta["quantization_annotation"] = QuantizationAnnotation(
            output_qspec=get_output_act_qspec(quantization_config),  # type: ignore[arg-type]
            _annotated=True,
        )
        _mark_nodes_as_annotated(partition)
        annotated_partitions.append(partition)
    return annotated_partitions


@register_annotator("sima_conv_bn_hardtanh")
def _sima_annotate_conv_bn_hardtanh(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    """
    Find conv + batchnorm + hardtanh parititions
    Note: This is only used for QAT. In PTQ, batchnorm should already be fused into the conv.
    """
    def get_pattern(conv_fn: Callable, hardtanh_is_inplace: bool):
        def _conv_bn(x, conv_weight, conv_bias, bn_weight, bn_bias, bn_rm, bn_rv):
            conv = conv_fn(x, conv_weight, conv_bias)
            bn = F.batch_norm(conv, bn_rm, bn_rv, bn_weight, bn_bias, training=True)
            output = F.hardtanh_(bn) if hardtanh_is_inplace else F.hardtanh(bn)
            return output, {
                "input": x,
                "conv": conv,
                "weight": conv_weight,
                "bias": conv_bias,
                "output": output,
            }

        return _WrapperModule(_conv_bn)

    # Needed for matching, otherwise the matches gets filtered out due to unused
    # nodes returned by batch norm
    gm.graph.eliminate_dead_code()
    gm.recompile()

    matches = []
    combinations = [
        (F.conv1d, _conv1d_bn_example_inputs),
        (F.conv2d, _conv2d_bn_example_inputs),
    ]

    # Add `is_cuda` and `hardtanh_is_inplace` dimensions
    combinations = itertools.product(
        combinations,
        [True, False] if torch.cuda.is_available() else [False],  # is_cuda
        [True, False],  # hardtanh_is_inplace
    )

    # Match against all conv dimensions and cuda variants
    for (conv_fn, example_inputs), is_cuda, hardtanh_is_inplace in combinations:
        pattern = get_pattern(conv_fn, hardtanh_is_inplace)
        pattern = get_aten_graph_module(pattern, example_inputs, is_cuda)
        pattern.graph.eliminate_dead_code()
        pattern.recompile()
        matcher = SubgraphMatcherWithNameNodeMap(pattern, ignore_literals=True)
        matches.extend(matcher.match(gm.graph))

    # Annotate nodes returned in the matches
    annotated_partitions = []
    for match in matches:
        name_node_map = match.name_node_map
        input_node = name_node_map["input"]
        conv_node = name_node_map["conv"]
        weight_node = name_node_map["weight"]
        bias_node = name_node_map["bias"]
        output_node = name_node_map["output"]

        # TODO: annotate the uses of input, weight, and bias separately instead
        # of assuming they come from a single conv node. This is not possible today
        # because input may have multiple users, and we can't rely on the conv node
        # always being the first user. This was the case in models with skip
        # connections like resnet18

        # Validate conv args
        if conv_node.args[0] is not input_node:
            raise ValueError("Conv arg did not contain input node ", input_node)
        if conv_node.args[1] is not weight_node:
            raise ValueError("Conv arg did not contain weight node ", weight_node)
        if len(conv_node.args) > 2 and conv_node.args[2] is not bias_node:
            raise ValueError("Conv arg did not contain bias node ", bias_node)

        # Skip if the partition is already annotated or is filtered out by the user
        partition = [conv_node, weight_node]
        if bias_node is not None:
            partition.append(bias_node)
        if _is_annotated(partition):
            continue
        if filter_fn and any(not filter_fn(n) for n in partition):
            continue

        # Annotate conv inputs and pattern output
        input_qspec_map = {}
        input_qspec_map[input_node] = get_input_act_qspec(quantization_config)
        input_qspec_map[weight_node] = get_weight_qspec(quantization_config)
        if bias_node is not None:
            input_qspec_map[bias_node] = get_bias_qspec(quantization_config)
        conv_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            _annotated=True,
        )
        output_node.meta["quantization_annotation"] = QuantizationAnnotation(
            output_qspec=get_output_act_qspec(quantization_config),  # type: ignore[arg-type]
            _annotated=True,
        )
        _mark_nodes_as_annotated(partition)
        annotated_partitions.append(partition)
    return annotated_partitions


@register_annotator("sima_conv_add_or_mul_const")
def _sima_annotate_conv_add_or_mul_const(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    annotated_partitions = []
    for n in gm.graph.nodes:
        if n.op != "call_function" or n.target not in [
            torch.ops.aten.add.Tensor,
            torch.ops.aten.mul.Tensor,
        ]:
            continue

        #check if any args is a constant

        if n.args[0].op == "get_attr":
            conv_node_id = 1
        elif n.args[1].op == "get_attr":
            conv_node_id = 0
        else:
            continue

        op_node = n
        # AFE recognizes Conv -> (x * 1/sqrt(2)) -> Erf as the primitive
        # lowering of GELU and materializes an INT8 grid between Conv and the
        # scale Multiply. Treating this Multiply as the generic fused
        # Conv+constant pattern hides that grid from PT2E QAT. Leave the pair
        # unfused so the normal Conv and Mul annotators expose both target
        # boundaries. This is semantic target matching, not a model-specific
        # name check.
        if (
            n.target == torch.ops.aten.mul.Tensor
            and any(
                user.op == "call_function"
                and user.target == torch.ops.aten.erf.default
                for user in n.users
            )
        ):
            continue
        maybe_conv_node = n.args[conv_node_id]
        if (
            not isinstance(maybe_conv_node, Node)
            or maybe_conv_node.op != "call_function"
            or maybe_conv_node.target
            not in [
                torch.ops.aten.conv1d.default,
                torch.ops.aten.conv2d.default,
            ]
        ):
            continue
        conv_node = maybe_conv_node

        input_qspec_map = {}
        input_act = conv_node.args[0]
        assert isinstance(input_act, Node)
        input_qspec_map[input_act] = get_input_act_qspec(quantization_config)

        weight = conv_node.args[1]
        assert isinstance(weight, Node)
        input_qspec_map[weight] = get_weight_qspec(quantization_config)

        # adding weight node to the partition as well
        partition = [op_node, conv_node, conv_node.args[1]]
        bias = conv_node.args[2] if len(conv_node.args) > 2 else None
        if isinstance(bias, Node):
            input_qspec_map[bias] = get_bias_qspec(quantization_config)
            partition.append(bias)

        if _is_annotated(partition):
            continue

        if filter_fn and any(not filter_fn(n) for n in partition):
            continue

        conv_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map, _annotated=True
        )
        op_node.meta["quantization_annotation"] = QuantizationAnnotation(
            output_qspec=get_output_act_qspec(quantization_config),  # type: ignore[arg-type]
            _annotated=True,
        )
        _mark_nodes_as_annotated(partition)
        annotated_partitions.append(partition)
    return annotated_partitions

@register_annotator("sima_slice_select_unsqueeze")
def _sima_annotate_slice_select_unsqueeze(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    annotated_partitions = []
    input_act_qspec = get_input_act_qspec(quantization_config)
    output_act_qspec = get_output_act_qspec(quantization_config)

    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target not in [
            torch.ops.aten.unsqueeze.default,
        ]:
            continue
        unsqueeze_node = node
        maybe_select_node = node.args[0]
        if (
            not isinstance(maybe_select_node, Node)
            or maybe_select_node.op != "call_function"
            or maybe_select_node.target != torch.ops.aten.select.int
        ):
            continue

        select_node = maybe_select_node
        maybe_slice_node = select_node.args[0]
        if (
            not isinstance(maybe_slice_node, Node)
            or maybe_slice_node.op != "call_function"
            or maybe_slice_node.target != torch.ops.aten.slice.Tensor
        ):
            continue
        
        slice_node = maybe_slice_node
        
        input_qspec_map = {}
        input_act = slice_node.args[0]
        assert isinstance(input_act, Node)
        input_qspec_map[input_act] = input_act_qspec

        # adding weight node to the partition as well
        partition = [unsqueeze_node, select_node, slice_node]

        if _is_annotated(partition):
            continue

        if filter_fn and any(not filter_fn(n) for n in partition):
            continue

        slice_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            _annotated=True,
        )
        unsqueeze_node.meta["quantization_annotation"] = QuantizationAnnotation(
            output_qspec=output_act_qspec,
            _annotated=True,
        )
        _mark_nodes_as_annotated(partition)
        annotated_partitions.append(partition)
    return annotated_partitions

@register_annotator("sima_dropout")
def _sima_annotate_dropout(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    annotated_partitions = []
    input_act_qspec = get_input_act_qspec(quantization_config)
    output_act_qspec = get_output_act_qspec(quantization_config)

    dropout_nodes = []
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target not in [
            torch.ops.aten.dropout.default,
            torch.ops.aten.dropout_.default,
        ]:
            continue
        dropout_node = node
        dropout_nodes.append(dropout_node)
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.args[0] not in dropout_nodes:
            continue
        node_after = node
        dropout_node = node.args[0]
        
        input_qspec_map = {}
        input_act = dropout_node.args[0]
        assert isinstance(input_act, Node)
        input_qspec_map[input_act] = input_act_qspec

        # adding weight node to the partition as well
        partition = [node_after, dropout_node]

        if _is_annotated(partition):
            continue

        if filter_fn and any(not filter_fn(n) for n in partition):
            continue

        dropout_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            _annotated=True,
        )
        node_after.meta["quantization_annotation"] = QuantizationAnnotation(
            output_qspec=output_act_qspec,
            _annotated=True,
        )
        _mark_nodes_as_annotated(partition)
        annotated_partitions.append(partition)
    return annotated_partitions

@register_annotator("sima_batchnorm")
def _annotate_batchnorm(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    annotated_partitions = []
    for n in gm.graph.nodes:
        if n.op != "call_function" or n.target not in [
            operator.getitem
        ]:
            continue
        getitem_node = n
        maybe_batchnorm_node = n.args[0]
        if (
            not isinstance(maybe_batchnorm_node, Node)
            or maybe_batchnorm_node.op != "call_function"
            or maybe_batchnorm_node.target
            not in [
                torch.ops.aten._native_batch_norm_legit.default
            ]
        ):
            continue
        batchnorm_node = maybe_batchnorm_node

        input_qspec_map = {}
        input_act = batchnorm_node.args[0]
        assert isinstance(input_act, Node)
        input_qspec_map[input_act] = get_input_act_qspec(quantization_config)

        partition = [getitem_node, batchnorm_node]

        if _is_annotated(partition):
            continue

        if filter_fn and any(not filter_fn(n) for n in partition):
            continue

        batchnorm_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            _annotated=True,
        )

        getitem_node.meta["quantization_annotation"] = QuantizationAnnotation(
            output_qspec=get_output_act_qspec(quantization_config),
            _annotated=True,
        )

        _mark_nodes_as_annotated(partition)
        annotated_partitions.append(partition)
    return annotated_partitions


@register_annotator("sima_cat")
def _sima_annotate_cat(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    cat_partitions = get_source_partitions(gm.graph, [torch.cat], filter_fn)
    cat_partitions = list(itertools.chain.from_iterable(cat_partitions.values()))
    # ``get_source_partitions`` does not return a partition for a cat whose
    # input list aliases one tensor repeatedly (for example a parameter-free
    # C1 -> C16 public-output pack).  Such cats are still ordinary aten cats
    # and must not silently escape QAT.  Add unpartitioned aten nodes as
    # single-node partitions while retaining source-partition filtering for
    # the common case.
    partitioned_cat_nodes = {
        partition.output_nodes[0] for partition in cat_partitions
    }
    quantize_all_concat = os.environ.get("SIMA_QAT_QUANTIZE_ALL_CONCAT", "1") == "1"
    fallback_cat_nodes = [
        node
        for node in gm.graph.nodes
        if node.target == torch.ops.aten.cat.default
        and node not in partitioned_cat_nodes
        and (
            quantize_all_concat
            or (
                isinstance(node.args[0], (list, tuple))
                and len(node.args[0]) >= 2
                and all(value is node.args[0][0] for value in node.args[0])
            )
        )
        and (filter_fn is None or filter_fn(node))
    ]
    cat_nodes = [
        partition.output_nodes[0] for partition in cat_partitions
    ] + fallback_cat_nodes
    annotated_partitions = []
    for cat_node in cat_nodes:
        if _is_annotated([cat_node]):
            continue

        if cat_node.target != torch.ops.aten.cat.default:
            # TODO: change this to AnnotationException
            raise Exception(
                f"Expected cat node: torch.ops.aten.cat.default, but found {cat_node.target}"
                " please check if you are calling the correct capture API"
            )

        annotated_partitions.append([cat_node])

        input_act_qspec = get_input_act_qspec(quantization_config)
        inputs = cat_node.args[0]

        # An inclusive tree scan shifts a tensor with an exact identity prefix:
        #   cat((zeros_like(x[:k]), x[:-k]))
        #   cat((ones_like(x[:k]),  x[:-k]))
        # This is a pure integer layout operation, not a numerical boundary.
        # Requantizing the identity prefix, shifted payload, and Cat output on
        # three independently observed grids adds avoidable error at every
        # prefix level. Share the payload grid across all three instead. The
        # rule is deliberately narrow so ordinary multi-source concatenations
        # retain independent input grids and an explicit output requantization.
        identity_padding_concat = False
        identity_padding_reference = None

        def ensure_concrete_output_qspec(reference: Node) -> None:
            """Give a shared-grid reference a union-find root in PT2E."""
            annotation = reference.meta.get("quantization_annotation")
            if annotation is None:
                annotation = QuantizationAnnotation()
                reference.meta["quantization_annotation"] = annotation
            if annotation.output_qspec is None:
                annotation.output_qspec = input_act_qspec
            annotation._annotated = True

        if len(inputs) == 2 and all(isinstance(value, Node) for value in inputs):
            identity_padding_targets = {
                torch.ops.aten.zeros_like.default,
                torch.ops.aten.ones_like.default,
            }
            if inputs[0].target in identity_padding_targets:
                identity_padding_concat = True
                identity_padding_reference = inputs[1]
                # A payload producer can already be marked annotated while
                # carrying only input qspecs.  Whole-sequence prefix scans
                # expose this on their first level when the payload is an
                # Einsum result.  Sharing the Cat with such a node used to
                # leave PT2E's union-find pointing at a node with no concrete
                # observer entry.  Materialize (or complete) the payload's
                # output annotation unconditionally so it is a valid shared
                # grid root.
                ensure_concrete_output_qspec(identity_padding_reference)

        repeated_input_concat = bool(inputs) and all(
            input_act is inputs[0] for input_act in inputs
        )
        if repeated_input_concat:
            # This also covers a one-input Cat emitted by an eager Python loop.
            # The vacuously repeated payload may be an otherwise-unannotated
            # Einsum result, so make it a concrete root before sharing.
            ensure_concrete_output_qspec(inputs[0])

        input_qspec_map = {}
        for input_act in inputs:
            if _is_annotated([input_act]):
                continue
            input_qspec_map[input_act] = (
                SharedQuantizationSpec(identity_padding_reference)
                if identity_padding_concat
                and input_act is not identity_padding_reference
                else input_act_qspec
            )

        # When every list entry is the same tensor, concatenation is only a
        # layout/public-ABI operation.  Sharing its output grid with that
        # tensor makes QAT and export model the exact integer-code repeat,
        # rather than learning a gratuitous terminal requantization.
        if identity_padding_concat:
            output_act_qspec = SharedQuantizationSpec(identity_padding_reference)
        elif repeated_input_concat:
            output_act_qspec = SharedQuantizationSpec(inputs[0])
        else:
            output_act_qspec = get_output_act_qspec(quantization_config)

        cat_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=output_act_qspec,
            _annotated=True,
        )
    return annotated_partitions
