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

from typing import Any, Callable, Dict, List, Optional, Set

import torch
import torch.nn.functional as F
from torch.ao.quantization.fake_quantize import (
    FakeQuantize,
)
from torch.ao.quantization.observer import (
    HistogramObserver,
    MinMaxObserver,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
    PlaceholderObserver,
)

from torch.ao.quantization.qconfig import _ObserverOrFakeQuantizeConstructor

from torch.ao.quantization.quantizer import (
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
    _get_linear_patterns,
    _get_module_name_filter,
    _get_not_module_type_or_name_filter,
)
from torch.ao.quantization.pt2e.utils import _get_aten_graph_module_for_pattern
from torch.fx.passes.utils.matcher_with_name_node_map_utils import (
    SubgraphMatcherWithNameNodeMap,
)

from torch.fx import Node
from torch.ao.quantization.pt2e.graph_utils import find_sequential_partitions


# from torch.ops.quantized_decomposed import quantize_per_tensor


_conv1d_bn_example_inputs = (
    torch.randn(1, 1, 3),
    torch.randn(1, 1, 1),
    torch.randn(1),
    torch.randn(1),
    torch.randn(1),
    torch.randn(1),
    torch.randn(1),
)
_conv2d_bn_example_inputs = (
    torch.randn(1, 1, 3, 3),
    torch.randn(1, 1, 1, 1),
    torch.randn(1),
    torch.randn(1),
    torch.randn(1),
    torch.randn(1),
    torch.randn(1),
)


__all__ = [
    "SimaFakeQuantize",
    "SimaQuantizer",
    "get_sima_quantization_config",
]


class SimaFakeQuantize(FakeQuantize):
    """Fake quantizer whose frozen buffers are authoritative for conversion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("sima_qparams_frozen", torch.tensor([False], dtype=torch.bool))

    def calculate_qparams(self):
        if bool(self.sima_qparams_frozen.item()):
            return self.scale, self.zero_point
        return super().calculate_qparams()

    def freeze_qparams(self) -> None:
        self.sima_qparams_frozen.fill_(True)


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
        "matmul": [
            [torch.matmul],
            [operator.matmul],
            [torch.mm],
            [torch.bmm],
            [torch.baddbmm],
        ],
        "softmax": [[torch.nn.Softmax], [F.softmax], [torch.softmax]],
        "layer_norm": [[torch.nn.LayerNorm], [F.layer_norm]],
        "erf": [[torch.erf]],
        "gelu": [[torch.nn.GELU], [F.gelu]],
        "cat": [[torch.cat]],
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
    ]:
        ops = _supported_symmetric_quantized_operators()
        for pattern_list in ops.values():
            supported_config_and_operators.append(
                OperatorConfig(quantization_config, pattern_list)
            )
    return copy.deepcopy(supported_config_and_operators)


@functools.lru_cache
def get_sima_quantization_config(
    is_qat: bool = False,
):
    # This configuration function selects QAT or PTQ. Weight fake quantization
    # is always enabled for QAT so training matches SiMa shift requantization.
    # Sima has a preferred encoding for activation and weight tensors that give 
    # best possible results. Since QAT is a high-effort activity, we only use the
    # best quantization settings possible here.
    #

    # Activations
    # ---------------------------------------------------
    act_extra_args: Dict[str, Any] = {"eps": 2**-12}
    if is_qat:
        act_observer_or_fake_quant_ctr = SimaFakeQuantize
        act_extra_args["observer"] = SimaMovingAverageMinMaxObserver
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
    wt_extra_args: Dict[str, Any] = {"eps": 2**-12}
    if is_qat:
        weight_observer_or_fake_quant_ctr = SimaFakeQuantize
        wt_extra_args["observer"] = MovingAveragePerChannelMinMaxObserver
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
        "sima_matmul",
        "sima_softmax",
        "sima_layer_norm",
        "sima_erf",
        "sima_gelu",
        "sima_unary_int8",
        "sima_binary_int8",
        "sima_einsum",
        "sima_pow",
        "sima_prelu",
        "sima_reduction",
        "sima_global_max_pool2d",
        "sima_mixed_output",
        "sima_conv_add_or_mul_const",
        "sima_conv_hardtanh",
        "conv_relu",
        "conv",
        # The stock source-partition annotator misses later calls when one
        # Conv2d module/weight is reused. Finish those invocations after the
        # normal Conv and fusion annotators have run.
        "sima_unannotated_conv2d",
        "sima_grid_preserving",
        "sima_split",
        "adaptive_avg_pool2d",
        "sima_add_hardtanh",
        "add_relu",
        "add",
        "mul_relu",
        "mul",
        "sima_cat",
        "sima_sigmoid",
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
        propagate_annotation(model)
        return model

    def _annotate_all_static_patterns(
        self,
        model: torch.fx.GraphModule,
        quantization_config: Optional[QuantizationConfig],
        filter_fn: Optional[Callable[[Node], bool]] = None,
    ) -> torch.fx.GraphModule:
        # TODO: implement the support for None to be canceling out previous annotations
        if quantization_config is None:
            return model

        ops = []
        if quantization_config.is_qat:
            ops += self.STATIC_QAT_ONLY_OPS
        ops += self.STATIC_OPS
        for op in ops:
            OP_TO_ANNOTATOR[op](model, quantization_config, filter_fn)
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
        unsupported_w8a8 = {
            torch.ops.aten.conv_transpose2d.input: (
                "ConvTranspose2d needs an output-channel weight qspec; the current "
                "per-channel axis-0 QAT contract is not correct for ONNX ConvTranspose"
            ),
            torch.ops.aten.embedding.default: (
                "Embedding/Gather is not supported by the compiler's INT8 contract"
            ),
            torch.ops.aten.grid_sampler_2d.default: (
                "GridSample is not supported by the compiler's INT8 contract"
            ),
            torch.ops.aten.grid_sampler.default: (
                "GridSample is not supported by the compiler's INT8 contract"
            ),
            torch.ops.aten.amin.default: (
                "ReduceMin is not supported by the compiler's INT8 contract"
            ),
            torch.ops.aten.cumsum.default: (
                "CumSum is not supported by the compiler's INT8 contract"
            ),
        }
        for node in model.graph.nodes:
            if node.op != "call_function":
                continue
            if node.target in unsupported_w8a8:
                raise ValueError(f"SiMa INT8 QAT: {unsupported_w8a8[node.target]}")
            if node.target == torch.ops.aten.gelu.default:
                approximate = _node_argument(node, 1, "approximate", "none")
                if approximate != "none":
                    raise ValueError(
                        "SiMa INT8 QAT supports only exact GELU "
                        "(approximate='none'); tanh GELU is not an AFE-supported contract"
                    )

    @classmethod
    def get_supported_operators(cls) -> List[OperatorConfig]:
        return cls.supported_config_and_operators


def _annotate_single_aten_op(
    gm: torch.fx.GraphModule,
    quantization_config: QuantizationConfig,
    targets: tuple[object, ...],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> List[List[Node]]:
    """Annotate unary ATen nodes without relying on source-partition metadata."""
    annotated_partitions = []
    for op_node in gm.graph.nodes:
        if op_node.op != "call_function" or op_node.target not in targets:
            continue
        if _is_annotated([op_node]) or (filter_fn and not filter_fn(op_node)):
            continue

        input_qspec_map = {}
        input_act = op_node.args[0]
        if isinstance(input_act, Node):
            input_qspec_map[input_act] = get_input_act_qspec(quantization_config)
        op_node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=get_output_act_qspec(quantization_config),
            _annotated=True,
        )
        annotated_partitions.append([op_node])
    return annotated_partitions


def _annotate_tensor_inputs(
    node: Node,
    inputs: List[Node],
    quantization_config: QuantizationConfig,
    *,
    output_qspec: bool = True,
) -> None:
    """Annotate explicit floating tensor operands and an optional float output."""
    input_qspec = get_input_act_qspec(quantization_config)
    node.meta["quantization_annotation"] = QuantizationAnnotation(
        input_qspec_map={value: input_qspec for value in inputs},
        output_qspec=(
            get_output_act_qspec(quantization_config) if output_qspec else None
        ),
        _annotated=True,
    )


@register_annotator("sima_unary_int8")
def _sima_annotate_unary_int8(
    gm: torch.fx.GraphModule,
    quantization_config: QuantizationConfig,
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> List[List[Node]]:
    """Annotate compiler-supported unary activation and normalization kernels."""
    return _annotate_single_aten_op(
        gm,
        quantization_config,
        (
            torch.ops.aten.abs.default,
            torch.ops.aten.elu.default,
            torch.ops.aten.exp.default,
            torch.ops.aten.hardsigmoid.default,
            torch.ops.aten.hardswish.default,
            torch.ops.aten.instance_norm.default,
            torch.ops.aten.leaky_relu.default,
            torch.ops.aten.log.default,
            torch.ops.aten.log_softmax.int,
            torch.ops.aten.neg.default,
            torch.ops.aten.reciprocal.default,
            torch.ops.aten.softplus.default,
            torch.ops.aten.sqrt.default,
            torch.ops.aten.tanh.default,
            torch.ops.aten.upsample_nearest2d.vec,
            torch.ops.aten.upsample_bilinear2d.vec,
        ),
        filter_fn,
    )


@register_annotator("sima_binary_int8")
def _sima_annotate_binary_int8(
    gm: torch.fx.GraphModule,
    quantization_config: QuantizationConfig,
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> List[List[Node]]:
    """Annotate supported two-tensor arithmetic without quantizing scalar metadata."""
    targets = {torch.ops.aten.div.Tensor, torch.ops.aten.sub.Tensor}
    annotated = []
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target not in targets:
            continue
        if _is_annotated([node]) or (filter_fn and not filter_fn(node)):
            continue
        inputs = [value for value in node.args[:2] if isinstance(value, Node)]
        if not inputs or any(_is_input_non_float_tensor(value) for value in inputs):
            continue
        _annotate_tensor_inputs(node, inputs, quantization_config)
        annotated.append([node])
    return annotated


@register_annotator("sima_einsum")
def _sima_annotate_einsum(
    gm: torch.fx.GraphModule,
    quantization_config: QuantizationConfig,
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> List[List[Node]]:
    annotated = []
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target != torch.ops.aten.einsum.default:
            continue
        if _is_annotated([node]) or (filter_fn and not filter_fn(node)):
            continue
        operands = node.args[1] if len(node.args) > 1 else ()
        inputs = [value for value in operands if isinstance(value, Node)]
        if len(inputs) != 2 or any(_is_input_non_float_tensor(value) for value in inputs):
            continue
        _annotate_tensor_inputs(node, inputs, quantization_config)
        annotated.append([node])
    return annotated


@register_annotator("sima_pow")
def _sima_annotate_pow(
    gm: torch.fx.GraphModule,
    quantization_config: QuantizationConfig,
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> List[List[Node]]:
    """Quantize the Pow base while preserving its required scalar exponent."""
    return _annotate_single_aten_op(
        gm, quantization_config, (torch.ops.aten.pow.Tensor_Scalar,), filter_fn
    )


@register_annotator("sima_prelu")
def _sima_annotate_prelu(
    gm: torch.fx.GraphModule,
    quantization_config: QuantizationConfig,
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> List[List[Node]]:
    annotated = []
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target != torch.ops.aten.prelu.default:
            continue
        if _is_annotated([node]) or (filter_fn and not filter_fn(node)):
            continue
        activation, slope = node.args[:2]
        if not isinstance(activation, Node) or not isinstance(slope, Node):
            continue
        node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map={
                activation: get_input_act_qspec(quantization_config),
                # PReLU alpha is a 1D compiler operand rather than a Conv/Linear
                # weight tensor. Per-channel weight observers require rank >= 2.
                slope: get_input_act_qspec(quantization_config),
            },
            output_qspec=get_output_act_qspec(quantization_config),
            _annotated=True,
        )
        annotated.append([node])
    return annotated


@register_annotator("sima_reduction")
def _sima_annotate_reduction(
    gm: torch.fx.GraphModule,
    quantization_config: QuantizationConfig,
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> List[List[Node]]:
    """Use independent output grids for reductions that change value ranges."""
    return _annotate_single_aten_op(
        gm,
        quantization_config,
        (
            torch.ops.aten.amax.default,
            torch.ops.aten.linalg_vector_norm.default,
            torch.ops.aten.logsumexp.default,
            torch.ops.aten.mean.dim,
            torch.ops.aten.sum.dim_IntList,
        ),
        filter_fn,
    )


@register_annotator("sima_global_max_pool2d")
def _sima_annotate_global_max_pool2d(
    gm: torch.fx.GraphModule,
    quantization_config: QuantizationConfig,
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> List[List[Node]]:
    """Annotate only the value output of adaptive/global max pooling."""
    annotated = []
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target != torch.ops.aten.adaptive_max_pool2d.default:
            continue
        if _is_annotated([node]) or (filter_fn and not filter_fn(node)):
            continue
        input_node = node.args[0]
        if not isinstance(input_node, Node):
            continue
        _annotate_tensor_inputs(node, [input_node], quantization_config, output_qspec=False)
        value_users = [
            user
            for user in node.users
            if user.op == "call_function"
            and user.target is operator.getitem
            and len(user.args) > 1
            and user.args[1] == 0
        ]
        for value_node in value_users:
            value_node.meta["quantization_annotation"] = QuantizationAnnotation(
                output_qspec=get_output_act_qspec(quantization_config),
                _annotated=True,
            )
        annotated.append([node, *value_users])
    return annotated


@register_annotator("sima_mixed_output")
def _sima_annotate_mixed_output(
    gm: torch.fx.GraphModule,
    quantization_config: QuantizationConfig,
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> List[List[Node]]:
    """Quantize float inputs/values while preserving integer result tensors."""
    annotated = []
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target not in {
            torch.ops.aten.argmax.default,
            torch.ops.aten.topk.default,
        }:
            continue
        if _is_annotated([node]) or (filter_fn and not filter_fn(node)):
            continue
        input_node = node.args[0]
        if not isinstance(input_node, Node):
            continue
        _annotate_tensor_inputs(node, [input_node], quantization_config, output_qspec=False)
        partition = [node]
        if node.target == torch.ops.aten.topk.default:
            value_users = [
                user
                for user in node.users
                if user.op == "call_function"
                and user.target is operator.getitem
                and len(user.args) > 1
                and user.args[1] == 0
            ]
            for value_node in value_users:
                value_node.meta["quantization_annotation"] = QuantizationAnnotation(
                    output_qspec=get_output_act_qspec(quantization_config),
                    _annotated=True,
                )
            partition.extend(value_users)
        annotated.append(partition)
    return annotated


@register_annotator("sima_grid_preserving")
def _sima_annotate_grid_preserving(
    gm: torch.fx.GraphModule,
    quantization_config: QuantizationConfig,
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> List[List[Node]]:
    """Share an existing activation grid across value-preserving layout operators."""
    targets = {
        torch.ops.aten.expand.default,
        torch.ops.aten.flatten.using_ints,
        torch.ops.aten.pad.default,
        torch.ops.aten.permute.default,
        torch.ops.aten.pixel_shuffle.default,
        torch.ops.aten.pixel_unshuffle.default,
        torch.ops.aten.repeat.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten.tile.default,
        torch.ops.aten.transpose.int,
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.view.default,
    }
    annotated = []
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target not in targets:
            continue
        if _is_annotated([node]) or (filter_fn and not filter_fn(node)):
            continue
        input_node = node.args[0]
        if not isinstance(input_node, Node):
            continue
        source_annotation = input_node.meta.get("quantization_annotation")
        if source_annotation is None or source_annotation.output_qspec is None:
            continue
        shared_qspec = SharedQuantizationSpec(input_node)
        node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map={input_node: shared_qspec},
            output_qspec=shared_qspec,
            _annotated=True,
        )
        annotated.append([node])
    return annotated


@register_annotator("sima_split")
def _sima_annotate_split(
    gm: torch.fx.GraphModule,
    quantization_config: QuantizationConfig,
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> List[List[Node]]:
    """Propagate the input grid independently to every tensor-valued split output."""
    annotated = []
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target not in {
            torch.ops.aten.split.Tensor,
            torch.ops.aten.split_with_sizes.default,
        }:
            continue
        if _is_annotated([node]) or (filter_fn and not filter_fn(node)):
            continue
        input_node = node.args[0]
        if not isinstance(input_node, Node):
            continue
        source_annotation = input_node.meta.get("quantization_annotation")
        if source_annotation is None or source_annotation.output_qspec is None:
            continue
        shared_qspec = SharedQuantizationSpec(input_node)
        node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map={input_node: shared_qspec},
            _annotated=True,
        )
        outputs = [
            user
            for user in node.users
            if user.op == "call_function" and user.target is operator.getitem
        ]
        for output in outputs:
            output.meta["quantization_annotation"] = QuantizationAnnotation(
                output_qspec=shared_qspec,
                _annotated=True,
            )
        annotated.append([node, *outputs])
    return annotated


def _node_argument(node: Node, position: int, name: str, default: Any) -> Any:
    """Read an ATen argument independent of positional/keyword capture form."""
    if name in node.kwargs:
        return node.kwargs[name]
    if len(node.args) > position:
        return node.args[position]
    return default


@register_annotator("sima_unannotated_conv2d")
def _sima_annotate_unannotated_conv2d(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    """Annotate Conv2d calls missed because multiple calls share one weight."""
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
        # The shared weight is not itself an invocation. Mark only this Conv so
        # every later call can receive its own edge annotations.
        _mark_nodes_as_annotated([node])
        annotated_partitions.append([node])
    return annotated_partitions


@register_annotator("sima_matmul")
def _sima_annotate_matmul(
    gm: torch.fx.GraphModule,
    quantization_config: QuantizationConfig,
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> List[List[Node]]:
    """Annotate activation MatMul/MM/BMM and fused BAddBMM boundaries."""
    targets = {
        torch.ops.aten.matmul.default,
        torch.ops.aten.mm.default,
        torch.ops.aten.bmm.default,
        torch.ops.aten.baddbmm.default,
    }
    annotated_partitions: List[List[Node]] = []
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target not in targets:
            continue
        if _is_annotated([node]) or (filter_fn and not filter_fn(node)):
            continue
        inputs = node.args[:3] if node.target == torch.ops.aten.baddbmm.default else node.args[:2]
        input_nodes = [value for value in inputs if isinstance(value, Node)]
        if len(input_nodes) != len(inputs) or any(
            _is_input_non_float_tensor(value) for value in input_nodes
        ):
            continue
        input_qspec = get_input_act_qspec(quantization_config)
        node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map={value: input_qspec for value in input_nodes},
            output_qspec=get_output_act_qspec(quantization_config),
            _annotated=True,
        )
        annotated_partitions.append([node])
    return annotated_partitions


@register_annotator("sima_softmax")
def _sima_annotate_softmax(
    gm: torch.fx.GraphModule,
    quantization_config: QuantizationConfig,
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> List[List[Node]]:
    # AFE automatic layout conversion maps the selected logical axis to the
    # MLA channel dimension, so QAT should not reject a source-level axis.
    return _annotate_single_aten_op(
        gm,
        quantization_config,
        (torch.ops.aten.softmax.int, torch.ops.aten._softmax.default),
        filter_fn,
    )


@register_annotator("sima_layer_norm")
def _sima_annotate_layer_norm(
    gm: torch.fx.GraphModule,
    quantization_config: QuantizationConfig,
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> List[List[Node]]:
    # Scale and bias remain floating constants. AFE quantizes/folds them when
    # the supported LayerNorm composite is selected.
    return _annotate_single_aten_op(
        gm,
        quantization_config,
        (torch.ops.aten.layer_norm.default,),
        filter_fn,
    )


@register_annotator("sima_erf")
def _sima_annotate_erf(
    gm: torch.fx.GraphModule,
    quantization_config: QuantizationConfig,
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> List[List[Node]]:
    return _annotate_single_aten_op(
        gm,
        quantization_config,
        (torch.ops.aten.erf.default,),
        filter_fn,
    )


@register_annotator("sima_gelu")
def _sima_annotate_gelu(
    gm: torch.fx.GraphModule,
    quantization_config: QuantizationConfig,
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> List[List[Node]]:
    annotated_partitions: List[List[Node]] = []
    for node in gm.graph.nodes:
        if node.op != "call_function" or node.target != torch.ops.aten.gelu.default:
            continue
        if _node_argument(node, 1, "approximate", "none") != "none":
            continue
        if _is_annotated([node]) or (filter_fn and not filter_fn(node)):
            continue
        input_node = node.args[0]
        if not isinstance(input_node, Node) or _is_input_non_float_tensor(input_node):
            continue
        node.meta["quantization_annotation"] = QuantizationAnnotation(
            input_qspec_map={input_node: get_input_act_qspec(quantization_config)},
            output_qspec=get_output_act_qspec(quantization_config),
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
    return _annotate_single_aten_op(
        gm,
        quantization_config,
        (torch.ops.aten.sigmoid.default,),
        filter_fn,
    )


@register_annotator("sima_silu")
def _sima_annotate_silu(
    gm: torch.fx.GraphModule,
    quantization_config: Optional[QuantizationConfig],
    filter_fn: Optional[Callable[[Node], bool]] = None,
) -> Optional[List[List[Node]]]:
    return _annotate_single_aten_op(
        gm,
        quantization_config,
        (torch.ops.aten.silu.default, torch.ops.aten.silu_.default),
        filter_fn,
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
        pattern = _get_aten_graph_module_for_pattern(pattern, example_inputs, is_cuda)
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
        if n.op == "call_function" and n.target == torch.ops.aten.batch_norm.default:
            if _is_annotated([n]) or (filter_fn and not filter_fn(n)):
                continue
            input_qspec_map = {}
            input_act = n.args[0]
            if isinstance(input_act, Node):
                input_qspec_map[input_act] = get_input_act_qspec(quantization_config)
            n.meta["quantization_annotation"] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=get_output_act_qspec(quantization_config),
                _annotated=True,
            )
            annotated_partitions.append([n])
            continue
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
    annotated_partitions = []
    for cat_node in gm.graph.nodes:
        if cat_node.op != "call_function" or cat_node.target != torch.ops.aten.cat.default:
            continue
        if _is_annotated([cat_node]) or (filter_fn and not filter_fn(cat_node)):
            continue
        annotated_partitions.append([cat_node])

        input_act_qspec = get_input_act_qspec(quantization_config)
        inputs = cat_node.args[0]

        def ensure_concrete_output_qspec(reference: Node) -> None:
            """Give a shared-grid reference a concrete PT2E observer root."""
            annotation = reference.meta.get("quantization_annotation")
            if annotation is None:
                annotation = QuantizationAnnotation()
                reference.meta["quantization_annotation"] = annotation
            if annotation.output_qspec is None:
                annotation.output_qspec = input_act_qspec
            annotation._annotated = True

        identity_padding_reference = None
        if len(inputs) == 2 and all(isinstance(value, Node) for value in inputs):
            identity_padding_targets = {
                torch.ops.aten.zeros_like.default,
                torch.ops.aten.ones_like.default,
            }
            if inputs[0].target in identity_padding_targets:
                identity_padding_reference = inputs[1]
                ensure_concrete_output_qspec(identity_padding_reference)

        repeated_input_concat = bool(inputs) and all(
            input_act is inputs[0] for input_act in inputs
        )
        if repeated_input_concat:
            ensure_concrete_output_qspec(inputs[0])

        input_qspec_map = {}
        for input_act in inputs:
            if _is_annotated([input_act]):
                continue
            input_qspec_map[input_act] = (
                SharedQuantizationSpec(identity_padding_reference)
                if identity_padding_reference is not None
                and input_act is not identity_padding_reference
                else input_act_qspec
            )

        # Repeated-input and identity-prefix concatenations are integer layout
        # operations. Sharing the payload grid avoids a gratuitous requantize
        # and matches AFE's concat grid contract.
        if identity_padding_reference is not None:
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
