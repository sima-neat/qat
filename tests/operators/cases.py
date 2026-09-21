"""Small model factories covering every active SiMa QAT operator pattern."""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor, nn

from sima_qat.operator_manifest import ONNX_TEST_CASE_IDS


InputFactory = Callable[[], tuple[Tensor, ...]]
ModelFactory = Callable[[], nn.Module]


@dataclass(frozen=True)
class OperatorCase:
    name: str
    pattern: str
    model_factory: ModelFactory
    input_factory: InputFactory
    annotation_targets: tuple[object, ...]
    weighted: bool = False
    onnx_family: str | None = None


class Conv1dModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv1d(3, 4, 3, padding=1)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


class Conv2dModel(nn.Module):
    def __init__(self, groups: int = 1, bias: bool = True) -> None:
        super().__init__()
        channels = 4 if groups > 1 else 3
        self.conv = nn.Conv2d(channels, 4, 3, padding=1, groups=groups, bias=bias)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


class ReusedConv2dModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Conv2d(3, 3, 1, bias=False)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.shared(torch.relu(self.shared(inputs)))


class ConvReluModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, inputs: Tensor) -> Tensor:
        return self.relu(self.conv(inputs))


class ConvHardtanhModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.activation = nn.Hardtanh()

    def forward(self, inputs: Tensor) -> Tensor:
        return self.activation(self.conv(inputs))


class ConvBnModel(nn.Module):
    def __init__(self, activation: nn.Module | None = None) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.bn = nn.BatchNorm2d(4)
        self.activation = activation

    def forward(self, inputs: Tensor) -> Tensor:
        output = self.bn(self.conv(inputs))
        return output if self.activation is None else self.activation(output)


class LinearModel(nn.Module):
    def __init__(self, relu: bool = False) -> None:
        super().__init__()
        self.linear = nn.Linear(8, 4)
        self.activation = nn.ReLU() if relu else nn.Identity()

    def forward(self, inputs: Tensor) -> Tensor:
        return self.activation(self.linear(inputs))


class MatMulModel(nn.Module):
    def __init__(self, operation: str) -> None:
        super().__init__()
        self.operation = operation

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        if self.operation == "mm":
            return torch.mm(left, right)
        if self.operation == "bmm":
            return torch.bmm(left, right)
        return torch.matmul(left, right)


class BAddBMMModel(nn.Module):
    def forward(self, bias: Tensor, left: Tensor, right: Tensor) -> Tensor:
        return torch.baddbmm(bias, left, right)


class SoftmaxModel(nn.Module):
    def forward(self, inputs: Tensor) -> Tensor:
        return torch.softmax(inputs, dim=1)


class LayerNormModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(8)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.norm(inputs)


class ErfModel(nn.Module):
    def forward(self, inputs: Tensor) -> Tensor:
        return torch.erf(inputs)


class ExactGeluModel(nn.Module):
    def forward(self, inputs: Tensor) -> Tensor:
        return torch.nn.functional.gelu(inputs, approximate="none")


class DecomposedGeluModel(nn.Module):
    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * 0.5 * (1.0 + torch.erf(inputs / math.sqrt(2.0)))


class ConvConstantPostOpModel(nn.Module):
    def __init__(self, operation: str) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)
        self.operation = operation
        self.register_buffer("constant", torch.linspace(0.5, 1.0, 4).reshape(1, 4, 1, 1))

    def forward(self, inputs: Tensor) -> Tensor:
        output = self.conv(inputs)
        if self.operation == "add":
            return output + self.constant
        return output * self.constant


class BinaryModel(nn.Module):
    def __init__(self, operation: str, activation: nn.Module | None = None) -> None:
        super().__init__()
        self.left = nn.Conv2d(3, 4, 1)
        self.right = nn.Conv2d(3, 4, 1)
        self.operation = operation
        self.activation = activation

    def forward(self, inputs: Tensor) -> Tensor:
        left = self.left(inputs)
        right = self.right(inputs)
        output = left + right if self.operation == "add" else left * right
        return output if self.activation is None else self.activation(output)


class ActivationModel(nn.Module):
    def __init__(self, activation: nn.Module) -> None:
        super().__init__()
        self.input_conv = nn.Conv2d(3, 4, 1)
        self.activation = activation
        self.output_conv = nn.Conv2d(4, 4, 1)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.output_conv(self.activation(self.input_conv(inputs)))


class PoolModel(nn.Module):
    def __init__(self, pool: nn.Module) -> None:
        super().__init__()
        self.input_conv = nn.Conv2d(3, 4, 1)
        self.pool = pool
        self.output_conv = nn.Conv2d(4, 4, 1)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.output_conv(self.pool(self.input_conv(inputs)))


class CatModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = nn.Conv2d(3, 4, 1)
        self.right = nn.Conv2d(3, 4, 1)
        self.output = nn.Conv2d(8, 4, 1)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.output(torch.cat((self.left(inputs), self.right(inputs)), dim=1))


class SliceSelectUnsqueezeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1)

    def forward(self, inputs: Tensor) -> Tensor:
        channels = [torch.unsqueeze(inputs[:, index], 1) for index in range(3)]
        return self.conv(torch.cat(channels, dim=1))


class BatchNormModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(4)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.bn(inputs)


class UnaryOperatorModel(nn.Module):
    def __init__(self, operation: Callable[[Tensor], Tensor]) -> None:
        super().__init__()
        self.operation = operation

    def forward(self, inputs: Tensor) -> Tensor:
        return self.operation(inputs)


class InstanceNormModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.operation = nn.InstanceNorm2d(3, track_running_stats=False)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.operation(inputs)


class DirectBinaryModel(nn.Module):
    def __init__(self, operation: Callable[[Tensor, Tensor], Tensor]) -> None:
        super().__init__()
        self.operation = operation

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        return self.operation(left, right)


class EinsumModel(nn.Module):
    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        return torch.einsum("bik,bkj->bij", left, right)


class PReluModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.activation = nn.PReLU(3)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.activation(inputs)


class LayoutAfterConvModel(nn.Module):
    def __init__(self, operation: Callable[[Tensor], Tensor]) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)
        self.operation = operation

    def forward(self, inputs: Tensor) -> Tensor:
        return self.operation(self.conv(inputs))


class ExpandAfterConvModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs).expand(-1, -1, 4, -1)


class SplitAfterConvModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)

    def forward(self, inputs: Tensor) -> Tensor:
        return torch.split(self.conv(inputs), 2, dim=1)[0]


class TopKValuesModel(nn.Module):
    def forward(self, inputs: Tensor) -> Tensor:
        return torch.topk(inputs, 3, dim=1).values


IMAGE_INPUT: InputFactory = lambda: (torch.randn(2, 3, 8, 8),)
FOUR_CHANNEL_INPUT: InputFactory = lambda: (torch.randn(2, 4, 8, 8),)
SEQUENCE_INPUT: InputFactory = lambda: (torch.randn(2, 3, 12),)
LINEAR_INPUT: InputFactory = lambda: (torch.randn(2, 8),)
ATTENTION_INPUT: InputFactory = lambda: (torch.randn(1, 4, 8),)
MM_INPUT: InputFactory = lambda: (torch.randn(4, 8), torch.randn(8, 3))
BMM_INPUT: InputFactory = lambda: (torch.randn(2, 4, 8), torch.randn(2, 8, 3))
BADDBMM_INPUT: InputFactory = lambda: (
    torch.randn(2, 4, 3),
    torch.randn(2, 4, 8),
    torch.randn(2, 8, 3),
)
POSITIVE_IMAGE_INPUT: InputFactory = lambda: (torch.rand(2, 3, 8, 8) + 0.25,)
PAIR_IMAGE_INPUT: InputFactory = lambda: (
    torch.randn(2, 3, 8, 8),
    torch.rand(2, 3, 8, 8) + 0.5,
)
EINSUM_INPUT: InputFactory = lambda: (
    torch.randn(2, 3, 4),
    torch.randn(2, 4, 5),
)
EXPAND_INPUT: InputFactory = lambda: (torch.randn(2, 3, 1, 8),)


OPERATOR_CASES = (
    OperatorCase(
        "conv1d",
        "conv",
        Conv1dModel,
        SEQUENCE_INPUT,
        (torch.ops.aten.conv1d.default,),
        weighted=True,
    ),
    OperatorCase(
        "conv2d",
        "conv",
        Conv2dModel,
        IMAGE_INPUT,
        (torch.ops.aten.conv2d.default,),
        weighted=True,
        onnx_family="conv",
    ),
    OperatorCase(
        "reused_conv2d",
        "sima_unannotated_conv2d",
        ReusedConv2dModel,
        IMAGE_INPUT,
        (torch.ops.aten.conv2d.default,),
        weighted=True,
    ),
    OperatorCase(
        "conv_relu",
        "conv_relu",
        ConvReluModel,
        IMAGE_INPUT,
        (torch.ops.aten.relu.default,),
        weighted=True,
    ),
    OperatorCase(
        "conv_hardtanh",
        "sima_conv_hardtanh",
        ConvHardtanhModel,
        IMAGE_INPUT,
        (torch.ops.aten.hardtanh.default, torch.ops.aten.hardtanh_.default),
        weighted=True,
    ),
    OperatorCase(
        "conv_bn",
        "conv_bn",
        ConvBnModel,
        IMAGE_INPUT,
        (torch.ops.aten.conv2d.default,),
        weighted=True,
    ),
    OperatorCase(
        "conv_bn_relu",
        "conv_bn_relu",
        lambda: ConvBnModel(nn.ReLU()),
        IMAGE_INPUT,
        (torch.ops.aten.relu.default,),
        weighted=True,
        onnx_family="conv_bn",
    ),
    OperatorCase(
        "conv_bn_hardtanh",
        "sima_conv_bn_hardtanh",
        lambda: ConvBnModel(nn.Hardtanh()),
        IMAGE_INPUT,
        (torch.ops.aten.hardtanh.default, torch.ops.aten.hardtanh_.default),
        weighted=True,
    ),
    OperatorCase(
        "linear",
        "linear",
        LinearModel,
        LINEAR_INPUT,
        (torch.ops.aten.linear.default,),
        weighted=True,
        onnx_family="linear",
    ),
    OperatorCase(
        "linear_relu",
        "linear_relu",
        lambda: LinearModel(relu=True),
        LINEAR_INPUT,
        (torch.ops.aten.relu.default,),
        weighted=True,
    ),
    OperatorCase(
        "mm",
        "sima_matmul",
        lambda: MatMulModel("mm"),
        MM_INPUT,
        (torch.ops.aten.mm.default,),
        onnx_family="matmul",
    ),
    OperatorCase(
        "matmul",
        "sima_matmul",
        lambda: MatMulModel("matmul"),
        BMM_INPUT,
        (torch.ops.aten.matmul.default,),
    ),
    OperatorCase(
        "bmm",
        "sima_matmul",
        lambda: MatMulModel("bmm"),
        BMM_INPUT,
        (torch.ops.aten.bmm.default,),
    ),
    OperatorCase(
        "baddbmm",
        "sima_matmul",
        BAddBMMModel,
        BADDBMM_INPUT,
        (torch.ops.aten.baddbmm.default,),
    ),
    OperatorCase(
        "softmax",
        "sima_softmax",
        SoftmaxModel,
        ATTENTION_INPUT,
        (torch.ops.aten.softmax.int, torch.ops.aten._softmax.default),
        onnx_family="softmax",
    ),
    OperatorCase(
        "layer_norm",
        "sima_layer_norm",
        LayerNormModel,
        ATTENTION_INPUT,
        (torch.ops.aten.layer_norm.default,),
        onnx_family="layer_norm",
    ),
    OperatorCase(
        "erf",
        "sima_erf",
        ErfModel,
        ATTENTION_INPUT,
        (torch.ops.aten.erf.default,),
        onnx_family="erf",
    ),
    OperatorCase(
        "decomposed_gelu",
        "sima_erf",
        DecomposedGeluModel,
        ATTENTION_INPUT,
        (torch.ops.aten.erf.default,),
    ),
    OperatorCase(
        "exact_gelu",
        "sima_gelu",
        ExactGeluModel,
        ATTENTION_INPUT,
        (torch.ops.aten.gelu.default,),
        onnx_family="gelu",
    ),
    OperatorCase(
        "conv_add_constant",
        "sima_conv_add_or_mul_const",
        lambda: ConvConstantPostOpModel("add"),
        IMAGE_INPUT,
        (torch.ops.aten.add.Tensor,),
        weighted=True,
    ),
    OperatorCase(
        "conv_mul_constant",
        "sima_conv_add_or_mul_const",
        lambda: ConvConstantPostOpModel("mul"),
        IMAGE_INPUT,
        (torch.ops.aten.mul.Tensor,),
        weighted=True,
    ),
    OperatorCase(
        "add",
        "add",
        lambda: BinaryModel("add"),
        IMAGE_INPUT,
        (torch.ops.aten.add.Tensor,),
        weighted=True,
        onnx_family="add",
    ),
    OperatorCase(
        "add_relu",
        "add_relu",
        lambda: BinaryModel("add", nn.ReLU()),
        IMAGE_INPUT,
        (torch.ops.aten.relu.default,),
        weighted=True,
    ),
    OperatorCase(
        "add_hardtanh",
        "sima_add_hardtanh",
        lambda: BinaryModel("add", nn.Hardtanh()),
        IMAGE_INPUT,
        (torch.ops.aten.hardtanh.default, torch.ops.aten.hardtanh_.default),
        weighted=True,
    ),
    OperatorCase(
        "mul",
        "mul",
        lambda: BinaryModel("mul"),
        IMAGE_INPUT,
        (torch.ops.aten.mul.Tensor,),
        weighted=True,
    ),
    OperatorCase(
        "mul_relu",
        "mul_relu",
        lambda: BinaryModel("mul", nn.ReLU()),
        IMAGE_INPUT,
        (torch.ops.aten.relu.default,),
        weighted=True,
    ),
    OperatorCase(
        "cat",
        "sima_cat",
        CatModel,
        IMAGE_INPUT,
        (torch.ops.aten.cat.default,),
        weighted=True,
        onnx_family="cat",
    ),
    OperatorCase(
        "sigmoid",
        "sima_sigmoid",
        lambda: ActivationModel(nn.Sigmoid()),
        IMAGE_INPUT,
        (torch.ops.aten.sigmoid.default,),
        weighted=True,
        onnx_family="activation",
    ),
    OperatorCase(
        "silu",
        "sima_silu",
        lambda: ActivationModel(nn.SiLU()),
        IMAGE_INPUT,
        (torch.ops.aten.silu.default, torch.ops.aten.silu_.default),
        weighted=True,
    ),
    OperatorCase(
        "adaptive_avg_pool2d",
        "adaptive_avg_pool2d",
        lambda: PoolModel(nn.AdaptiveAvgPool2d((4, 4))),
        IMAGE_INPUT,
        (torch.ops.aten.adaptive_avg_pool2d.default,),
        weighted=True,
        onnx_family="pool",
    ),
    OperatorCase(
        "slice_select_unsqueeze",
        "sima_slice_select_unsqueeze",
        SliceSelectUnsqueezeModel,
        IMAGE_INPUT,
        (torch.ops.aten.unsqueeze.default,),
        weighted=True,
    ),
    OperatorCase(
        "batchnorm",
        "sima_batchnorm",
        BatchNormModel,
        FOUR_CHANNEL_INPUT,
        (
            operator.getitem,
            torch.ops.aten.batch_norm.default,
            torch.ops.aten._native_batch_norm_legit.default,
        ),
    ),
    OperatorCase(
        "abs",
        "sima_unary_int8",
        lambda: UnaryOperatorModel(torch.abs),
        IMAGE_INPUT,
        (torch.ops.aten.abs.default,),
    ),
    OperatorCase(
        "elu",
        "sima_unary_int8",
        lambda: UnaryOperatorModel(torch.nn.functional.elu),
        IMAGE_INPUT,
        (torch.ops.aten.elu.default,),
    ),
    OperatorCase(
        "exp",
        "sima_unary_int8",
        lambda: UnaryOperatorModel(torch.exp),
        IMAGE_INPUT,
        (torch.ops.aten.exp.default,),
    ),
    OperatorCase(
        "hardsigmoid",
        "sima_unary_int8",
        lambda: UnaryOperatorModel(torch.nn.functional.hardsigmoid),
        IMAGE_INPUT,
        (torch.ops.aten.hardsigmoid.default,),
    ),
    OperatorCase(
        "hardswish",
        "sima_unary_int8",
        lambda: UnaryOperatorModel(torch.nn.functional.hardswish),
        IMAGE_INPUT,
        (torch.ops.aten.hardswish.default,),
    ),
    OperatorCase(
        "instance_norm",
        "sima_unary_int8",
        InstanceNormModel,
        IMAGE_INPUT,
        (torch.ops.aten.instance_norm.default,),
    ),
    OperatorCase(
        "leaky_relu",
        "sima_unary_int8",
        lambda: UnaryOperatorModel(torch.nn.functional.leaky_relu),
        IMAGE_INPUT,
        (torch.ops.aten.leaky_relu.default,),
    ),
    OperatorCase(
        "log",
        "sima_unary_int8",
        lambda: UnaryOperatorModel(torch.log),
        POSITIVE_IMAGE_INPUT,
        (torch.ops.aten.log.default,),
    ),
    OperatorCase(
        "log_softmax",
        "sima_unary_int8",
        lambda: UnaryOperatorModel(lambda value: torch.log_softmax(value, dim=1)),
        IMAGE_INPUT,
        (torch.ops.aten.log_softmax.int,),
    ),
    OperatorCase(
        "neg",
        "sima_unary_int8",
        lambda: UnaryOperatorModel(torch.neg),
        IMAGE_INPUT,
        (torch.ops.aten.neg.default,),
    ),
    OperatorCase(
        "reciprocal",
        "sima_unary_int8",
        lambda: UnaryOperatorModel(torch.reciprocal),
        POSITIVE_IMAGE_INPUT,
        (torch.ops.aten.reciprocal.default,),
    ),
    OperatorCase(
        "softplus",
        "sima_unary_int8",
        lambda: UnaryOperatorModel(torch.nn.functional.softplus),
        IMAGE_INPUT,
        (torch.ops.aten.softplus.default,),
    ),
    OperatorCase(
        "sqrt",
        "sima_unary_int8",
        lambda: UnaryOperatorModel(torch.sqrt),
        POSITIVE_IMAGE_INPUT,
        (torch.ops.aten.sqrt.default,),
    ),
    OperatorCase(
        "tanh",
        "sima_unary_int8",
        lambda: UnaryOperatorModel(torch.tanh),
        IMAGE_INPUT,
        (torch.ops.aten.tanh.default,),
    ),
    OperatorCase(
        "div",
        "sima_binary_int8",
        lambda: DirectBinaryModel(torch.div),
        PAIR_IMAGE_INPUT,
        (torch.ops.aten.div.Tensor,),
    ),
    OperatorCase(
        "sub",
        "sima_binary_int8",
        lambda: DirectBinaryModel(torch.sub),
        PAIR_IMAGE_INPUT,
        (torch.ops.aten.sub.Tensor,),
    ),
    OperatorCase(
        "einsum",
        "sima_einsum",
        EinsumModel,
        EINSUM_INPUT,
        (torch.ops.aten.einsum.default,),
    ),
    OperatorCase(
        "pow",
        "sima_pow",
        lambda: UnaryOperatorModel(lambda value: torch.pow(value, 2)),
        IMAGE_INPUT,
        (torch.ops.aten.pow.Tensor_Scalar,),
    ),
    OperatorCase(
        "reduce_mean",
        "sima_reduction",
        lambda: UnaryOperatorModel(lambda value: torch.mean(value, dim=(2, 3))),
        IMAGE_INPUT,
        (torch.ops.aten.mean.dim,),
    ),
    OperatorCase(
        "reduce_sum",
        "sima_reduction",
        lambda: UnaryOperatorModel(lambda value: torch.sum(value, dim=(2, 3))),
        IMAGE_INPUT,
        (torch.ops.aten.sum.dim_IntList,),
    ),
    OperatorCase(
        "reduce_max",
        "sima_reduction",
        lambda: UnaryOperatorModel(lambda value: torch.amax(value, dim=(2, 3))),
        IMAGE_INPUT,
        (torch.ops.aten.amax.default,),
    ),
    OperatorCase(
        "reduce_l1",
        "sima_reduction",
        lambda: UnaryOperatorModel(
            lambda value: torch.linalg.vector_norm(value, ord=1, dim=(2, 3))
        ),
        IMAGE_INPUT,
        (torch.ops.aten.linalg_vector_norm.default,),
    ),
    OperatorCase(
        "reduce_logsumexp",
        "sima_reduction",
        lambda: UnaryOperatorModel(lambda value: torch.logsumexp(value, dim=(2, 3))),
        IMAGE_INPUT,
        (torch.ops.aten.logsumexp.default,),
    ),
    OperatorCase(
        "global_average_pool",
        "adaptive_avg_pool2d",
        lambda: PoolModel(nn.AdaptiveAvgPool2d((1, 1))),
        IMAGE_INPUT,
        (torch.ops.aten.adaptive_avg_pool2d.default,),
        weighted=True,
    ),
    OperatorCase(
        "global_max_pool",
        "sima_global_max_pool2d",
        lambda: UnaryOperatorModel(
            lambda value: torch.nn.functional.adaptive_max_pool2d(value, (1, 1))[0]
        ),
        IMAGE_INPUT,
        (torch.ops.aten.adaptive_max_pool2d.default,),
    ),
    OperatorCase(
        "argmax",
        "sima_mixed_output",
        lambda: UnaryOperatorModel(lambda value: torch.argmax(value, dim=1)),
        IMAGE_INPUT,
        (torch.ops.aten.argmax.default,),
    ),
    OperatorCase(
        "topk_values",
        "sima_mixed_output",
        TopKValuesModel,
        IMAGE_INPUT,
        (torch.ops.aten.topk.default,),
    ),
    OperatorCase(
        "resize_nearest",
        "sima_unary_int8",
        lambda: UnaryOperatorModel(
            lambda value: torch.nn.functional.interpolate(
                value, scale_factor=2, mode="nearest"
            )
        ),
        IMAGE_INPUT,
        (torch.ops.aten.upsample_nearest2d.vec,),
    ),
    OperatorCase(
        "resize_bilinear",
        "sima_unary_int8",
        lambda: UnaryOperatorModel(
            lambda value: torch.nn.functional.interpolate(
                value, scale_factor=2, mode="bilinear", align_corners=False
            )
        ),
        IMAGE_INPUT,
        (torch.ops.aten.upsample_bilinear2d.vec,),
    ),
    OperatorCase(
        "expand",
        "sima_grid_preserving",
        ExpandAfterConvModel,
        EXPAND_INPUT,
        (torch.ops.aten.expand.default,),
        weighted=True,
    ),
    OperatorCase(
        "flatten",
        "sima_grid_preserving",
        lambda: LayoutAfterConvModel(lambda value: torch.flatten(value, 1)),
        IMAGE_INPUT,
        (torch.ops.aten.flatten.using_ints,),
        weighted=True,
    ),
    OperatorCase(
        "pad",
        "sima_grid_preserving",
        lambda: LayoutAfterConvModel(
            lambda value: torch.nn.functional.pad(value, (1, 1, 1, 1))
        ),
        IMAGE_INPUT,
        (torch.ops.aten.pad.default,),
        weighted=True,
    ),
    OperatorCase(
        "reshape",
        "sima_grid_preserving",
        lambda: LayoutAfterConvModel(
            lambda value: torch.reshape(value, (value.shape[0], 4, 64))
        ),
        IMAGE_INPUT,
        (torch.ops.aten.reshape.default,),
        weighted=True,
    ),
    OperatorCase(
        "split",
        "sima_split",
        SplitAfterConvModel,
        IMAGE_INPUT,
        (torch.ops.aten.split.Tensor,),
        weighted=True,
    ),
    OperatorCase(
        "depth_to_space",
        "sima_grid_preserving",
        lambda: LayoutAfterConvModel(lambda value: torch.pixel_shuffle(value, 2)),
        IMAGE_INPUT,
        (torch.ops.aten.pixel_shuffle.default,),
        weighted=True,
    ),
    OperatorCase(
        "space_to_depth",
        "sima_grid_preserving",
        lambda: LayoutAfterConvModel(lambda value: torch.pixel_unshuffle(value, 2)),
        IMAGE_INPUT,
        (torch.ops.aten.pixel_unshuffle.default,),
        weighted=True,
    ),
    OperatorCase(
        "transpose",
        "sima_grid_preserving",
        lambda: LayoutAfterConvModel(lambda value: value.transpose(2, 3)),
        IMAGE_INPUT,
        (torch.ops.aten.transpose.int,),
        weighted=True,
    ),
    OperatorCase(
        "tile",
        "sima_grid_preserving",
        lambda: LayoutAfterConvModel(lambda value: torch.tile(value, (1, 1, 2, 1))),
        IMAGE_INPUT,
        (torch.ops.aten.tile.default,),
        weighted=True,
    ),
)


PROPAGATED_CASES = (
    OperatorCase(
        "max_pool2d",
        "max_pool2d_propagated",
        lambda: PoolModel(nn.MaxPool2d(2)),
        IMAGE_INPUT,
        (torch.ops.aten.max_pool2d.default,),
        weighted=True,
        onnx_family="max_pool",
    ),
)


ALL_OPERATOR_CASES = OPERATOR_CASES + PROPAGATED_CASES
WEIGHTED_CASES = tuple(case for case in ALL_OPERATOR_CASES if case.weighted)
ONNX_CASES = tuple(case for case in ALL_OPERATOR_CASES if case.name in ONNX_TEST_CASE_IDS)


def case_ids(case: OperatorCase) -> str:
    return case.name
