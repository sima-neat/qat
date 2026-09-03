"""Small model factories covering every active SiMa QAT operator pattern."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor, nn


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
    def __init__(self, operation: str = "matmul") -> None:
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
        return torch.softmax(inputs, dim=-1)


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


IMAGE_INPUT: InputFactory = lambda: (torch.randn(2, 3, 8, 8),)
FOUR_CHANNEL_INPUT: InputFactory = lambda: (torch.randn(2, 4, 8, 8),)
SEQUENCE_INPUT: InputFactory = lambda: (torch.randn(2, 3, 12),)
LINEAR_INPUT: InputFactory = lambda: (torch.randn(2, 8),)
MM_INPUT: InputFactory = lambda: (torch.randn(4, 8), torch.randn(8, 3))
BMM_INPUT: InputFactory = lambda: (torch.randn(2, 4, 8), torch.randn(2, 8, 3))
BADDBMM_INPUT: InputFactory = lambda: (
    torch.randn(2, 4, 3),
    torch.randn(2, 4, 8),
    torch.randn(2, 8, 3),
)


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
        MatMulModel,
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
        LINEAR_INPUT,
        (torch.ops.aten.softmax.int, torch.ops.aten._softmax.default),
        onnx_family="softmax",
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
ONNX_CASES = tuple(case for case in ALL_OPERATOR_CASES if case.onnx_family is not None)


def case_ids(case: OperatorCase) -> str:
    return case.name
