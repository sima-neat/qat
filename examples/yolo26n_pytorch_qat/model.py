"""Minimal, training-only YOLO26n definition built from ordinary PyTorch modules.

The module layout intentionally matches the public YOLO26n checkpoint layout so
that a converted state dictionary can be loaded strictly. Inference decoding,
top-k selection, and NMS are deliberately outside this training graph.
"""

from __future__ import annotations

import copy
import math

import torch
from torch import Tensor, nn


def _autopad(kernel: int | tuple[int, int], padding=None, dilation: int = 1):
    if dilation > 1:
        kernel = (
            dilation * (kernel - 1) + 1
            if isinstance(kernel, int)
            else tuple(dilation * (value - 1) + 1 for value in kernel)
        )
    if padding is None:
        padding = (
            kernel // 2
            if isinstance(kernel, int)
            else tuple(value // 2 for value in kernel)
        )
    return padding


class Conv(nn.Module):
    """Convolution, BatchNorm, and SiLU used by YOLO26."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int | tuple[int, int] = 1,
        stride: int = 1,
        padding=None,
        groups: int = 1,
        dilation: int = 1,
        activation: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel,
            stride,
            _autopad(kernel, padding, dilation),
            groups=groups,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU(inplace=True) if activation else nn.Identity()

    def forward(self, inputs: Tensor) -> Tensor:
        return self.act(self.bn(self.conv(inputs)))


class DWConv(Conv):
    def __init__(self, in_channels: int, out_channels: int, kernel=1, stride=1, activation=True):
        super().__init__(
            in_channels,
            out_channels,
            kernel,
            stride,
            groups=math.gcd(in_channels, out_channels),
            activation=activation,
        )


class Bottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        groups: int = 1,
        kernels: tuple[int, int] = (3, 3),
        expansion: float = 0.5,
    ) -> None:
        super().__init__()
        hidden = int(out_channels * expansion)
        self.cv1 = Conv(in_channels, hidden, kernels[0])
        self.cv2 = Conv(hidden, out_channels, kernels[1], groups=groups)
        self.add = shortcut and in_channels == out_channels

    def forward(self, inputs: Tensor) -> Tensor:
        output = self.cv2(self.cv1(inputs))
        return inputs + output if self.add else output


class C2f(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        repeats: int = 1,
        shortcut: bool = False,
        groups: int = 1,
        expansion: float = 0.5,
    ) -> None:
        super().__init__()
        self.c = int(out_channels * expansion)
        self.cv1 = Conv(in_channels, 2 * self.c)
        self.cv2 = Conv((2 + repeats) * self.c, out_channels)
        self.m = nn.ModuleList(
            Bottleneck(
                self.c,
                self.c,
                shortcut,
                groups,
                kernels=(3, 3),
                expansion=1.0,
            )
            for _ in range(repeats)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = list(self.cv1(inputs).chunk(2, dim=1))
        outputs.extend(block(outputs[-1]) for block in self.m)
        return self.cv2(torch.cat(outputs, dim=1))


class C3(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        repeats: int = 1,
        shortcut: bool = True,
        groups: int = 1,
        expansion: float = 0.5,
    ) -> None:
        super().__init__()
        hidden = int(out_channels * expansion)
        self.cv1 = Conv(in_channels, hidden)
        self.cv2 = Conv(in_channels, hidden)
        self.cv3 = Conv(2 * hidden, out_channels)
        self.m = nn.Sequential(
            *(
                Bottleneck(
                    hidden,
                    hidden,
                    shortcut,
                    groups,
                    kernels=(1, 3),
                    expansion=1.0,
                )
                for _ in range(repeats)
            )
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(inputs)), self.cv2(inputs)), dim=1))


class C3k(C3):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        repeats: int = 1,
        shortcut: bool = True,
        groups: int = 1,
        expansion: float = 0.5,
        kernel: int = 3,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            repeats,
            shortcut,
            groups,
            expansion,
        )
        hidden = int(out_channels * expansion)
        self.m = nn.Sequential(
            *(
                Bottleneck(
                    hidden,
                    hidden,
                    shortcut,
                    groups,
                    kernels=(kernel, kernel),
                    expansion=1.0,
                )
                for _ in range(repeats)
            )
        )


class Attention(nn.Module):
    def __init__(self, channels: int, heads: int, attention_ratio: float = 0.5) -> None:
        super().__init__()
        self.num_heads = heads
        self.head_dim = channels // heads
        self.key_dim = int(self.head_dim * attention_ratio)
        self.scale = self.key_dim**-0.5
        qkv_channels = channels + 2 * self.key_dim * heads
        self.qkv = Conv(channels, qkv_channels, activation=False)
        self.proj = Conv(channels, channels, activation=False)
        self.pe = Conv(channels, channels, 3, groups=channels, activation=False)

    def forward(self, inputs: Tensor) -> Tensor:
        batch, channels, height, width = inputs.shape
        tokens = height * width
        qkv = self.qkv(inputs)
        query, key, value = qkv.view(
            batch,
            self.num_heads,
            2 * self.key_dim + self.head_dim,
            tokens,
        ).split((self.key_dim, self.key_dim, self.head_dim), dim=2)
        probabilities = ((query * self.scale).transpose(-2, -1) @ key).softmax(dim=-1)
        attended = (value @ probabilities.transpose(-2, -1)).view(
            batch, channels, height, width
        )
        positioned = self.pe(value.reshape(batch, channels, height, width))
        return self.proj(attended + positioned)


class PSABlock(nn.Module):
    def __init__(self, channels: int, heads: int, shortcut: bool = True) -> None:
        super().__init__()
        self.attn = Attention(channels, heads=heads, attention_ratio=0.5)
        self.ffn = nn.Sequential(
            Conv(channels, channels * 2),
            Conv(channels * 2, channels, activation=False),
        )
        self.add = shortcut

    def forward(self, inputs: Tensor) -> Tensor:
        output = inputs + self.attn(inputs) if self.add else self.attn(inputs)
        return output + self.ffn(output) if self.add else self.ffn(output)


class C2PSA(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, repeats: int = 1) -> None:
        super().__init__()
        if in_channels != out_channels:
            raise ValueError("C2PSA requires equal input and output channels")
        self.c = int(in_channels * 0.5)
        self.cv1 = Conv(in_channels, 2 * self.c)
        self.cv2 = Conv(2 * self.c, out_channels)
        self.m = nn.Sequential(
            *(
                PSABlock(self.c, heads=max(self.c // 64, 1))
                for _ in range(repeats)
            )
        )

    def forward(self, inputs: Tensor) -> Tensor:
        left, right = self.cv1(inputs).split((self.c, self.c), dim=1)
        return self.cv2(torch.cat((left, self.m(right)), dim=1))


class C3k2(C2f):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        repeats: int = 1,
        c3k: bool = False,
        expansion: float = 0.5,
        attention: bool = False,
        groups: int = 1,
        shortcut: bool = True,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            repeats,
            shortcut,
            groups,
            expansion,
        )
        blocks = []
        for _ in range(repeats):
            if attention:
                block = nn.Sequential(
                    Bottleneck(self.c, self.c, shortcut, groups),
                    PSABlock(self.c, heads=max(self.c // 64, 1)),
                )
            elif c3k:
                block = C3k(self.c, self.c, 2, shortcut, groups)
            else:
                block = Bottleneck(self.c, self.c, shortcut, groups)
            blocks.append(block)
        self.m = nn.ModuleList(blocks)


class SPPF(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int = 5,
        repeats: int = 3,
        shortcut: bool = False,
    ) -> None:
        super().__init__()
        hidden = in_channels // 2
        self.cv1 = Conv(in_channels, hidden, activation=False)
        self.cv2 = Conv(hidden * (repeats + 1), out_channels)
        self.m = nn.MaxPool2d(kernel, stride=1, padding=kernel // 2)
        self.n = repeats
        self.add = shortcut and in_channels == out_channels

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = [self.cv1(inputs)]
        outputs.extend(self.m(outputs[-1]) for _ in range(self.n))
        output = self.cv2(torch.cat(outputs, dim=1))
        return output + inputs if self.add else output


class Concat(nn.Module):
    def __init__(self, dimension: int = 1) -> None:
        super().__init__()
        self.d = dimension

    def forward(self, inputs: list[Tensor]) -> Tensor:
        return torch.cat(inputs, dim=self.d)


class Detect(nn.Module):
    """YOLO26 dual raw head. Decoding and top-k are intentionally omitted."""

    def __init__(self, classes: int = 80, channels=(64, 128, 256)) -> None:
        super().__init__()
        self.nc = classes
        self.nl = len(channels)
        self.reg_max = 1
        self.no = classes + 4
        self.stride = torch.tensor((8.0, 16.0, 32.0))
        box_channels = max(16, channels[0] // 4, self.reg_max * 4)
        class_channels = max(channels[0], min(classes, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(value, box_channels, 3),
                Conv(box_channels, box_channels, 3),
                nn.Conv2d(box_channels, 4, 1),
            )
            for value in channels
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(value, value, 3), Conv(value, class_channels)),
                nn.Sequential(
                    DWConv(class_channels, class_channels, 3),
                    Conv(class_channels, class_channels),
                ),
                nn.Conv2d(class_channels, classes, 1),
            )
            for value in channels
        )
        self.dfl = nn.Identity()
        self.one2one_cv2 = copy.deepcopy(self.cv2)
        self.one2one_cv3 = copy.deepcopy(self.cv3)

    @staticmethod
    def _forward_head(
        features: list[Tensor],
        box_head: nn.ModuleList,
        class_head: nn.ModuleList,
        classes: int,
    ) -> dict[str, Tensor | list[Tensor]]:
        batch = features[0].shape[0]
        boxes = torch.cat(
            [box_head[index](value).view(batch, 4, -1) for index, value in enumerate(features)],
            dim=-1,
        )
        scores = torch.cat(
            [
                class_head[index](value).view(batch, classes, -1)
                for index, value in enumerate(features)
            ],
            dim=-1,
        )
        return {"boxes": boxes, "scores": scores, "feats": features}

    def forward(self, features: list[Tensor]):
        one2many = self._forward_head(features, self.cv2, self.cv3, self.nc)
        detached = [value.detach() for value in features]
        one2one = self._forward_head(
            detached,
            self.one2one_cv2,
            self.one2one_cv3,
            self.nc,
        )
        return {"one2many": one2many, "one2one": one2one}


class YOLO26n(nn.Module):
    """COCO YOLO26 nano training graph with raw dual-head outputs."""

    classes = 80
    strides = (8, 16, 32)
    reg_max = 1

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.ModuleList(
            [
                Conv(3, 16, 3, 2),
                Conv(16, 32, 3, 2),
                C3k2(32, 64, 1, c3k=False, expansion=0.25),
                Conv(64, 64, 3, 2),
                C3k2(64, 128, 1, c3k=False, expansion=0.25),
                Conv(128, 128, 3, 2),
                C3k2(128, 128, 1, c3k=True),
                Conv(128, 256, 3, 2),
                C3k2(256, 256, 1, c3k=True),
                SPPF(256, 256, 5, 3, shortcut=True),
                C2PSA(256, 256, 1),
                nn.Upsample(scale_factor=2, mode="nearest"),
                Concat(1),
                C3k2(384, 128, 1, c3k=True),
                nn.Upsample(scale_factor=2, mode="nearest"),
                Concat(1),
                C3k2(256, 64, 1, c3k=True),
                Conv(64, 64, 3, 2),
                Concat(1),
                C3k2(192, 128, 1, c3k=True),
                Conv(128, 128, 3, 2),
                Concat(1),
                C3k2(384, 256, 1, c3k=True, expansion=0.5, attention=True),
                Detect(80, (64, 128, 256)),
            ]
        )

    def forward(self, inputs: Tensor):
        layer0 = self.model[0](inputs)
        layer1 = self.model[1](layer0)
        layer2 = self.model[2](layer1)
        layer3 = self.model[3](layer2)
        layer4 = self.model[4](layer3)
        layer5 = self.model[5](layer4)
        layer6 = self.model[6](layer5)
        layer7 = self.model[7](layer6)
        layer8 = self.model[8](layer7)
        layer9 = self.model[9](layer8)
        layer10 = self.model[10](layer9)
        layer11 = self.model[11](layer10)
        layer12 = self.model[12]([layer11, layer6])
        layer13 = self.model[13](layer12)
        layer14 = self.model[14](layer13)
        layer15 = self.model[15]([layer14, layer4])
        layer16 = self.model[16](layer15)
        layer17 = self.model[17](layer16)
        layer18 = self.model[18]([layer17, layer13])
        layer19 = self.model[19](layer18)
        layer20 = self.model[20](layer19)
        layer21 = self.model[21]([layer20, layer10])
        layer22 = self.model[22](layer21)
        return self.model[23]([layer16, layer19, layer22])


def build_yolo26n() -> YOLO26n:
    model = YOLO26n()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 2_572_280:
        raise RuntimeError(
            f"YOLO26n construction drifted: expected 2,572,280 parameters, found {parameter_count:,}"
        )
    return model
