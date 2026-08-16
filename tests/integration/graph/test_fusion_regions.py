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
import operator

import onnx
import onnxruntime
import pytest
import torch
from torch import nn
import torch.nn.functional as F
from torch.ao.nn.intrinsic import qat as intrinsic_qat
from torch.ao.quantization import FakeQuantize
from torch.ao.quantization.observer import PerChannelMinMaxObserver

from sima_qat.qat_api import (
    sima_export_onnx,
    sima_finalize_qat_model,
    sima_prepare_qat_model,
)


class ConvReluModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.activation = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, inputs):
        return self.pool(self.activation(self.conv(inputs)))


def test_conv_relu_is_one_public_fx_qat_fusion():
    example_inputs = (torch.randn(2, 3, 8, 8),)
    prepared = sima_prepare_qat_model(ConvReluModel(), example_inputs, "cpu")

    fused_modules = [
        module
        for module in prepared.modules()
        if isinstance(module, intrinsic_qat.ConvReLU2d)
    ]
    assert len(fused_modules) == 1
    assert not any(type(module) is nn.ReLU for module in prepared.modules())
    assert isinstance(
        fused_modules[0].weight_fake_quant,
        PerChannelMinMaxObserver,
    )

    outputs = prepared(example_inputs[0])
    assert outputs.shape == (2, 4, 1, 1)
    finalized = sima_finalize_qat_model(prepared)

    assert torch.isfinite(finalized(example_inputs[0])).all()
    assert isinstance(fused_modules[0].weight_fake_quant, FakeQuantize)

class ConvHardtanhModel(nn.Module):
    def __init__(self, *, batchnorm=False, functional=False, keyword=False):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=not batchnorm)
        self.use_batchnorm = batchnorm
        if batchnorm:
            self.batchnorm = nn.BatchNorm2d(4)
        self.hardtanh = None if functional else nn.Hardtanh(-1.0, 1.0)
        self.keyword = keyword

    def forward(self, inputs):
        outputs = self.conv(inputs)
        if self.use_batchnorm:
            outputs = self.batchnorm(outputs)
        if self.hardtanh is None:
            return F.hardtanh(outputs, min_val=-1.0, max_val=1.0)
        if self.keyword:
            return self.hardtanh(input=outputs)
        return self.hardtanh(outputs)


class AddHardtanhModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hardtanh = nn.Hardtanh(-1.0, 1.0)

    def forward(self, left, right):
        return self.hardtanh(left + right)


class ConvConstantArithmeticModel(nn.Module):
    def __init__(self, operation, *, reverse=False, scalar=False, keyword=False):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.operation = operation
        self.reverse = reverse
        self.scalar = scalar
        self.keyword = keyword
        self.register_buffer("constant", torch.randn(1, 4, 1, 1))

    def forward(self, inputs):
        outputs = self.conv(inputs)
        constant = 0.25 if self.scalar else self.constant
        if self.operation == "add":
            if self.keyword:
                return torch.add(input=outputs, other=constant)
            return constant + outputs if self.reverse else outputs + constant
        return constant * outputs if self.reverse else outputs * constant


class ConvRuntimeArithmeticModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)

    def forward(self, inputs, runtime_value):
        return self.conv(inputs) + runtime_value


class SiluModel(nn.Module):
    def __init__(self, *, functional):
        super().__init__()
        self.silu = None if functional else nn.SiLU()

    def forward(self, inputs):
        if self.silu is None:
            return F.silu(inputs)
        return self.silu(inputs)


def _is_activation_fake_quant(model, node):
    return (
        getattr(node, "op", None) == "call_module"
        and isinstance(model.get_submodule(node.target), FakeQuantize)
        and not bool(model.get_submodule(node.target).is_per_channel)
    )


def _assert_single_fake_quant_user(model, node):
    users = list(node.users)
    assert len(users) == 1
    assert _is_activation_fake_quant(model, users[0])


def _binary_graph_operands(node):
    if len(node.args) >= 2:
        return node.args[0], node.args[1]
    return node.kwargs["input"], node.kwargs["other"]


def _fusion_case(case):
    if case == "conv_hardtanh":
        return ConvHardtanhModel(), (torch.randn(2, 3, 8, 8),)
    if case == "conv_functional_hardtanh":
        return ConvHardtanhModel(functional=True), (torch.randn(2, 3, 8, 8),)
    if case == "conv_keyword_hardtanh":
        return ConvHardtanhModel(keyword=True), (torch.randn(2, 3, 8, 8),)
    if case == "conv_bn_hardtanh":
        return ConvHardtanhModel(batchnorm=True), (torch.randn(2, 3, 8, 8),)
    if case == "add_hardtanh":
        return AddHardtanhModel(), (
            torch.randn(2, 4, 8, 8),
            torch.randn(2, 4, 8, 8),
        )
    if case == "conv_add_constant":
        return ConvConstantArithmeticModel("add"), (torch.randn(2, 3, 8, 8),)
    if case == "conv_mul_constant":
        return ConvConstantArithmeticModel("mul"), (torch.randn(2, 3, 8, 8),)
    if case == "constant_add_conv":
        return ConvConstantArithmeticModel(
            "add", reverse=True
        ), (torch.randn(2, 3, 8, 8),)
    if case == "constant_mul_conv":
        return ConvConstantArithmeticModel(
            "mul", reverse=True
        ), (torch.randn(2, 3, 8, 8),)
    if case == "conv_add_scalar":
        return ConvConstantArithmeticModel(
            "add", scalar=True
        ), (torch.randn(2, 3, 8, 8),)
    if case == "conv_add_keyword_constant":
        return ConvConstantArithmeticModel(
            "add", keyword=True
        ), (torch.randn(2, 3, 8, 8),)
    if case == "conv_add_runtime":
        return ConvRuntimeArithmeticModel(), (
            torch.randn(2, 3, 8, 8),
            torch.randn(2, 4, 8, 8),
        )
    if case == "silu_module":
        return SiluModel(functional=False), (torch.randn(2, 4, 8, 8),)
    if case == "silu_functional":
        return SiluModel(functional=True), (torch.randn(2, 4, 8, 8),)
    raise AssertionError(f"Unknown fusion case {case}")


def _assert_legacy_region(model, case):
    if "hardtanh" in case:
        hardtanh = next(
            node
            for node in model.graph.nodes
            if (
                node.op == "call_module"
                and isinstance(model.get_submodule(node.target), nn.Hardtanh)
            )
            or (node.op == "call_function" and node.target is F.hardtanh)
        )
        hardtanh_input = (
            hardtanh.args[0]
            if hardtanh.args
            else hardtanh.kwargs["input"]
        )
        assert not _is_activation_fake_quant(model, hardtanh_input)
        _assert_single_fake_quant_user(model, hardtanh)
        return

    if "runtime" in case:
        arithmetic = next(
            node
            for node in model.graph.nodes
            if node.op == "call_function"
            and node.target in (operator.add, torch.add)
        )
        operands = _binary_graph_operands(arithmetic)
        assert all(
            _is_activation_fake_quant(model, argument)
            for argument in operands
        )
        _assert_single_fake_quant_user(model, arithmetic)
        return

    if "constant" in case or "scalar" in case:
        targets = (operator.add, torch.add) if "add" in case else (
            operator.mul,
            torch.mul,
        )
        arithmetic = next(
            node
            for node in model.graph.nodes
            if node.op == "call_function" and node.target in targets
        )
        operands = _binary_graph_operands(arithmetic)
        assert not any(
            _is_activation_fake_quant(model, argument)
            for argument in operands
        )
        assert any(
            not hasattr(argument, "op") or argument.op == "get_attr"
            for argument in operands
        )
        _assert_single_fake_quant_user(model, arithmetic)
        return

    silu = next(
        node
        for node in model.graph.nodes
        if (
            node.op == "call_module"
            and isinstance(model.get_submodule(node.target), nn.SiLU)
        )
        or (node.op == "call_function" and node.target is F.silu)
    )
    assert _is_activation_fake_quant(model, silu.args[0])
    _assert_single_fake_quant_user(model, silu)


@pytest.mark.parametrize(
    "case",
    [
        "conv_hardtanh",
        "conv_functional_hardtanh",
        "conv_bn_hardtanh",
        "conv_keyword_hardtanh",
        "add_hardtanh",
        "conv_add_constant",
        "conv_mul_constant",
        "constant_add_conv",
        "constant_mul_conv",
        "conv_add_scalar",
        "conv_add_keyword_constant",
        "conv_add_runtime",
        "silu_module",
        "silu_functional",
    ],
)
def test_legacy_fusion_regions_export_with_standard_qdq(case, tmp_path):
    torch.manual_seed(23)
    model, example_inputs = _fusion_case(case)
    prepared = sima_prepare_qat_model(model, example_inputs, "cpu")
    _assert_legacy_region(prepared, case)
    prepared(*example_inputs)

    finalized = sima_finalize_qat_model(prepared)
    if case == "conv_bn_hardtanh":
        assert not any(
            isinstance(module, nn.modules.batchnorm._BatchNorm)
            for module in finalized.modules()
        )

    with torch.no_grad():
        expected = finalized(*example_inputs).detach().cpu()

    output_path = tmp_path / f"{case}.onnx"
    input_names = [f"input_{index}" for index in range(len(example_inputs))]
    sima_export_onnx(
        finalized,
        example_inputs,
        str(output_path),
        input_names=input_names,
        output_names=["output"],
        device="cpu",
    )
    exported = onnx.load(output_path)
    onnx.checker.check_model(exported)

    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = (
        onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    session = onnxruntime.InferenceSession(
        str(output_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    actual = session.run(
        None,
        {
            name: value.detach().cpu().numpy()
            for name, value in zip(input_names, example_inputs)
        },
    )[0]
    torch.testing.assert_close(
        torch.from_numpy(actual),
        expected,
        rtol=1e-6,
        atol=1e-6,
    )
