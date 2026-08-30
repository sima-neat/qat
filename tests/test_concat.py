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
import copy
import torch

from sima_qat.qat_api import (sima_prepare_qat_model, 
                              sima_finalize_qat_model, 
                              sima_export_onnx)

import pytest


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(10, 10, (1, 1))
        self.conv2 = torch.nn.Conv2d(10, 10, (1, 1))
        self.conv2.weight = torch.nn.Parameter(self.conv1.weight + 1.)
        self.conv2.bias = torch.nn.Parameter(self.conv1.bias + 1.)
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x = torch.cat([x1, x2])
        return x


class RepeatedInputConcatModel(torch.nn.Module):
    """C1-to-C16 ABI packing must remain on one activation grid."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, kernel_size=1)

    def forward(self, x):
        depth = torch.relu(self.conv(x))
        return torch.cat((depth,) * 16, dim=1)


class IdentityPaddingConcatModel(torch.nn.Module):
    """Tree-prefix identity padding must not introduce a new grid."""

    def __init__(self, identity: float):
        super().__init__()
        self.identity = identity
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        prefix_source = x[:1]
        prefix = (
            torch.zeros_like(prefix_source)
            if self.identity == 0.0
            else torch.ones_like(prefix_source)
        )
        return torch.cat((prefix, x[:-1]), dim=0)


class SingleInputConcatEinsumModel(torch.nn.Module):
    """A one-iteration eager loop may emit Cat([Einsum]) in the FX graph."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(3))

    def forward(self, x):
        projected = torch.einsum("nchw,oc->nohw", x, self.weight)
        return torch.cat([projected], dim=0)


@pytest.mark.regression
@pytest.mark.parametrize("model", [Model()])
def test_concat(model: torch.nn.Module):
    input_tensor = torch.randn(1, 10, 16, 16)
    model(input_tensor)
    example_inputs = (input_tensor, )

    prepared_model = sima_prepare_qat_model(model, example_inputs, 'cpu')

    prepared_model(example_inputs[0])

    prepared_model.cpu()

    converted_model = sima_finalize_qat_model(prepared_model)

    scales = []
    for node in converted_model.graph.nodes:

        if 'dequantize_per_tensor' in node.name:

            if 'cat' in [n.name for n in node.users]:
                scales.append(node.args[1])

                cat_node = [n for n in node.users][0]
                # On newer torch a graph-terminal cat feeds 'output' directly with no
                # output re-quantization, so only walk the cat -> quantize -> dequantize
                # chain when it actually exists.
                for q_node in cat_node.users:
                    if 'quantize_per_tensor' not in q_node.name or 'dequantize' in q_node.name:
                        continue
                    for dq_node in q_node.users:
                        if 'dequantize_per_tensor' in dq_node.name and dq_node.args[1] not in scales:
                            scales.append(dq_node.args[1])

    # Concat must preserve distinct per-input quantization scales (not collapse them).
    assert len(scales) >= 2
    assert not all(x == scales[0] for x in scales)


@pytest.mark.regression
def test_repeated_input_concat_is_quantized_on_shared_grid():
    model = RepeatedInputConcatModel().eval()
    input_tensor = torch.randn(1, 3, 8, 8)
    prepared = sima_prepare_qat_model(model, (input_tensor,), "cpu")
    prepared(input_tensor)

    cat = next(
        node
        for node in prepared.graph.nodes
        if node.target == torch.ops.aten.cat.default
    )
    assert any(node.op == "call_module" for node in cat.users)

    converted = sima_finalize_qat_model(prepared)
    cat = next(
        node
        for node in converted.graph.nodes
        if node.target == torch.ops.aten.cat.default
    )
    quantize = next(iter(cat.users))
    assert (
        quantize.target
        == torch.ops.quantized_decomposed.quantize_per_tensor.default
    )
    dequantize_inputs = list(cat.args[0])
    assert len(dequantize_inputs) == 16
    assert len({node for node in dequantize_inputs}) == 1
    input_dequantize = dequantize_inputs[0]
    assert quantize.args[1:3] == input_dequantize.args[1:3]


@pytest.mark.regression
@pytest.mark.parametrize("identity", [0.0, 1.0])
def test_identity_padding_concat_uses_payload_grid(identity):
    model = IdentityPaddingConcatModel(identity).eval()
    input_tensor = torch.randn(4, 3, 8, 8)
    prepared = sima_prepare_qat_model(model, (input_tensor,), "cpu")
    prepared(input_tensor)
    converted = sima_finalize_qat_model(prepared)

    cat = next(
        node
        for node in converted.graph.nodes
        if node.target == torch.ops.aten.cat.default
    )
    input_dequantize = list(cat.args[0])
    assert len(input_dequantize) == 2
    output_quantize = next(
        node
        for node in cat.users
        if node.target == torch.ops.quantized_decomposed.quantize_per_tensor.default
    )
    assert input_dequantize[0].args[1:3] == input_dequantize[1].args[1:3]
    assert output_quantize.args[1:3] == input_dequantize[1].args[1:3]


@pytest.mark.regression
def test_single_input_concat_has_concrete_shared_grid_root():
    model = SingleInputConcatEinsumModel().eval()
    input_tensor = torch.randn(1, 3, 8, 8)
    prepared = sima_prepare_qat_model(model, (input_tensor,), "cpu")
    prepared(input_tensor)
    converted = sima_finalize_qat_model(prepared)

    cat = next(
        node
        for node in converted.graph.nodes
        if node.target == torch.ops.aten.cat.default
    )
    input_dequantize = list(cat.args[0])
    assert len(input_dequantize) == 1
    output_quantize = next(
        node
        for node in cat.users
        if node.target == torch.ops.quantized_decomposed.quantize_per_tensor.default
    )
    assert output_quantize.args[1:3] == input_dequantize[0].args[1:3]
