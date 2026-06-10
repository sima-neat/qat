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
import torch
from torch import nn, Tensor

from sima_qat.qat_api import (sima_prepare_qat_model, 
                              sima_finalize_qat_model, 
                              sima_export_onnx,
                              check_graph_nodes, 
                              device_modifier_ops)

import pytest

def stochastic_depth(input: Tensor, p: float, mode: str, training: bool = True) -> Tensor:
    """
    Implements the Stochastic Depth from `"Deep Networks with Stochastic Depth"
    <https://arxiv.org/abs/1603.09382>`_ used for randomly dropping residual
    branches of residual architectures.

    Args:
        input (Tensor[N, ...]): The input tensor or arbitrary dimensions with the first one
                    being its batch i.e. a batch with ``N`` rows.
        p (float): probability of the input to be zeroed.
        mode (str): ``"batch"`` or ``"row"``.
                    ``"batch"`` randomly zeroes the entire input, ``"row"`` zeroes
                    randomly selected rows from the batch.
        training: apply stochastic depth if is ``True``. Default: ``True``

    Returns:
        Tensor[N, ...]: The randomly zeroed tensor.
    """
    if p < 0.0 or p > 1.0:
        raise ValueError(f"drop probability has to be between 0 and 1, but got {p}")
    if mode not in ["batch", "row"]:
        raise ValueError(f"mode has to be either 'batch' or 'row', but got {mode}")
    if not training or p == 0.0:
        return input

    survival_rate = 1.0 - p
    if mode == "row":
        size = [input.shape[0]] + [1] * (input.ndim - 1)
    else:
        size = [1] * input.ndim
    noise = torch.empty(size, dtype=input.dtype, device=input.device)
    noise = noise.bernoulli_(survival_rate)
    if survival_rate > 0.0:
        noise.div_(survival_rate)
    return input * noise


class CheckDeviceModel(torch.nn.Module):

    def __init__(self, p: float, mode: str):
        super(CheckDeviceModel, self).__init__()
        self.p = p
        self.mode = mode
        

    def forward(self, input):
        return stochastic_depth(input, self.p, self.mode, self.training)


@pytest.mark.regression
@pytest.mark.parametrize("model", [CheckDeviceModel(p=0.05, mode="row")])
def test_prepared_model_device(model: torch.nn.Module):
    
    example_inputs = (torch.randn(1, 3, 224, 224),)
    prepared_model = sima_prepare_qat_model(model, example_inputs, 'cpu')
    
    for n in prepared_model.graph.nodes:
        #check for parameters not being in the same device as the model
        if n.target in device_modifier_ops:
            n_kwargs = dict(n.kwargs)
            assert n_kwargs['device'] is 'cpu'
    
    prepared_model = check_graph_nodes(prepared_model, 'cuda')

    for n in prepared_model.graph.nodes:
        #check for parameters not being in the same device as the model
        if n.target in device_modifier_ops:
            n_kwargs = dict(n.kwargs)
            assert n_kwargs['device'] is 'cuda'
    
    prepared_model.train(True)
    prepared_model = check_graph_nodes(prepared_model, 'cpu')
    prepared_model(example_inputs[0])
    prepared_model.train(False)
            
    finalized_model = sima_finalize_qat_model(prepared_model)
    post_export_model = sima_export_onnx(qat_model=finalized_model, inputs=example_inputs, output_file='exported_model.onnx', device='cuda')
    
    for n in post_export_model.graph.nodes:
        #check for parameters not being in the same device as the model
        if n.target in device_modifier_ops:
            n_kwargs = dict(n.kwargs)
            assert n_kwargs['device'] is 'cuda'