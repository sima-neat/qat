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
from torch.ao.quantization import move_exported_model_to_eval, move_exported_model_to_train
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


def sima_prepare_qat_model(input_graph: nn.Module, inputs: Tuple, device: torch.device) -> GraphModule:
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

    cfg = get_sima_quantization_config(is_qat=True)
    quantizer = SimaQuantizer().set_global(cfg)
    gm = prepare_qat_pt2e(m, quantizer)
    sima_mod = SimaQatWrapper(source=gm, label='scaffold')
    sima_mod.to(device)
    sima_mod.train()
    sima_mod = check_graph_nodes(sima_mod, device)

    return sima_mod


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
    print(f"Removing QAT scaffold and quantizing network ...")
    device = _get_module_device(qat_model)
    _ensure_bn_tracking_meta(qat_model)
    m = convert_pt2e(qat_model, use_reference_representation=False)
    sima_mod = SimaQatWrapper(source=m, label='fq')
    sima_mod.to(device)
    # We must call eval() to invoke internal functions to put the GraphModule in eval state. Once we are
    # in FQ mode, we always remain in eval mode.
    sima_mod.eval()
    sima_mod = replace_batchnorm(sima_mod)
    return sima_mod


def sima_export_onnx(qat_model: nn.Module, inputs: Tuple[Tensor], output_file: str, input_names: Optional[List[str]] = None, 
                     output_names: Optional[List[str]] = None,
                     device: Optional[Union[str, torch.device]] = None) -> GraphModule:
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
    """
    if not isinstance(qat_model, nn.Module):
        raise RuntimeError(f"Input graph to export function must be of type nn.Module, found {type(qat_model)}")

    original_device = _get_module_device(qat_model)
    restore_device = torch.device(device) if device is not None else original_device
    export_device_arg = "cpu"
    export_device = torch.device(export_device_arg)

    qat_model.to(export_device)
    export_inputs = _move_value_to_device(inputs, export_device)
    qat_model = check_graph_nodes(qat_model, device=export_device_arg)
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
        self.register_buffer(
            "qat_state",
            torch.tensor([state_id], dtype=torch.int8, device=_get_module_device(self)),
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

        if state_dict['qat_state'] != state_id:
            raise RuntimeError(f"Error: model QAT state {state_id} doesn't match state_dict QAT state {state_dict['qat_state']}")

        return super().load_state_dict(state_dict, strict, assign)


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

