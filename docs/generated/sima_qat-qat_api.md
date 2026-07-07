# `sima_qat.qat_api`

Source: `sima_qat/qat_api.py`

## Public API

### Function: `sima_prepare_qat_model(input_graph, inputs, device)`

This function is the first transformation needed to perform QAT on a Pytorch model. It takes an
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

### Function: `sima_finalize_qat_model(qat_model)`

This function takes a QAT scaffolded model which has completed the training regimen and
converts it to an inference-only (via fakequant) form. Once this process is complete, the model
can no longer be trained, and is intended for export via ONNX.

Args:
    qat_model: a trained QAT model to be converted into inference-only form.

Returns:
    GraphModule: an inference-only version of the QAT model, which can be run in Pytorch
        `eval(True)` mode, or exported via ONNX.

### Function: `sima_export_onnx(qat_model, inputs, output_file, input_names, output_names, device)`

This function exports a finalized QAT model to ONNX format.

Args:
    qat_model: The finalized ML model to export to ONNX.
    inputs: a `Tuple` of tensor inputs used to infer the proper shapes of all internal tensors.
        This is used by the Pytorch ONNX exporter.
    output_file: the path name of the .onnx file to generate.
    input_names: a list of tensor names used to label the ONNX model inputs.
    output_names: a list of tensor names used to label the ONNX model outputs.
    device: optional device to restore the returned model to after CPU ONNX export.
        If unset, the model returns to its original device.

### Class: `SimaQatWrapper`

This is a Sima-defined wrapper which allows Pytorch GraphModule objects to behave
like `nn.Module`s at training time. It is used so that commonly called Pytorch functions
work correctly when QAT is invoked.

Note:
    This wrapper can only be created from an existing GraphModule. The source GraphModules
    are created by Pytorch at each control point during QAT runtime.

### Function: `check_graph_nodes(prepared_mod, device)`

Checks the prepared model for inconsistent device paramterers and
also for setting the dropout layers to inactive mode

### Function: `replace_dropout(m)`

No docstring available.

### Function: `replace_batchnorm(m)`

FX Graph rewriter to replace a flavor of batchnorm with one that can be exported
