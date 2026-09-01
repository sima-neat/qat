# `sima_qat.qat_api`

Source: `sima_qat/qat_api.py`

## Public API

### Function: `sima_prepare_qat_model(input_graph, inputs, device, shift_aware, activation_observer, full_range_ste, learn_scales)`

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
    shift_aware: when ``True`` (the default), fake-quantize weights during training and prepare
        them for SiMa's power-of-two requantization. Set this to ``False`` to retain the legacy
        observer-only weight behavior.
    activation_observer: activation range estimator: ``moving_average``,
        ``minmax``, or ``histogram``. The default preserves the installed
        environment's policy.
    full_range_ste: use a full-range straight-through activation fake
        quantizer. This keeps gradients outside the observed INT8 range.
    learn_scales: make activation scales trainable when ``full_range_ste``
        is enabled.

Returns:
    GraphModule: a compiled version of the given graph with QAT annotations, ready to begin training.

### Function: `sima_qat_activation_diagnostics(qat_model, inputs)`

Measure strict fake-quant error at every activation boundary.

The report is a one-forward diagnostic, not calibration. It records the
quantities that determine whether an A8 domain can represent its signal:
RMS, RMSE, SQNR, step/RMS, saturation, and zero-code fraction.

### Function: `sima_qat_activation_sensitivity(qat_model, inputs, objective, *, candidate_names)`

Rank activation grids by first-order task-loss sensitivity.

Local quantization RMSE alone is not a useful optimization priority: a
large error on a masked coordinate or an insensitive residual can matter
less than a small error on an attention probability.  For fake-quantizer
``i`` this diagnostic measures the Taylor term

``|dL/dq_i * (q_i - x_i)|``

under a caller-supplied scalar objective ``L``.  The returned
``taylor_l1`` is the cancellation-free sum over tensor elements and calls;
``taylor_dot_abs`` is the absolute signed first-order loss change.  The
method changes no qparams or model weights and is intended to select a
bounded set of learned activation scales before QAT.

``candidate_names`` should be used for large graphs so only the relevant
model region retains quantization residuals for backward.

### Function: `sima_freeze_qat(qat_model)`

Freeze QAT observers and lock Model Compiler-compatible power-of-two weight scales.

Call this after observer warm-up, then continue fine-tuning with fake quantization
enabled. For a model prepared with ``shift_aware=False``, this only freezes the
existing activation observers and therefore retains the legacy behavior.

### Function: `sima_thaw_qat_scales(qat_model, scale_parameters)`

Thaw learned grids from the authoritative frozen/export scale buffers.

### Function: `sima_project_qat_to_target_grids(qat_model)`

Project live learned grids onto the exact shift-realizable target set.

Learned activation scales and power-of-two Conv/Linear weight scales are a
coupled discrete system.  Training them independently and solving the
coupling only in :func:`sima_freeze_qat` changes the model at freeze time.
This operation runs the same fail-atomic solver used by freeze, then
re-enables precisely the activation scales which were live beforehand.
Calling it after an optimizer step therefore makes the next QAT forward
identical to the grids that a subsequent freeze/export will retain.

Observer updates remain disabled; this is constraint projection, not a new
calibration pass.  The projected activation scale is copied back into the
corresponding ``log_scale`` parameter so optimizer-visible state and the
executable fake quantizer cannot drift apart.

### Function: `sima_finalize_qat_model(qat_model)`

This function takes a QAT scaffolded model which has completed the training regimen and
converts it to an inference-only (via fakequant) form. Once this process is complete, the model
can no longer be trained, and is intended for export via ONNX.

Args:
    qat_model: a trained QAT model to be converted into inference-only form.

Returns:
    GraphModule: an inference-only version of the QAT model, which can be run in Pytorch
        `eval(True)` mode, or exported via ONNX.

### Function: `sima_export_onnx(qat_model, inputs, output_file, input_names, output_names, device, export_device)`

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
    export_device: optional device on which to trace and constant-fold the
        ONNX graph. CPU is the portable default. Set this explicitly when
        reproducing a device-qualified export whose constant-folding
        contract was established on an accelerator.

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
