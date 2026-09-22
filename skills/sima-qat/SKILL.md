---
name: sima-qat
description: Integrate, train, resume, freeze, finalize, and export PyTorch models with the SiMa QAT package. Use when adding INT8 quantization-aware training or validating QAT checkpoints and QDQ ONNX exports. Do not use for model compilation, graph surgery, or deployment.
---

# Use SiMa QAT

Adapt QAT to the user's existing PyTorch training project. Preserve its model,
dataset, preprocessing, augmentations, loss, optimizer policy, scheduler, and
task metric unless the user asks to change them. Prefer a pretrained
floating-point checkpoint when one is available.

SiMa QAT supports Python 3.10 or newer and requires PyTorch 2.8.x. If the
package is missing, download it with `sima-cli neat install qat`, then install
the downloaded wheel in the model's training environment. Do not replace that
environment or its PyTorch build without checking the project's constraints.

## Inspect the project

Before editing, locate:

- model construction and floating-point checkpoint loading;
- the model's positional input structure, shapes, and dtypes;
- optimizer construction, training and validation loops, and checkpointing;
- the task-level accuracy metric;
- any existing ONNX export or runtime comparison path.

Use the closest example from the
[QAT examples](https://github.com/sima-neat/qat/tree/main/examples) for
additional context: `mnist` for a small workflow, `imagenet` for classifier
fine-tuning, or `yolo26n_pytorch_qat` for a plain-PyTorch detector with resume
and export. Treat examples as patterns, not replacements for the user's
training code.

## Integrate the lifecycle

Import only the public API:

```python
from sima_qat import (
    sima_export_onnx,
    sima_finalize_qat_model,
    sima_freeze_qat,
    sima_prepare_qat_model,
)
```

Follow these lifecycle invariants:

1. Recreate the eager model and load its floating-point weights.
2. Build an input tuple that matches the model's positional inputs, shapes,
   and dtypes.
3. Call `sima_prepare_qat_model(source_model, example_inputs, device)` before
   constructing the optimizer. Train the returned graph, not the source model.
4. Run normal training to collect observer ranges. Measure the project's real
   validation metric during this warm-up.
5. Call `sima_freeze_qat(qat_model)` after observer warm-up and continue
   training for at least one recovery phase with locked quantization grids.
6. Save a resumable checkpoint before finalization, including the prepared
   model and optimizer state dictionaries.
7. After training, move the QAT model and example inputs to CPU, call
   `sima_finalize_qat_model`, and export with `sima_export_onnx`. Finalization
   is inference-only.

Choose the freeze point from validation evidence. A reasonable initial
experiment warms observers for most of a short fine-tuning run and reserves
one or more later epochs for recovery. If accuracy falls sharply at freeze,
freeze earlier and allow more recovery rather than changing quantization
internals.

## Preserve batch semantics

Preparation captures example shapes exactly by default. Keep a fixed training
batch, using `drop_last=True` when appropriate, unless the model is genuinely
batch-polymorphic.

Use `dynamic_batch=True` only when the leading tensor dimension is an ordinary
batch dimension. Do not enable it for models that fold batch into recurrence,
direction, channel, or layout geometry. A dynamic training capture still
exports the concrete shapes supplied to `sima_export_onnx`.

## Resume training

Recreate the same eager model, load the same floating-point starting weights,
and prepare it with the same example-input structure and batch policy. Build
the optimizer from the prepared graph, then restore the prepared model and
optimizer state dictionaries. Preserve whether QAT was already frozen; the
checkpoint contains observer state, freeze state, and the qparams later used
for finalization and ONNX export.

Do not resume from a finalized model. Keep finalized artifacts separate from
training checkpoints.

## Validate each boundary

Use representative validation data and record the task metric for:

- the original floating-point model;
- the prepared model before freeze;
- the prepared model after freeze and recovery;
- the finalized PyTorch model;
- the exported ONNX model in ONNX Runtime, when available.

Run `onnx.checker.check_model` on the exported model. Confirm that the export
uses opset 17 and contains `QuantizeLinear` and `DequantizeLinear` nodes where
QAT annotations apply. Compare finalized PyTorch and ONNX Runtime outputs with
output-appropriate tolerances, then use the task metric for the acceptance
decision.

Integer result tensors such as `ArgMax` and `TopK` indices must remain integer.
PReLU, ConvTranspose, Embedding/Gather, GridSample, ReduceMin, and CumSum are
trainable but intentionally unannotated in the current package. Do not report
their lack of QDQ annotation as a QAT integration failure.

## Handle failures narrowly

- If preparation fails, reduce the case to the model region or PyTorch export
  construct that failed; do not rewrite the ONNX graph as part of this skill.
- If explicit dynamic-batch capture is rejected, return to static capture
  unless the user wants to change the model's batch semantics.
- If freeze reports incomplete or invalid quantization parameters, preserve
  the failure and identify the affected weighted operation. Do not bypass
  freeze or edit private observer state.
- If finalization warns that QAT was not frozen, stop and add an explicit
  freeze and recovery phase instead of relying on finalization to lock scales.
- If the request moves to ONNX compatibility, compiler partitioning, graph
  surgery, compilation, or deployment, hand off the exported QDQ ONNX model to
  the corresponding workflow. Do not change QAT annotations merely to obtain a
  particular compiler assignment.

## Deliver the result

Report the files changed, the selected example-input and batch contract, the
observer warm-up and freeze schedule, checkpoint/resume behavior, and
validation results at each completed lifecycle boundary. State which
validations could not be run and why.

## Sources

Use the maintained
[QAT user guide](https://github.com/sima-neat/qat/blob/main/docs/index.md) as
the primary lifecycle reference. Consult source only when the guide does not
resolve an integration or diagnostic question:

- [`sima_qat/__init__.py`](https://github.com/sima-neat/qat/blob/main/sima_qat/__init__.py)
  defines the supported public imports.
- [`sima_qat/qat_api.py`](https://github.com/sima-neat/qat/blob/main/sima_qat/qat_api.py)
  defines lifecycle behavior, validation, and error handling.
- [`sima_qat/operator_manifest.py`](https://github.com/sima-neat/qat/blob/main/sima_qat/operator_manifest.py)
  is the source of truth for QAT annotation and export behavior. It does not
  describe compiler partitioning or hardware assignment.
