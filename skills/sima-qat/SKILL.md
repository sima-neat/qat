---
name: sima-qat
description: Add SiMa QAT to an existing PyTorch training project, resume QAT checkpoints, and validate QDQ ONNX exports. Use for quantization-aware fine-tuning, not model compilation, graph surgery, or deployment.
---

# Use SiMa QAT

Integrate QAT into the user's existing training project. Preserve its model,
data, preprocessing, augmentations, loss, optimizer policy, scheduler, and
task metric unless the user asks to change them. Prefer pretrained weights.

Requires Python 3.10+ and PyTorch 2.8.x. If missing, download the package with
`sima-cli neat install qat`, then install the wheel in the training environment.
Respect that environment's constraints when choosing a PyTorch build.

Use the [QAT user guide](https://github.com/sima-neat/qat/blob/main/docs/index.md)
for working code, validation, and checkpoint examples.

## Training workflow

1. Create the eager model and load pretrained weights when available. Build
   an example-input tuple matching its positional inputs, shapes, and dtypes.
2. Call `sima_prepare_qat_model(source_model, example_inputs, device)`.
   Construct the optimizer afterward from the returned QAT model's parameters.
   Train and validate that model with batches on the same device; use CUDA
   when available. Preparation leaves the source model and inputs unchanged.
3. Train normally to warm up observers. Fake quantization is already enabled
   during warm-up. Track the project's validation metric.
4. Call `sima_freeze_qat(qat_model)` and continue training to recover accuracy.
   Freezing locks ranges and scales; weights remain trainable. Choose the
   freeze point from validation evidence and leave time for recovery.
5. Save the prepared model and optimizer state before finalization.
6. Call `sima_finalize_qat_model(qat_model)`, then `sima_export_onnx` with the
   finalized model and example inputs. Validate it on the same device as
   training. It is inference-only and cannot resume training.

Finalization runs on the model's existing device and returns an inference model
on that device.
Export temporarily uses CPU and restores the supplied model's device and graph
device settings, even on failure; caller inputs stay unchanged. Usually omit
export's `device` argument. An explicit value selects the PyTorch model's
device after successful export, not the ONNX runtime device.

## Validation and checkpoints

During validation, keep fake quantization enabled and disable observers.
`eval()` and `inference_mode()` alone do not stop observers. Preserve and
restore each fake-quantizer's `observer_enabled` buffer and the model's
previous mode in `finally`; observers disabled by freezing must stay disabled.
Use the validation helper in the guide.

Compare the task metric for the floating-point model, QAT before freezing,
QAT after recovery, finalized PyTorch model, and ONNX Runtime output when
available. Run `onnx.checker.check_model`; verify opset 17 and QDQ nodes where
annotations apply. Judge numerical differences with output-appropriate
tolerances and the task metric.

To resume, recreate the same eager model and floating-point starting weights.
Prepare it with the same example-input structure and batch policy, construct
its optimizer, then restore both state dictionaries. The checkpoint preserves
observers, freeze state, and scales.
Use the prepared checkpoint rather than finalized artifacts.

## Batch size and failures

The leading dimension is a dynamic batch by default. Set `dynamic_batch=False`
only for an intentional fixed-batch contract, such as batch-size assertions,
fixed recurrent state, or batch folded into other layout dimensions. If dynamic
capture reports changed output semantics, verify that contract before opting out.

For preparation or freeze errors, identify the affected model operation.
Preserve the error; do not bypass freeze or edit private observer state. If
finalization warns that scales were not frozen, recommend an explicit freeze
and recovery phase for better accuracy.

For annotation questions, consult the
[operator manifest](https://github.com/sima-neat/qat/blob/main/sima_qat/operator_manifest.py).
Unannotated operations can remain trainable; missing QDQ alone is not an
integration failure. Compilation and hardware assignment are separate workflows.

Use [examples](https://github.com/sima-neat/qat/tree/main/examples) as patterns:
`mnist` for a small workflow, `imagenet` for classifier fine-tuning, and
`yolo26n_pytorch_qat` for detector training with resume and export.

Report changes, the freeze schedule, checkpoint behavior, and validation
results. State any checks that could not be run and why.
