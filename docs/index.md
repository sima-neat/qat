---
title: Quantization-Aware Training
sidebar_label: Quantization-aware training
sidebar_position: 1
---

# Quantization-Aware Training

Quantization-aware training (QAT) fine-tunes a PyTorch model while simulating
the numerical effects of INT8 inference. Use it when post-training quantization
causes an unacceptable accuracy loss and you can retrain the model with
representative data.

SiMa QAT is designed to run inside the model's existing PyTorch training
project. It does not replace the dataset, augmentations, loss function,
optimizer, or validation metric. Keeping those parts of the original project
is important because they are usually what made the floating-point model
accurate in the first place.

Start from a pretrained floating-point checkpoint when possible. Training from
random initialization is supported, but usually requires substantially more
time and data.

## How QAT works

Preparation adds observers and fake-quantization operations to the model.
Observers measure activation ranges, while fake quantization rounds and clamps
values during the forward pass to approximate INT8 execution. Tensors,
gradients, and optimizer updates remain floating point, allowing training to
adapt the model weights to those quantization effects.

The workflow is:

1. **Prepare** the eager PyTorch model for QAT.
2. **Warm up** observers using the normal training loop.
3. **Freeze** the activation ranges and SiMa-compatible weight scales.
4. **Recover** accuracy by continuing to train with the locked scales.
5. **Finalize** the model for inference.
6. **Export** a standard opset-17 ONNX model containing
   `QuantizeLinear` and `DequantizeLinear` (QDQ) nodes.

## Install

The QAT wheel requires Python 3.10 or newer and PyTorch 2.8.x. Install it in the
environment that already contains the model's training dependencies.

Download the QAT package with `sima-cli`:

```bash
sima-cli neat install qat
```

The command downloads the wheel without changing the active Python
environment. Activate the training environment and install the downloaded
wheel:

```bash
python -m pip install ./sima_qat-*.whl
python -c "import torch, sima_qat; print(torch.__version__, sima_qat.__file__)"
```

## Add QAT to a training project

Prepare the model before constructing the optimizer. Preparation returns an
isolated QAT graph; it does not modify or move the source model or example
inputs.

```python
from pathlib import Path

import torch
from sima_qat import (
    sima_export_onnx,
    sima_finalize_qat_model,
    sima_freeze_batchnorm_stats,
    sima_freeze_qat,
    sima_prepare_qat_model,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Recreate the model and load the floating-point checkpoint.
source_model = MyModel()
source_model.load_state_dict(torch.load("model-fp32.pt", map_location="cpu"))
source_model.train()

# The tuple must match the model's positional inputs, dtypes, and shapes.
example_inputs = (torch.randn(1, 3, 224, 224),)
qat_model = sima_prepare_qat_model(
    source_model,
    example_inputs,
    device=device,
)

# Build the optimizer from the prepared model, not the source model.
optimizer = torch.optim.AdamW(qat_model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()

# Optional for pretrained models: keep their learned BatchNorm statistics.
# QAT observers continue to collect activation ranges.
sima_freeze_batchnorm_stats(qat_model)

freeze_epoch = 2
num_epochs = 4

for epoch in range(num_epochs):
    qat_model.train()

    # Reserve one or more later epochs for recovery training.
    if epoch == freeze_epoch:
        sima_freeze_qat(qat_model)

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        predictions = qat_model(images)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

    validate(qat_model, validation_loader, device)

    # Save before finalization so training can be resumed.
    torch.save(
        {
            "epoch": epoch,
            "model": qat_model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        Path("checkpoints") / f"qat-{epoch:02d}.pt",
    )

# Finalization is inference-only. Export on CPU after training has finished.
final_model = sima_finalize_qat_model(qat_model.cpu())
export_inputs = tuple(value.cpu() for value in example_inputs)
sima_export_onnx(
    final_model,
    export_inputs,
    "model.qdq.onnx",
    input_names=["images"],
    output_names=["predictions"],
    device="cpu",
)
```

The freeze epoch is model-dependent. A useful starting point is to warm up
observers for most of a short fine-tuning run and reserve at least one final
epoch for recovery. Track validation accuracy before and after freezing. If
accuracy drops sharply, freeze earlier and allow more recovery training.

For pretrained models, freezing BatchNorm statistics early often avoids
overwriting useful running statistics during a short QAT run. Models trained
from scratch or on a substantially different data distribution may benefit
from allowing BatchNorm to adapt during observer warm-up instead.

Checkpoints preserve observer state, the frozen state, and the exact
quantization parameters used by finalization and ONNX export. To resume,
recreate and prepare the same model with the same example-input structure,
construct the optimizer, and load both state dictionaries.

## Batch size

Preparation captures the example shapes exactly by default. Keep the QAT batch
size fixed, using `drop_last=True` when necessary, or explicitly opt into a
dynamic leading batch dimension:

```python
qat_model = sima_prepare_qat_model(
    source_model,
    example_inputs,
    device=device,
    dynamic_batch=True,
)
```

Use dynamic batch only when the model is genuinely batch-polymorphic. The
package validates the capture and rejects models where batch size participates
in recurrence, direction, channel, or other layout geometry. This option
affects the training graph; ONNX export uses the concrete shapes passed to
`sima_export_onnx`.

## Validate and compile

Measure the task-level metric for the original floating-point model, the
prepared model before and after freezing, the finalized PyTorch model, and the
ONNX model. This makes it clear which lifecycle step introduced a regression.

Validate the exported file before compilation:

```bash
python - <<'PY'
import onnx

model = onnx.load("model.qdq.onnx")
onnx.checker.check_model(model)
print("ONNX model is valid")
PY
```

Compare the finalized PyTorch model with ONNX Runtime on representative
validation samples. Small elementwise differences can occur at quantization
boundaries, so use tolerances appropriate to the outputs and confirm the
model's real accuracy metric.

Common weighted, activation, normalization, pooling, reduction, and
shape/layout operations are covered. `ArgMax` and `TopK` keep index outputs as
integers. PReLU, ConvTranspose, Embedding/Gather, GridSample, ReduceMin, and
CumSum remain trainable but are not QAT-annotated in this release.

The exported QDQ ONNX model is the handoff to Model Compiler. Import,
partitioning, optimization, and hardware assignment are separate compilation
steps.

## Runnable examples

The repository's [examples](https://github.com/sima-neat/qat/tree/main/examples)
include a small CPU-friendly MNIST workflow, ImageNet fine-tuning of a
pretrained classifier, and a plain-PyTorch YOLO26n workflow with checkpoint
resume and export.
