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

Add SiMa QAT to your existing PyTorch training project. Keep its dataset,
preprocessing, augmentations, loss, optimizer settings, and validation metric.

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

QDQ nodes describe the conversion between floating-point values and INT8
values in the exported ONNX graph.

## Install

The QAT wheel requires Python 3.10 or newer and PyTorch 2.8.x. Install it in the
environment that already contains the model's training dependencies.

Download the QAT package with `sima-cli`:

```bash
sima-cli neat install qat
```

The command downloads the wheel and installs or refreshes the QAT coding-agent
skill for Codex and Claude. It does not change the active Python environment.
Activate the training environment and install the downloaded wheel:

```bash
python -m pip install ./sima_qat-*.whl
python -c "import torch, sima_qat; print(torch.__version__, sima_qat.__file__)"
```

## Add QAT to a training project

The following steps form one continuous workflow. Adapt the model, data,
optimizer, loss, and validation calls to the existing training project.

### 1. Prepare the model

Prepare the model before constructing the optimizer. Preparation returns an
isolated QAT graph; it does not modify or move the source model or example
inputs. The input tuple must match the model's positional inputs, dtypes, and
shapes. The `device` argument selects the training device for the returned QAT
graph. Keep training and validation batches on that device.

```python
import torch
from sima_qat import (
    sima_export_onnx,
    sima_finalize_qat_model,
    sima_freeze_qat,
    sima_prepare_qat_model,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Recreate the model and load the floating-point checkpoint.
source_model = MyModel()
source_model.load_state_dict(torch.load("model-fp32.pt", map_location="cpu"))
source_model.train()

example_inputs = (torch.randn(1, 3, 224, 224),)
qat_model = sima_prepare_qat_model(
    source_model,
    example_inputs,
    device=device,
)

optimizer = torch.optim.AdamW(qat_model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()
```

### 2. Train, freeze, and recover

During warm-up, observers measure activation ranges while fake quantization
already simulates INT8. Freezing stops range and scale updates, **not weight
training**. Keep training afterward so the weights can recover accuracy with
those fixed settings.

Validation must keep fake quantization enabled and observers disabled so
held-out data cannot change the measured ranges. `eval()` alone does not stop
observers. This classification helper restores the previous mode and observer
states, including observers already disabled by freezing. Adapt the metric
for detection or other tasks.

```python
from torch.ao.quantization import disable_observer
from torch.ao.quantization.fake_quantize import FakeQuantizeBase

def validate(model, loader, device):
    was_training = model.training
    observer_states = [
        (module, module.observer_enabled.clone())
        for module in model.modules()
        if isinstance(module, FakeQuantizeBase)
    ]
    correct = total = 0
    try:
        model.eval()
        model.apply(disable_observer)
        with torch.inference_mode():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                predictions = model(images).argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.numel()
    finally:
        model.train(was_training)
        for module, enabled in observer_states:
            module.observer_enabled.copy_(enabled)
    return correct / total

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

    accuracy = validate(qat_model, validation_loader, device)
    print(f"Epoch {epoch}: validation accuracy {accuracy:.2%}")
```

The example uses two warm-up epochs and two recovery epochs. Treat this as a
starting point: choose the freeze epoch and recovery duration from validation
results.

### 3. Save a checkpoint

Save the prepared model before finalization so training can be resumed. A
checkpoint should contain both the QAT model and optimizer state.

```python
from pathlib import Path

checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(parents=True, exist_ok=True)
torch.save(
    {
        "epoch": epoch,
        "model": qat_model.state_dict(),
        "optimizer": optimizer.state_dict(),
    },
    checkpoint_dir / f"qat-{epoch:02d}.pt",
)
```

**To resume later (optional)**, recreate and prepare the same model with the
same example-input and batch contract before loading the saved states:

```python
checkpoint = torch.load("checkpoints/qat-03.pt", map_location="cpu")

source_model = MyModel()
source_model.load_state_dict(torch.load("model-fp32.pt", map_location="cpu"))
qat_model = sima_prepare_qat_model(
    source_model,
    example_inputs,
    device=device,
)
optimizer = torch.optim.AdamW(qat_model.parameters(), lr=1e-5)

qat_model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])
start_epoch = checkpoint["epoch"] + 1
```

The checkpoint preserves observer state, the frozen state, and the exact
quantization parameters used by finalization and ONNX export.

### 4. Finalize and export

Finalization converts the trained QAT model into an inference-only graph;
export writes that graph to ONNX. Finalized models cannot resume training.

```python
final_model = sima_finalize_qat_model(qat_model)
accuracy = validate(final_model, validation_loader, device)
sima_export_onnx(
    final_model,
    example_inputs,
    "model.qdq.onnx",
    input_names=["images"],
    output_names=["predictions"],
)
```

| Operation | Device behavior |
|---|---|
| Prepare | Uses your selected training device. |
| Finalize | Stays on the QAT model's device. |
| Export | Uses CPU temporarily, then restores the model even on failure. Inputs stay unchanged. |

Omit export's `device` argument to keep the model's device. An explicit value
moves the PyTorch model after successful export; it does not select the ONNX
runtime device.

## Batch size

Preparation keeps the leading tensor dimension dynamic by default. This lets
the same prepared graph train with ordinary data-loader batch sizes, handle a
short final batch, and export with the concrete batch size passed to
`sima_export_onnx`.

Most models need no batch option. Set `dynamic_batch=False` only when the model
intentionally requires the exact example batch size:

```python
qat_model = sima_prepare_qat_model(
    source_model,
    example_inputs,
    device=device,
    dynamic_batch=False,
)
```

Fixed-batch capture is appropriate when the model asserts or branches on batch
size, uses fixed-size recurrent state, or folds batch into direction, channel,
or other layout geometry. Dynamic capture compares the captured output with
the original model on the supplied example and fails with guidance to disable
dynamic batch when it would change the model's behavior.

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

Pass the exported QDQ ONNX model to Model Compiler on CPU. Import,
partitioning, optimization, and hardware assignment are separate compilation
steps.

## Runnable examples

The repository's [examples](https://github.com/sima-neat/qat/tree/main/examples)
include a small CPU-friendly MNIST workflow, ImageNet fine-tuning of a
pretrained classifier, and a plain-PyTorch YOLO26n workflow with checkpoint
resume and export.
