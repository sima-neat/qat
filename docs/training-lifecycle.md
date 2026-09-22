---
title: Training lifecycle
sidebar_position: 2
---

# Training lifecycle

The recommended lifecycle is prepare, observer warm-up, freeze, recovery
training, checkpoint, finalization, and ONNX export.

## Prepare

Prepare from the eager PyTorch module before constructing the optimizer. The
returned graph is an isolated copy; preparation does not mutate or move the
source module or example inputs.

```python
import torch
from sima_qat import sima_prepare_qat_model

source_model = MyModel()
example_inputs = (torch.randn(1, 3, 224, 224),)

qat_model = sima_prepare_qat_model(
    source_model,
    example_inputs,
    device="cuda",
)
optimizer = torch.optim.AdamW(qat_model.parameters(), lr=1e-5)
```

Example shapes are static by default. Set `dynamic_batch=True` only when the
model genuinely supports different leading batch sizes:

```python
qat_model = sima_prepare_qat_model(
    source_model,
    example_inputs,
    device="cuda",
    dynamic_batch=True,
)
```

This option changes training capture only. ONNX export uses the concrete shapes
of the inputs supplied to `sima_export_onnx`.

## Warm up and freeze

Run the ordinary forward, loss, backward, and optimizer steps. During this
period observers collect activation ranges while fake quantization models the
effect of INT8 execution.

For pretrained models, BatchNorm statistics can be frozen independently while
observers continue to update:

```python
from sima_qat import sima_freeze_batchnorm_stats

sima_freeze_batchnorm_stats(qat_model)
```

After observer warm-up, lock the activation grids and shift-aware weight
scales, then continue training for one or more recovery epochs:

```python
from sima_qat import sima_freeze_qat

sima_freeze_qat(qat_model)
# Continue the normal training loop with the same optimizer.
```

Repeated freeze calls are safe. Finalization can freeze automatically, but an
explicit freeze followed by recovery training generally produces better
accuracy.

## Checkpoint and resume

Save the prepared model's state dictionary before finalization. Observer state,
the frozen marker, and the exact quantization parameters used by finalization
and ONNX export are included.

```python
torch.save(
    {
        "model": qat_model.state_dict(),
        "optimizer": optimizer.state_dict(),
    },
    "qat-checkpoint.pt",
)
```

To resume, reconstruct the same eager model, prepare it with the same example
input structure, and then load the saved model and optimizer states:

```python
checkpoint = torch.load("qat-checkpoint.pt", map_location="cpu")

qat_model = sima_prepare_qat_model(
    MyModel(),
    example_inputs,
    device="cuda",
)
optimizer = torch.optim.AdamW(qat_model.parameters(), lr=1e-5)

qat_model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])
```

A checkpoint written after `sima_freeze_qat` resumes with observers and scales
still frozen. A checkpoint written before it resumes observer collection.

## Finalize and export

Finalization converts the trained graph to inference-only form. It is
idempotent, but the resulting model is not intended for further training.

```python
from sima_qat import sima_export_onnx, sima_finalize_qat_model

qat_model = sima_finalize_qat_model(qat_model)
sima_export_onnx(
    qat_model,
    example_inputs,
    "model.qdq.onnx",
    input_names=["input"],
    output_names=["output"],
    device="cuda",
)
```

The exporter writes opset-17 ONNX. Validate the artifact before handing it to
another tool:

```bash
python -c "import onnx; model = onnx.load('model.qdq.onnx'); onnx.checker.check_model(model)"
```

Compare the finalized PyTorch model and ONNX Runtime on representative inputs
as part of the model's accuracy validation. The exported QDQ graph is the
handoff to Model Compiler; QAT does not assign graph regions to hardware or
rewrite the graph for a particular compiler backend.
