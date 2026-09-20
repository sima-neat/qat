# Sima QAT

Quantization-Aware Training (QAT) APIs for preparing PyTorch models for SiMa.ai hardware.

It wraps PyTorch's PT2E quantization flow (`prepare_qat_pt2e` / `convert_pt2e`) with a SiMa-specific
quantizer and ONNX exporter, so a standard `nn.Module` can be fine-tuned with fake-quantization and
exported to an INT8 Q/DQ ONNX graph.

**Required PyTorch:** 2.8.x (the provided environment pins 2.8.0; Python ≥ 3.10).

## Setup

```bash
# install uv (one time)
curl -LsSf https://astral.sh/uv/install.sh | sh
# restart your shell, or:
source $HOME/.local/bin/env

# then from the repo root:
cd /path/to/qat
./setup_env.sh                 # -> .venv, python 3.12
```

`setup_env.sh` accepts optional arguments to override the defaults:

```bash
./setup_env.sh .venv311 3.11   # custom venv dir and python version
```

Once setup completes (look for `SETUP_DONE_OK`), activate the environment:

```bash
source .venv/bin/activate
```

## Usage

The recommended workflow is **prepare → warm up → freeze → fine-tune → finalize → export**.

```python
import torch
from sima_qat import (
    sima_prepare_qat_model,
    sima_freeze_qat,
    sima_finalize_qat_model,
    sima_export_onnx,
)

model = ...                                  # any torch.nn.Module
example_inputs = (torch.randn(1, 3, 224, 224),)

# 1. Insert shift-aware fake-quant scaffolding.
qat_model = sima_prepare_qat_model(model, example_inputs, device='cuda')

# 2. Warm up observers with your normal training loop ...

# 3. Lock AFE-compatible power-of-two scales, then fine-tune for a few more epochs.
sima_freeze_qat(qat_model)
# ... continue training qat_model ...

# 4. Convert to inference-only (fake-quant / INT8) form
qat_model = sima_finalize_qat_model(qat_model)

# 5. Export to an INT8 Q/DQ ONNX graph
sima_export_onnx(qat_model, example_inputs, 'model.onnx', device='cuda')
```

Shift-aware QAT constrains each convolution or linear weight scale so that AFE can use its native
integer shift requantization without rescaling the learned INT8 weight codes. Calling
`sima_freeze_qat` explicitly leaves time to fine-tune against those locked scales. Finalization will
lock them automatically if necessary, but fine-tuning after the explicit call generally gives better
accuracy.

## Operator contract

The QAT operator set follows the forms that awesome-frontend can canonicalize and place on MLA.
Automatic layout conversion may remap a logical Softmax or LayerNorm axis to channel, and converts
supported batched MatMul forms to MLA BatchMatmul/Einsum forms. `any_shape_on_mla` permits supported
operators to retain non-4D ranks; it does not remove operator-specific or precision restrictions.

| family | QAT contract |
|---|---|
| Conv1d, Conv2d, Linear | signed W8A8 with shift-aware per-channel weight locking |
| MatMul, MM, BMM, BAddBMM | activation W8A8; deployment still requires AFE to canonicalize the concrete shape/equation |
| Softmax | W8A8; automatic layout conversion must map its logical axis to MLA channel |
| LayerNorm | W8A8 boundary; AFE-supported rank, axis, batch and channel limits still apply |
| Erf and GELU | Erf and exact (`approximate="none"`) GELU only; tanh GELU is rejected |
| Concat | W8A8; repeated-input and identity-prefix layouts share the payload grid |

GridSample is deliberately not annotated as INT8 because awesome-frontend supports it only in BF16.
Dynamic Embedding/Gather is also excluded because it currently has no general MLA runtime-index
lowering. These require an explicit mixed-precision/offload policy outside this strict INT8 QAT API.

ConvTranspose2d is deferred even though MLA has an INT8 kernel. PyTorch 2.8 rewrites its required
output-channel weight axis from 1 to 0 during PT2E conversion, while the per-tensor fallback produces
a scalar weight QDQ scale that current awesome-frontend constant surgery cannot import. It must not
be advertised until one of those downstream contracts is fixed and covered end to end.

Preparation preserves the exact example shapes by default. Models that are
truly batch-polymorphic can opt in with
`sima_prepare_qat_model(..., dynamic_batch=True)`. The opt-in capture is
validated on the original example and fails closed when batch participates in
folded recurrence, scan-direction, or layout geometry.
This deliberately replaces prior automatic batch-one duplication; existing
three-argument calls remain valid but now capture static shapes.
Preparation captures isolated CPU copies of the module and example-input
pytree, then moves only the returned QAT graph to `device`. The caller's
module, parameters, buffers, training modes, devices, and input tensors are
left unchanged on both successful and failed capture.

| function | purpose |
|---|---|
| `sima_prepare_qat_model(model, inputs, device, *, dynamic_batch=False)` | Capture the model and insert SiMa shift-aware fake-quant annotations. Dynamic training batch is explicit opt-in. |
| `sima_freeze_qat(qat_model)` | Freeze observers and lock AFE-compatible weight scales before final fine-tuning. |
| `sima_finalize_qat_model(qat_model)` | Fold the trained scaffolding into an inference-only quantized graph. |
| `sima_export_onnx(qat_model, inputs, output_file, ...)` | Export the finalized model to an ONNX QuantizeLinear/DequantizeLinear graph. |

## Examples

[examples/mnist](examples/mnist) and [examples/imagenet](examples/imagenet) are runnable
PyTorch-Lightning workflows that wire the API into the `on_train_start` (prepare) /
`on_train_epoch_start` (freeze) / `on_train_end` (finalize) / `on_fit_end` (export) hooks.
By default, they freeze the quantization grids at the start of the final epoch, leaving that epoch
for recovery training. Use `--freeze-epoch N` to select another zero-based epoch, or
`--freeze-epoch -1` to retain the old finalize-only behavior. Each has the same four scripts:
`*_lit.py` (the Lightning module holding the QAT calls), `train.py`, `export_onnx.py`, and
`test_onnx.py`. Run them from inside the example directory; use `--help` for all options.

**MNIST** — small, from-scratch, CPU-friendly; downloads the dataset on demand:

```bash
cd examples/mnist
python train.py -e 2 -b 32 --device cpu --download --export-on-end  # train (QAT) + export ONNX
python test_onnx.py                                                  # eval the exported ONNX
python export_onnx.py                                                # re-export latest checkpoint
```

**ImageNet** — QAT fine-tunes a pretrained torchvision classifier; you must supply the
ImageNet-2012 dataset (`-d`), and pick the architecture with `--model`:

```bash
cd examples/imagenet
python train.py --model resnet18 -d /data/imagenet -b 100 --device cuda --export-on-end
python test_onnx.py --dsroot /data/imagenet --split val
python export_onnx.py --model resnet18
```

Pass `--disable-qat` to either `train.py` to train a plain float baseline instead of QAT.

**YOLO26n** — [examples/yolo26n_pytorch_qat](examples/yolo26n_pytorch_qat) is a
plain-PyTorch COCO QAT workflow. It reads COCO JSON directly, reproduces the
YOLO26 dual-head objective, and exports six QDQ raw heads for native BoxDecode.
Ultralytics is used only by an isolated, one-time checkpoint converter; the
model, data path, loss, optimizer, and training process have no Ultralytics
runtime dependency. See the example README for COCO128 smoke and full COCO 2017
commands.

## Tests

```bash
pytest
```

- `tests/operators/` — matrix-driven coverage for every supported annotation pattern, including
  shift-grid locking, finalization, and representative ONNX Runtime parity.
- `tests/qat/` — scale-locking, BatchNorm, recovery-training, checkpoint, and failure regressions.
- `tests/integration/` — graph-transformation and device-rewrite tests.
- `tests/end_to_end/` — nightly CIFAR10 accuracy runs for DenseNet and ResNet50. ResNet50 requires
  CUDA.

Run only the fast/default regression tiers or the nightly model tests with:

```bash
pytest -m regression tests/operators tests/qat tests/integration
pytest -m nightly tests/end_to_end
```

Generated ONNX models are written to `exported_models/` (gitignored).

## Layout

```
sima_qat/            # the package
  qat_api.py         # public API: prepare / freeze / finalize / export
  sima_quantizer.py  # SiMa PT2E quantizer (annotators, fusion patterns)
  onnx_ops.py        # custom ONNX symbolic functions for Q/DQ ops
examples/            # MNIST and ImageNet training + export examples
tests/               # operator matrix, QAT lifecycle, integration, and nightly model tests
setup_env.sh         # one-shot venv + dependency bootstrap
```
