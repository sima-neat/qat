# Sima QAT

Quantization-Aware Training (QAT) APIs for preparing PyTorch models for SiMa.ai hardware.

It wraps PyTorch's PT2E quantization flow (`prepare_qat_pt2e` / `convert_pt2e`) with a SiMa-specific
quantizer and ONNX exporter, so a standard `nn.Module` can be fine-tuned with fake-quantization and
exported to an INT8 Q/DQ ONNX graph.

**Supported PyTorch:** 2.3.x through 2.8.x (Python ≥ 3.10).

## Setup

```bash
# install uv (one time)
curl -LsSf https://astral.sh/uv/install.sh | sh
# restart your shell, or add uv to PATH for this shell:
export PATH="$HOME/.local/bin:$PATH"

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
from sima_qat.qat_api import (
    sima_prepare_qat_model,
    sima_freeze_qat,
    sima_finalize_qat_model,
    sima_export_onnx,
)

model = ...                                  # any torch.nn.Module
example_inputs = (torch.randn(1, 3, 224, 224),)

# 1. Insert fake-quant scaffolding. Shift-aware QAT is enabled by default.
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

To reproduce the observer-only weight behavior from earlier releases, opt out during preparation:

```python
qat_model = sima_prepare_qat_model(
    model, example_inputs, device='cuda', shift_aware=False
)
```

| function | purpose |
|---|---|
| `sima_prepare_qat_model(model, inputs, device, shift_aware=True)` | Capture the model and insert SiMa fake-quant annotations. Power-of-two-aware weight QAT is the default; pass `False` for the legacy behavior. |
| `sima_freeze_qat(qat_model)` | Freeze observers and lock AFE-compatible weight scales before final fine-tuning. |
| `sima_finalize_qat_model(qat_model)` | Fold the trained scaffolding into an inference-only quantized graph. |
| `sima_export_onnx(qat_model, inputs, output_file, ...)` | Export the finalized model to an ONNX QuantizeLinear/DequantizeLinear graph. |

## Examples

[examples/mnist](examples/mnist) and [examples/imagenet](examples/imagenet) are runnable
PyTorch-Lightning workflows that wire the API into the `on_train_start` (prepare) /
`on_train_end` (finalize) / `on_fit_end` (export) hooks. Each has the same four scripts:
`*_lit.py` (the Lightning module holding the QAT calls), `train.py`, `export_onnx.py`, and
`test_onnx.py`. Run them from inside the example directory; use `--help` for all options.

**MNIST** — small, from-scratch, CPU-friendly; downloads the dataset on demand:

```bash
cd examples/mnist
python train.py -e 2 -b 32 --device cpu --download --export-on-end  # train (QAT) + export ONNX
python test_onnx.py                                                  # eval the exported ONNX
python export_onnx.py                                                # re-export latest checkpoint
```

**ImageNet / Imagenette** - QAT fine-tunes a pretrained torchvision classifier. For a
small smoke test, use the ImageNet-style Imagenette subset:

```bash
cd examples/imagenet
mkdir -p data
curl -L -o data/imagenette2-160.tgz https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz
tar -xzf data/imagenette2-160.tgz -C data

python train.py --model resnet18 -d data/imagenette2-160 -b 64 --device cuda \
  --samples-limit 64 --epochs 1 --workers 4 --export-on-end
python export_onnx.py --model resnet18 --device cuda
python test_onnx.py --onnx exported_ckpt_resnet18_onnx_model.onnx \
  --dsroot data/imagenette2-160 --split val --samples-limit 50
```

For ImageNet/Imagenette, `--samples-limit` selects a class-balanced subset rather
than the first N sorted files. When the dataset is the 10-class Imagenette subset,
the example also remaps WNID folder labels to the matching ImageNet-1K output
indices during both training and ONNX validation.

For the full ImageNet-2012 dataset, point `-d` / `--dsroot` at a directory containing
`train/<class>/...` and `val/<class>/...`.

Pass `--disable-qat` to either `train.py` to train a plain float baseline instead of QAT.

During ResNet/ImageNet finalization, PyTorch may print repeated
`erase_node(batch_norm_*) on an already erased node` warnings from PT2E batchnorm cleanup.
They are noisy but non-fatal when the run continues to `post_p2e_graph.txt`, ONNX export,
and ONNXRuntime validation.

## Tests

```bash
pytest -m smoke
pytest
tox -e smoke
```

- `tests/test_smoke.py` -- fast import/package checks for validating a built or installed QAT wheel.
- `tests/*.py` — fast, synthetic graph-structure unit tests (one QAT op pattern each; no training).
- `tests/end_to_end/` — full-model QAT runs on CIFAR10 (DenseNet, ResNet50) gated on accuracy. The
  ResNet50 test trains on a GPU and is **skipped when no CUDA device is available**.

Generated ONNX models, graph dumps, checkpoints, example datasets, and Lightning logs are gitignored.

## Neat Artifact Package

QAT extension bundles are published to Vulcan as independent Neat artifacts:

```bash
# amd64 host
sima-cli neat install qat/amd64
# arm64 host
sima-cli neat install qat/arm64

# tagged or branch-specific versions
sima-cli neat install qat/amd64@v1.0.0
sima-cli neat install qat/amd64@develop
```

The installer creates a separate QAT virtual environment under `/sdk-extensions/qat`,
`/sdk-add-on/qat`, or `~/sdk-extensions/qat`, and adds `activate-qat` / `deactivate-qat` shell helpers.

```bash
./scripts/build_qat_bundle.sh --target-arch amd64 --output-dir dist/amd64
./scripts/generate_api_docs.py
```

Generated API reference docs live under `docs/generated/` and are linked from the main software docs site.

## Layout

```text
sima_qat/                         # Python package
  qat_api.py                      # public API: prepare / finalize / export
  sima_quantizer.py               # SiMa PT2E quantizer
  onnx_ops.py                     # custom ONNX symbolic functions for Q/DQ ops
  misc.py                         # shared helper utilities
examples/                         # MNIST and ImageNet training + export examples
tests/                            # unit, smoke, and end_to_end tests
docs/generated/                   # generated API reference docs
skills/qat_prepare_export/        # QAT workflow skill/playbook
scripts/                          # bundle, install, metadata, and docs generation scripts
.github/workflows/                # GitHub Actions publishing workflow
setup.py                          # Python package metadata/build entry point
setup_env.sh                      # one-shot venv + dependency bootstrap
```
