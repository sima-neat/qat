# SiMa QAT

Quantization-aware training for SiMa.ai hardware, designed to run inside the
existing Model Compiler Python environment.

The backend uses PyTorch FX graph-mode QAT and the legacy TorchScript ONNX
exporter. It does not use Dynamo, torch.export, capture_pre_autograd_graph,
prepare_qat_pt2e, or convert_pt2e. This is required by the released Model
Compiler stack:

- Python 3.12.3
- PyTorch 2.3.1
- torchvision 0.18.1
- NumPy 1.26.4
- ONNX 1.17.0
- ONNX Runtime 1.21.1
- PyTorch Lightning 2.4.0

The runtime is also regression-tested with PyTorch 2.8/torchvision 0.23, but
the installable artifact targets the Model Compiler versions above.

## Install with Model Compiler

Install and activate Model Compiler first, then install the QAT Neat artifact:

```bash
sima-cli neat install model-compiler
sima-cli neat install qat/amd64
activate-model-compiler
```

Use `qat/arm64` on arm64 systems. The QAT installer:

1. locates the existing Model Compiler virtual environment;
2. validates the exact Python version and shared package baseline;
3. validates the QAT wheel without resolving dependencies;
4. removes the legacy `swml-qat` distribution, which owns the same
   `sima_qat` package namespace;
5. installs `sima-qat` with `--no-deps`;
6. runs `pip check` and a functional prepare/train/finalize/export/ORT smoke.

It never creates a QAT venv and never upgrades Python, Torch, torchvision, or
another Model Compiler dependency.

Model Compiler installation replaces its entire virtual environment. Reinstall
QAT after every Model Compiler reinstall or upgrade.

For source development, activate Model Compiler and run tests directly from
this checkout. Avoid a normal `pip install .`, because dependency resolution
must not modify the shared stack.

```bash
activate-model-compiler
cd /path/to/qat
python -m pytest -m smoke
```

`setup_env.sh` remains available only for the separate PyTorch 2.8 reference
test environment used by QAT contributors.

## API

The lifecycle remains prepare, train, finalize, export:

```python
import torch
from sima_qat.qat_api import (
    sima_export_onnx,
    sima_finalize_qat_model,
    sima_prepare_qat_model,
)

model = ...
example_inputs = (torch.randn(1, 3, 224, 224),)

qat_model = sima_prepare_qat_model(model, example_inputs, device="cuda")

# Fine-tune qat_model with the existing optimizer and training loop.

qat_model = sima_finalize_qat_model(qat_model)
sima_export_onnx(
    qat_model,
    example_inputs,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    device="cuda",
)
```

| Function | Purpose |
|---|---|
| `sima_prepare_qat_model(model, inputs, device)` | Symbolically trace the module, apply supported FX fusion, remove dropout, and insert signed-int8 activation fake quantization plus per-channel weight observers. |
| `sima_finalize_qat_model(qat_model)` | Freeze observers, quantize weights, fold Conv-BN, and lock the graph in inference mode. |
| `sima_export_onnx(qat_model, inputs, output_file, ...)` | Export a standard opset-17 Q/DQ ONNX model without Dynamo and restore the requested model device. |

Preparation preserves the original `Parameter` objects. Optimizers created
before preparation continue to update the prepared model, which is required by
the supplied PyTorch Lightning examples.

The quantization contract remains compatible with the earlier backend:

- activations: signed INT8, per-tensor affine, `[-128, 127]`;
- weights during training: per-channel observation only;
- weights at finalization: signed INT8, per-channel symmetric,
  `[-127, 127]`, axis 0;
- bias: floating point;
- dropout: removed before observer insertion.

Finalized models reject `train(True)`. Checkpoints contain `qat_state` and an
FX-backend schema version. Legacy PT2E checkpoints are rejected with a clear
error because their graph and state-dictionary layouts are incompatible.

The ONNX artifact has float inputs/outputs, scalar activation Q/DQ parameters,
and INT8 weight initializers feeding axis-0 `DequantizeLinear` nodes. Export
fails rather than emitting an unresolved per-channel `QuantizeLinear`, custom
operator domain, or redundant direct DQ-to-Q edge.

## Traceability boundary

FX symbolic tracing requires static Python control flow. A model whose forward
constructs tensors from symbolic Python sizes, such as an unwrapped custom
stochastic-depth implementation, is rejected with an actionable preparation
error. Standard torchvision ResNet, DenseNet, MobileNet, EfficientNet, and
SqueezeNet families are covered by validation.

A future ATen/make_fx fallback would be a separate backend: it needs its own
operator table, name mapping, fusion behavior, and checkpoint schema.

## Examples

The MNIST and ImageNet examples use the same public lifecycle through
PyTorch Lightning hooks:

```bash
activate-model-compiler

cd examples/mnist
python train.py -e 2 -b 32 --device cpu --download --export-on-end
python test_onnx.py

cd ../imagenet
python train.py --model resnet18 -d data/imagenette2-160 \
  -b 64 --device cuda --samples-limit 64 --epochs 1 \
  --export-on-end
```

Use `--disable-qat` for a float baseline.

## Validation

Run the fast lifecycle and operator tests in both stacks:

```bash
activate-model-compiler
python -m pytest -m smoke
python -m pytest tests --ignore=tests/end_to_end

source .venv/bin/activate
python -m pytest tests --ignore=tests/end_to_end
```

The local acceptance suite covers optimizer identity, gradients, fusion, signed
quantization, dropout, residual/concat/slice regions, checkpoint stages,
standard ONNX Q/DQ, ONNX Runtime, and ResNet export.

Run the opt-in ONNX compilation tests from an activated Model Compiler
environment. The pre-QAT test applies compiler PTQ to the float ONNX model; the
post-QAT test lowers the exported Q/DQ parameters and verifies their
arithmetic-folded representation before checking the generated MPK archive.

```bash
activate-model-compiler
SIMA_QAT_RUN_MODEL_COMPILER_TESTS=1 \
SIMA_QAT_MODEL_COMPILER_TARGET=modalix \
python -m pytest -q -s -o addopts= \
  -n 0 \
  -m model_compiler tests/model_compiler
```

Use `mlsoc` instead of `modalix` to select the Gen1 target. Release
qualification must run these compile regressions on both supported host
architectures.

Generated ONNX models, graph dumps, checkpoints, datasets, and Lightning logs
are gitignored.

## Build the Neat artifact

```bash
activate-model-compiler
./scripts/build_qat_bundle.sh \
  --target-arch amd64 \
  --output-dir dist/amd64
```

The bundle contains the QAT wheel, strict shared-environment manifest,
installer, and installed-environment smoke test.

## Layout

```text
sima_qat/
  qat_api.py            # prepare, finalize, export, lifecycle/checkpoint state
  sima_quantizer.py     # Dynamo-free FX QConfig and backend policies
  onnx_ops.py           # standard Q/DQ weight folding and validation
examples/               # MNIST and ImageNet workflows
tests/                  # lifecycle, operator, ONNX, and end-to-end tests
scripts/
  install_qat_wheels.sh # installs into Model Compiler with --no-deps
  smoke_test_qat.py     # installed-environment functional acceptance test
  source.json           # immutable shared-environment contract
```
