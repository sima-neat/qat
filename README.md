# SiMa QAT

SiMa QAT provides quantization-aware training for SiMa.ai hardware from the
existing Model Compiler Python environment. It does not create or own a second
customer virtual environment.

The backend uses PyTorch FX graph-mode QAT and the legacy TorchScript ONNX
exporter. It deliberately avoids Dynamo, `torch.export`,
`capture_pre_autograd_graph`, `prepare_qat_pt2e`, and `convert_pt2e` so it can
run with the released Model Compiler stack:

- Python 3.12.3
- PyTorch 2.3.1
- torchvision 0.18.1
- NumPy 1.26.4
- ONNX 1.17.0
- ONNX Runtime 1.21.1
- PyTorch Lightning 2.4.0

A separate contributor control profile exercises PyTorch 2.8 and torchvision
0.23. The customer artifact continues to target the versions above.

## Install with Model Compiler

Install Model Compiler and the QAT artifact for the host architecture, then
activate the shared environment:

```bash
sima-cli neat install model-compiler
sima-cli neat install qat/amd64
activate-model-compiler
```

Use `qat/arm64` on an arm64 host. The QAT installer:

1. locates the existing Model Compiler virtual environment;
2. validates Python and the shared package baseline;
3. validates the QAT wheel without resolving dependencies;
4. removes the legacy `swml-qat` distribution, which owns the same
   `sima_qat` package namespace;
5. installs `sima-qat` with `--no-deps`; and
6. runs `pip check` plus a functional prepare/train/finalize/export/ORT smoke.

It never creates a QAT virtual environment and never upgrades Python, Torch,
torchvision, or another Model Compiler dependency. Reinstall QAT after every
Model Compiler reinstall or upgrade because Model Compiler replaces its whole
virtual environment.

For source development, activate Model Compiler and run the checkout directly.
Do not use a normal `pip install .`; dependency resolution must not modify the
shared stack.

```bash
activate-model-compiler
cd /path/to/qat
python -m pytest -q -m smoke tests/integration
```

## Repository path contract

Run documented commands from the repository root. Script defaults are resolved
from their source location, not from the caller's current working directory.

| Root | Purpose | Examples |
|---|---|---|
| `data/` | Reusable, local dataset and input caches | `data/mnist/`, the torchvision CIFAR-10 cache in `data/`, and an explicitly selected ImageNet root |
| `build/` | Disposable test, example, tool, graph, checkpoint, log, ONNX, and report output | `build/pytest/`, `build/test-results/`, `build/examples/`, `build/tools/` |
| `dist/<arch>/` | Publishable QAT extension bundles only | `dist/amd64/`, `dist/arm64/` |

Pytest recreates its selected base temporary directory at the start of a run.
Copy evidence that must be retained into the external qualification system
before rerunning the same command. Do not use `dist/` for compiler-test MPKs or
other scratch files.

## API

The public lifecycle is prepare, train, finalize, then export:

```python
from pathlib import Path

import torch
from sima_qat import (
    sima_export_onnx,
    sima_finalize_qat_model,
    sima_prepare_qat_model,
)

model = ...
example_inputs = (torch.randn(1, 3, 224, 224),)

qat_model = sima_prepare_qat_model(model, example_inputs, device="cpu")

# Fine-tune qat_model with the existing optimizer and training loop.

qat_model = sima_finalize_qat_model(qat_model)
output = Path("build/exports/model.onnx")
output.parent.mkdir(parents=True, exist_ok=True)
sima_export_onnx(
    qat_model,
    example_inputs,
    output,
    input_names=["input"],
    output_names=["output"],
    device="cpu",
)
```

| Function | Purpose |
|---|---|
| `sima_prepare_qat_model(model, inputs, device)` | Symbolically trace the module, apply supported FX fusion, remove dropout, and insert signed-INT8 activation fake quantization plus per-channel weight observers. |
| `sima_finalize_qat_model(qat_model)` | Freeze observers, quantize weights, fold Conv-BN regions, and lock the graph in inference mode. |
| `sima_export_onnx(qat_model, inputs, output_file, ...)` | Export an opset-17 Q/DQ ONNX model without Dynamo and restore the requested model device. |

`sima_qat.__version__` reports the installed Python wheel version sourced from
`VERSION.in`; it is distinct from the architecture bundle metadata version.

Preparation preserves the original `Parameter` objects. An optimizer created
before preparation therefore continues to update the prepared model, as used
by the PyTorch Lightning examples.

The quantization contract is:

- activations: signed INT8, per-tensor affine, `[-128, 127]`;
- weights during training: per-channel observation only;
- weights at finalization: signed INT8, per-channel symmetric, `[-127, 127]`,
  axis 0;
- bias: floating point; and
- dropout: removed before observer insertion.

Finalized models reject `train(True)`. Checkpoints contain `qat_state` and an
FX-backend schema version. Legacy PT2E checkpoints are rejected because their
graph and state-dictionary layouts are incompatible.

The exported ONNX model has float inputs and outputs, scalar activation Q/DQ
parameters, and INT8 weight initializers feeding axis-0 `DequantizeLinear`
nodes. Export fails instead of emitting an unresolved per-channel
`QuantizeLinear`, a custom operator domain, or a redundant direct DQ-to-Q edge.

## Traceability boundary

FX symbolic tracing requires static Python control flow. Preparation rejects a
model whose `forward` constructs tensors from symbolic Python sizes or uses
other non-traceable control flow, and reports an actionable error. The
integration suite covers the supported fusion and quantization regions; the
end-to-end suite covers a compact DenseNet topology and a CUDA-only full
ResNet50 topology. Qualify every additional customer architecture explicitly.

An ATen or `make_fx` fallback would be a separate backend with its own operator
table, name mapping, fusion behavior, checkpoint schema, and validation.

## Examples

The examples use the public lifecycle through PyTorch Lightning hooks and keep
all generated output below `build/examples/`.

MNIST:

```bash
activate-model-compiler
python -m examples.mnist.train \
  --data data/mnist \
  --output-dir build/examples/mnist \
  --epochs 2 --batch 32 --device cpu --download --export-on-end
python -m examples.mnist.test_onnx \
  --onnx build/examples/mnist/exports/mnist_qat.onnx \
  --dsroot data/mnist
```

ImageNet-style data requires explicit `train/` and `val/` directories:

```bash
python -m examples.imagenet.train \
  --model resnet18 \
  --data /path/to/imagenet \
  --output-dir build/examples/imagenet \
  --batch 64 --device cuda --samples-limit 64 --epochs 1 \
  --export-on-end
python -m examples.imagenet.test_onnx \
  --onnx build/examples/imagenet/exports/resnet18_qat.onnx \
  --dsroot /path/to/imagenet --split val --samples-limit 50
```

Use `--disable-qat` for a float baseline. See
[`examples/mnist/README.md`](examples/mnist/README.md) and
[`examples/imagenet/README.md`](examples/imagenet/README.md) for checkpoint and
standalone-export commands.

The optional CIFAR-10 distillation utility uses `torch.compile` and therefore
runs only in the disposable Torch 2.8 contributor profile. Its `torch-kmeans`
and `transformers` dependencies are deliberately excluded from the customer
QAT bundle and shared Model Compiler environment:

This contributor workflow requires `uv` on `PATH` and package-index network access.

```bash
./setup_env.sh build/venvs/torch28-control
uv pip install \
  --python build/venvs/torch28-control/bin/python \
  -r requirements-distill.txt
build/venvs/torch28-control/bin/python scripts/distill_cifar10.py \
  --data-dir data \
  --output build/tools/distill_cifar10/mini_samples.json \
  --device cpu \
  --allow-download
```
The first run needs `--allow-download`; later runs use only the existing
CIFAR-10 and TinyCLIP caches. The supplied Torch 2.8 control profile is CPU-only;
a separately qualified CUDA control environment is required for `--device cuda`.

## Validation

Run the lifecycle and graph integration suite inside Model Compiler:

```bash
activate-model-compiler
python -m pytest -q -m smoke tests/integration
python -m pytest -q tests/integration
```

The default integration base directory is `build/pytest/integration`, its cache is
`build/pytest-cache`, and tox writes JUnit reports to `build/test-results`.
The integration suite covers optimizer identity, gradients, fusion, signed
quantization, dropout, residual/concat/slice regions, checkpoint stages,
standard ONNX Q/DQ, and ONNX Runtime. End-to-end training gates are explicit;
see [`tests/README.md`](tests/README.md).

Pull requests run the exact Torch 2.3 Model Compiler baseline, the Torch 2.8
control suite, generated-document checks, customer CLI checks, and a staged
bundle installer smoke. Candidate packaging and every release/publish job are
skipped on pull requests. Protected branches and semantic release tags use the same
validation before candidate construction; publication still requires the external
qualification and legal gates described below.

Run the opt-in compiler tests serially from an activated Model Compiler
environment. The pre-QAT test applies compiler PTQ to float ONNX; the post-QAT
test lowers exported Q/DQ parameters, checks their arithmetic-folded
representation, compares compiler execution with ONNX Runtime, and compiles an
MPK:

```bash
activate-model-compiler
SIMA_QAT_RUN_MODEL_COMPILER_TESTS=1 \
SIMA_QAT_MODEL_COMPILER_TARGET=modalix \
python -m pytest -q -s \
  --basetemp=build/pytest/model-compiler/modalix \
  -m model_compiler \
  tests/acceptance/model_compiler

find build/pytest/model-compiler/modalix \
  -type f -name '*_mpk.tar.gz' -print
```

Use `mlsoc` instead of `modalix` and a matching
`build/pytest/model-compiler/mlsoc` base directory for the Gen1 target. Release
qualification must run both targets on both supported host architectures; a
single local run is development evidence only.

For the contributor-only PyTorch 2.8 CPU control profile, create a disposable
environment below `build/`:

```bash
./setup_env.sh build/venvs/torch28-control
build/venvs/torch28-control/bin/python -m pytest -q tests/integration
```

This profile is not an installation path for Model Compiler customers.

## Build the Neat artifact

Build from an activated Model Compiler environment. The default output is the
canonical architecture directory:

```bash
activate-model-compiler
./scripts/build_qat_bundle.sh --target-arch amd64
# publishable files: dist/amd64/
```

Use `--target-arch arm64` for `dist/arm64/`. `VERSION.in` is the Python wheel
version. The Neat bundle metadata version is derived independently from a
release tag when present, otherwise from the SDK version, branch, and Git hash;
release automation may set it explicitly with `--bundle-version`.

A development candidate contains six base resources: one QAT wheel, the strict
shared-environment manifest, the installer, the installed-environment smoke test,
metadata, and the wheel inventory. A publishable release additionally requires,
includes, and checksums the approved LICENSE and NOTICE files; release automation
refuses placeholder or missing legal artifacts.

## Layout

```text
sima_qat/
  qat_api.py            # prepare, finalize, export, and lifecycle state
  sima_quantizer.py     # Dynamo-free FX QConfig and backend policies
  onnx_ops.py           # standard Q/DQ weight folding and validation
examples/               # MNIST and ImageNet-style workflows
scripts/
  build_qat_bundle.sh   # architecture-specific publishable bundle
  install_qat_wheels.sh # shared Model Compiler installation with --no-deps
  smoke_test_qat.py     # installed-environment functional acceptance
  distill_cifar10.py    # optional CIFAR-10 mini-index utility
tests/
  integration/          # default lifecycle and graph contracts
  end_to_end/           # explicit CIFAR training/export/accuracy gates
  acceptance/
    model_compiler/     # opt-in pre-QAT and post-QAT compilation
  README.md             # suite taxonomy, data gates, and artifact locations
```
