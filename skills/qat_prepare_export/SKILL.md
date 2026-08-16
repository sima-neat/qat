---
name: sima-qat-prepare-export
description: Use when installing SiMa QAT into the shared Model Compiler environment; preparing, training, checkpointing, or finalizing PyTorch models with the Dynamo-free FX backend; exporting and validating standard Q/DQ ONNX; compiling pre-QAT or post-QAT ONNX for MLSoC or Modalix; locating MPK artifacts; managing QAT repository data/build/dist paths; or troubleshooting QAT CUDA and Model Compiler test environments.
---

# Use SiMa QAT with Model Compiler

## Preserve the shared environment

Install Model Compiler and the QAT artifact for the host architecture, then
activate the shared environment:

```bash
sima-cli neat install model-compiler
sima-cli neat install qat/amd64  # use qat/arm64 on arm64
activate-model-compiler
```

Target the SiMa 2.1.3 baseline: Python 3.12.3, Torch 2.3.1,
torchvision 0.18.1, NumPy 1.26.4, ONNX 1.17.0, ONNX Runtime 1.21.1, and
PyTorch Lightning 2.4.0.

- Do not create a customer QAT virtual environment.
- Do not run a normal `pip install .` or allow pip to upgrade shared packages.
- Reinstall QAT after any Model Compiler reinstall or upgrade; Model Compiler
  replaces the virtual environment.
- Run customer and contributor commands from the activated SiMa 2.1.3 Model
  Compiler environment; do not maintain a separate QAT or control environment.

## Use the repository path contract

Run repository commands from the repository root and keep paths role-specific:

- store reusable MNIST data in `data/mnist` and torchvision CIFAR-10 data in
  `data` (`data/cifar-10-batches-py`);
- require an explicit root for ImageNet-style input;
- write all disposable pytest, example, tool, graph, checkpoint, log, ONNX,
  `.sima`, MPK, and report output under `build`;
- write only publishable QAT extension bundles under `dist/amd64` or
  `dist/arm64`; and
- never rely on a process-wide `chdir` or an implicit current-directory output.

Treat pytest base directories as replaceable. Export retained evidence to the
external qualification system before rerunning the same base directory.

## Apply the lifecycle

Use the public API from `sima_qat`:

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

Preserve these invariants:

- Pass representative inputs as a tuple.
- Prepare the original model when an optimizer already exists. Preparation
  preserves `Parameter` identities, so that optimizer remains valid.
- Train only the prepared scaffold.
- Finalize once training is complete. Finalization freezes observers and
  weights, folds Conv-BN regions, and permanently rejects `train(True)`.
- Rebuild an identically prepared scaffold before loading a prepared-stage
  checkpoint.
- Reject legacy PT2E checkpoints because their graphs and state dictionaries
  are incompatible with the FX backend.
- Require FX-symbolically-traceable forward logic. Do not fall back silently to
  Dynamo, `capture_pre_autograd_graph`, `prepare_qat_pt2e`, or `convert_pt2e`.

Preserve signed INT8 per-tensor affine activations `[-128, 127]`, per-channel
observed weights during training, finalized per-channel symmetric weights
`[-127, 127]` on axis 0, floating-point bias, and dropout removal before
observer insertion.

## Validate exported ONNX

Export only a finalized model. Require an opset-17 graph with:

- float inputs and outputs;
- scalar activation `QuantizeLinear`/`DequantizeLinear` parameters;
- INT8 weight initializers feeding axis-0 `DequantizeLinear` nodes;
- no unresolved per-channel weight `QuantizeLinear`;
- no custom operator domain; and
- no direct redundant DQ-to-Q edge.

Use ONNX Runtime with graph optimizations disabled for strict exporter parity.
Use optimized CPU ONNX Runtime as a secondary shape and finite-output check.
Apply compiler-versus-ORT numeric tolerance in the post-QAT acceptance gate.

## Choose the validation gate

Run fast lifecycle and graph coverage first:

```bash
activate-model-compiler
python -m pytest -q -rs -m smoke tests/integration
python -m pytest -q tests/integration
```

Run explicit end-to-end gates with the canonical CIFAR cache. Permit a download
only when authorized:

```bash
SIMA_QAT_TEST_DATA_DIR=data \
SIMA_QAT_ALLOW_DATA_DOWNLOAD=1 \
python -m pytest -q -s \
  --basetemp=build/pytest/end-to-end/densenet \
  tests/end_to_end/test_densenet.py
```

Run the CUDA-only full-CIFAR10 ResNet50 gate only when the active Python process
exposes an NVIDIA device:

```bash
python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'
CUDA_VISIBLE_DEVICES=0 \
SIMA_QAT_TEST_DATA_DIR=data \
python -m pytest -q -s -rs \
  --basetemp=build/pytest/end-to-end/resnet50 \
  tests/end_to_end/test_resnet50.py
```

Do not infer container CUDA access from host `nvidia-smi`. Recreate a locally
managed SDK container with GPU allocation such as `--gpus all` when
`/dev/nvidia*` is absent. Do not expect a remote DevKit to inherit a laptop
GPU. Changing a virtual environment cannot expose hardware.

## Compile pre-QAT and post-QAT ONNX

Run the opt-in compiler tests serially from the activated Model Compiler
environment. These small compiler gates are CPU tests and do not require CUDA:

```bash
SIMA_QAT_RUN_MODEL_COMPILER_TESTS=1 \
SIMA_QAT_MODEL_COMPILER_TARGET=modalix \
python -m pytest -q -s \
  --basetemp=build/pytest/model-compiler/modalix \
  -m model_compiler \
  tests/acceptance/model_compiler
```

Use `mlsoc` for Gen1 and a matching
`build/pytest/model-compiler/mlsoc` base directory. Interpret the gates
separately:

1. `pre_qat` exports float ONNX, applies Model Compiler PTQ, and compiles it.
2. `post_qat` exports Q/DQ ONNX, verifies compiler lowering of the QAT
   activation parameters and arithmetic-folded weights, compares compiler
   execution with ONNX Runtime within two output LSBs, and compiles it.

Use the public `load_model(...).quantize(...)` compiler path for both gates. On
post-QAT ONNX, this lowers the existing Q/DQ contract. Do not use the deprecated
`is_quantized=True` importer, remove Q/DQ nodes, or replace the path with a
PTQ-only shortcut.

Locate compiler artifacts with:

```bash
find build/pytest/model-compiler/modalix \
  -type f \
  \( -path '*/pre_qat/*' -o -path '*/post_qat/*' \) \
  -print
```

Expect `pre_qat_mpk.tar.gz` and `post_qat_mpk.tar.gz` with their `.sima` and
ONNX intermediates. Pytest clears the selected base directory before its next
run. Do not place compiler evidence in `dist`.

Never type a documented `...` placeholder literally as a pytest path.

## Qualify and package changes

- Run the narrow affected regression before the complete integration suite.
- Run both compiler gates for Model Compiler integration changes.
- Qualify `modalix` and `mlsoc` on both supported host architectures before a
  release; one local Modalix run is development evidence only.
- Qualify every additional customer architecture explicitly; choosing a
  torchvision model name does not establish compiler support.
- Build publishable bundles with
  `./scripts/build_qat_bundle.sh --target-arch amd64` or `arm64`; accept the
  default `dist/<arch>` destination.
- Validate the installed bundle outside the source checkout so the checkout's
  `sima_qat/` cannot shadow the installed wheel.
- Keep generated models, compiler files, graphs, datasets, and logs out of the
  source package and Git history unless the repository intentionally tracks a
  test fixture.
