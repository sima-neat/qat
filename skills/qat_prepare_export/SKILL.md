---
name: sima-qat-prepare-export
description: Use when installing SiMa QAT into the shared Model Compiler environment; preparing, training, checkpointing, or finalizing PyTorch models with the Dynamo-free FX backend; exporting and validating standard Q/DQ ONNX; compiling pre-QAT or post-QAT ONNX for MLSoC or Modalix; locating MPK artifacts; or troubleshooting QAT CUDA and Model Compiler test environments.
---

# Use SiMa QAT with Model Compiler

## Keep the shared environment intact

Install Model Compiler and the QAT artifact for the host architecture when
needed, then activate the shared environment:

```bash
sima-cli neat install model-compiler
sima-cli neat install qat/amd64  # use qat/arm64 on arm64
activate-model-compiler
```

Target the released shared stack: Python 3.12.3, Torch 2.3.1,
torchvision 0.18.1, NumPy 1.26.4, ONNX 1.17.0, ONNX Runtime 1.21.1,
and PyTorch Lightning 2.4.0.

- Do not create a QAT venv.
- Do not run a normal `pip install .` or let pip upgrade shared packages.
- Reinstall QAT after any Model Compiler reinstall or upgrade; that operation
  replaces the Model Compiler venv.
- Treat Torch 2.8/torchvision 0.23 as a contributor control stack, not the
  installable artifact target.

## Apply the lifecycle

Use the public API from `sima_qat.qat_api`:

```python
import torch
from sima_qat.qat_api import (
    sima_export_onnx,
    sima_finalize_qat_model,
    sima_prepare_qat_model,
)

model = ...
example_inputs = (torch.randn(1, 3, 224, 224),)

qat_model = sima_prepare_qat_model(model, example_inputs, device="cpu")
# Fine-tune qat_model with the existing optimizer/training loop.
qat_model = sima_finalize_qat_model(qat_model)
sima_export_onnx(
    qat_model,
    example_inputs,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    device="cpu",
)
```

Preserve these invariants:

- Pass representative inputs as a tuple.
- When an optimizer already exists, prepare that original model rather than a
  deep copy. Preparation preserves `Parameter` identities so its optimizer
  remains valid.
- Train only the prepared scaffold. Finalization freezes observers and
  weights, folds Conv-BN regions, and permanently rejects `train(True)`.
- Rebuild an identically prepared scaffold before loading a prepared-state
  checkpoint.
- Reject legacy PT2E checkpoints; their graphs and state dictionaries are not
  compatible with this FX backend.
- Require FX-symbolically-traceable forward logic. Do not silently fall back to
  Dynamo, `capture_pre_autograd_graph`, `prepare_qat_pt2e`, or `convert_pt2e`.

The quantization contract is signed INT8 per-tensor affine activations
`[-128, 127]`, per-channel observed weights during training, finalized
per-channel symmetric weights `[-127, 127]` on axis 0, floating-point bias,
and dropout removal before observer insertion.

## Validate exported ONNX

Export only a finalized model. Require a standard opset-17 graph with:

- float inputs and outputs;
- scalar activation `QuantizeLinear`/`DequantizeLinear` parameters;
- INT8 weight initializers feeding axis-0 `DequantizeLinear` nodes;
- no unresolved per-channel weight `QuantizeLinear`;
- no custom operator domain or direct redundant DQ-to-Q edge.

Use ONNX Runtime with graph optimizations disabled for strict exporter parity.
Run optimized ORT as a secondary shape/finite-output check; compiler-versus-ORT
numeric tolerance belongs to the post-QAT compilation gate below.

## Choose the validation gate

Run fast installed/lifecycle coverage first:

```bash
activate-model-compiler
python -m pytest -q -rs -m smoke tests/integration
python -m pytest -q tests/integration
```

Run the CUDA-only full-CIFAR10 ResNet50 gate only when the current environment
actually exposes an NVIDIA device:

```bash
python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'
CUDA_VISIBLE_DEVICES=0 python -m pytest -q -rs tests/end_to_end/test_resnet50.py
```

Host `nvidia-smi` success is insufficient for a container. For a locally
managed Docker SDK container, absence of `/dev/nvidia*` means it must be
recreated with GPU allocation such as `--gpus all`. A remote Modalix DevKit
cannot inherit the laptop GPU. Changing the venv cannot expose hardware.

## Compile pre-QAT and post-QAT ONNX

Run the opt-in compiler tests serially from the activated Model Compiler venv:

These tiny compiler gates are CPU tests and do not require CUDA.

```bash
SIMA_QAT_RUN_MODEL_COMPILER_TESTS=1 \
SIMA_QAT_MODEL_COMPILER_TARGET=modalix \
python -m pytest -q -s \
  --basetemp=build/model-compiler-pytest \
  -m model_compiler \
  tests/acceptance/model_compiler
```

Use `mlsoc` for the Gen1 target and `modalix` for Gen2. The two tests prove
different paths:

1. `pre_qat` exports float ONNX, applies Model Compiler PTQ, and compiles it.
2. `post_qat` exports Q/DQ ONNX, verifies that compiler lowering preserves QAT
   activation parameters and arithmetic-folded weight quantization, compares
   compiler execution with ONNX Runtime within two output LSBs, and compiles it.

Use the normal public `load_model(...).quantize(...)` compiler path for both
tests; on post-QAT ONNX this lowers the existing Q/DQ contract. Do not use the
deprecated `is_quantized=True` importer, strip Q/DQ nodes, or substitute a
different PTQ-only path merely to make compilation pass.

The compiler tests use pytest temporary paths. `--basetemp` makes the location
predictable but pytest clears it at the beginning of the next run. Locate MPK
archives with:

```bash
find build/model-compiler-pytest -type f -name '*_mpk.tar.gz' -print
```

Expect `pre_qat_mpk.tar.gz` and `post_qat_mpk.tar.gz`, plus their `.sima` and
ONNX intermediates. Copy release evidence elsewhere before the next run.

Never type a documented `...` placeholder literally as a pytest path.

## Qualify changes

- Run the narrow affected regression before the full non-end-to-end suite.
- Run both compile tests for Model Compiler integration changes.
- Qualify `modalix` and `mlsoc` and both supported host architectures before a
  release; a single local Modalix run is development evidence only.
- Build the Neat bundle with `scripts/build_qat_bundle.sh` and verify the
  installed-environment smoke outside the source checkout so `sima_qat/` from
  the checkout cannot shadow the installed wheel.
- Keep generated ONNX, `.sima`, MPK, graph dumps, datasets, and logs out of the
  source package and Git history unless explicitly requested as fixtures.
