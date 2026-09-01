# SiMa.ai QAT

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

The recommended customer workflow is **prepare → calibrate → freeze target
grids → ordinary PyTorch training → validate → export**. The returned object is a normal
`torch.nn.Module`, so it works with standard optimizers and training loops.

```python
import torch
import sima_qat

model = ...
example_inputs = (torch.randn(1, 3, 224, 224),)

qat = sima_qat.prepare(
    model,
    example_inputs,
    target="modalix",
    device="cuda",
)

# Uses the qualified recipe minimum. The state-space recipe consumes at least
# 128 representative batches and checks power-of-two shift-tier stability.
qat.calibrate(calibration_loader)
qat.calibration_report.raise_for_failure()

# Lock the exact power-of-two target grids before the first optimizer update.
# QAT performed only before this call trains a different numerical function.
qat.freeze()

optimizer = torch.optim.AdamW(qat.parameters(), lr=1e-5)
total_steps = len(train_loader)
for step, (images, labels) in enumerate(train_loader):
    optimizer.zero_grad()
    quant_strength = qat.curriculum(
        step,
        total_steps,
        # Optional for difficult recurrent graphs: elementwise QDrop is
        # training-only and decays to strict Q/DQ before validation/export.
        dropout_probability=0.5,
        dropout_decay_fraction=0.75,
    )
    prediction = qat(images)
    task_loss = criterion(prediction, labels)

    # feature_loss= is optional and may match any model-specific intermediate
    # teacher/student features. FP32 output preservation is always included.
    optional_feature_loss = None
    loss = qat.loss(task_loss, feature_loss=optional_feature_loss)
    loss.backward()
    optimizer.step()

# Optional for difficult recurrent graphs. This evaluates existing Q/DQ range
# candidates on a held-out, re-iterable validation loader and keeps a change
# only when the requested task metric improves. It adds no graph operators.
range_report = qat.refine_ranges(
    validation_loader,
    evaluate_task,
    metric="d1",
)

report = qat.validate(validation_loader, evaluator=evaluate_task)
report.raise_for_failure()

bundle = qat.export("build/model_int8")
print(bundle.onnx_path, bundle.onnx_sha256)
```

`recipe="auto"` is the default. It selects `strict_int8_ssm` when Mamba,
selective-scan, SS2D, or TinyVim modules are present and `strict_int8`
otherwise. Model-specific qualification can pin a built-in or data-only YAML
recipe:

```python
qat = sima_qat.prepare(
    model,
    example_inputs,
    target="modalix",
    device="cuda",
    recipe="strict_int8_ssm",
)
```

`qat.summary()` reports lifecycle state, detected state-space regions, fake
quantizer counts, and weighted-operator coverage. `qat.explain()` explains the
selected policy. Export creates `model.onnx` and `qat_manifest.json`; the
manifest binds the ONNX checksum and clearly distinguishes host QAT/ONNX
validation from stock Model Compiler and board evidence.

`qat.refine_ranges(...)` is an optional post-QAT recovery step. Its automatic
policy is intentionally narrow: it searches positive unit-grid Multiply
outputs inside recurrent/state-space regions, excludes grids directly coupled
to Conv/Linear operators, and transactionally restores the original ranges if
the task metric does not improve. The winning grid is persisted through the
PT2E observer contract and the full search receipt is included in
`qat_manifest.json`. Use `ActivationRangeSelector` only when qualifying a
broader model family.

The session keeps its frozen FP32 teacher out of `parameters()` and
`state_dict()`, so it does not double optimizer or checkpoint size. Loading a
session checkpoint restores the frozen-grid lifecycle state.

### Low-level compatibility API

Framework integrations can continue to use the original PT2E control-point
functions. They remain backward compatible:

```python
from sima_qat.qat_api import (
    sima_prepare_qat_model,
    sima_freeze_qat,
    sima_finalize_qat_model,
    sima_export_onnx,
)

qat_model = sima_prepare_qat_model(model, example_inputs, device="cuda")
# observer warm-up and training ...
sima_freeze_qat(qat_model)
# fine-tune locked grids ...
qat_model = sima_finalize_qat_model(qat_model)
sima_export_onnx(qat_model, example_inputs, "model.onnx", device="cuda")
```

Shift-aware QAT constrains each convolution or linear weight scale so the
Model Compiler can use its native integer shift requantization without
rescaling learned INT8 weight codes. Freezing explicitly leaves time to
fine-tune against locked scales; low-level finalization still freezes
automatically as a compatibility fallback.

### DepthART dynamic-INT8 recurrence extension

DepthART's selective scan can use the qualified `q8 + p64` recurrence without
exposing compiler details in the training loop. The model remains ordinary
PyTorch; a compatible SS2D module supplies
`enable_depthart_dynamic_p64_scan_()` and
`freeze_depthart_dynamic_p64_()` methods.

```python
import torch
from sima_qat import (
    enable_depthart_dynamic_p64_fake_quant,
    freeze_depthart_dynamic_p64,
    prepare_depthart_dynamic_p64,
    write_depthart_dynamic_p64_compile_profile,
)

model = load_depthart().cuda().eval()
prepared = prepare_depthart_dynamic_p64(
    model,
    safety_margin=1.05,
    observe=True,
)

# Observe real recurrence ranges without quantizing the forward pass.
enable_depthart_dynamic_p64_fake_quant(model, False)
with torch.no_grad():
    for image in calibration_loader:
        model(image.cuda())

# Freeze base grids, then train/evaluate the exact integer forward with STE.
contracts = freeze_depthart_dynamic_p64(model)
enable_depthart_dynamic_p64_fake_quant(model, True)
fine_tune_with_task_loss(model, train_loader)

example = torch.zeros(1, 3, 448, 576, device="cuda")
torch.onnx.export(model.eval(), (example,), "depthart.onnx", opset_version=17)
profile = write_depthart_dynamic_p64_compile_profile(
    model, "depthart.onnx", "depthart.depthart_p64.json"
)
```

Preparation automatically splits DepthART's independent state into physical
C128 blocks and removes only the first chunk's exact `a * 0 + b` identity.
The latter is real-math equivalent and prevents size-dependent constant
folding from invalidating source bindings. Profile creation rejects a
provably-zero state product if a custom model bypasses this architecture rule.

The ONNX remains a standard Q/DQ + Mul/Add graph. The companion profile is
content-bound to the exact ONNX bytes and names every recurrence boundary;
the SiMa Model Compiler fails closed if a binding, static grid, or carrier ABI
does not match. Host recurrence accuracy is an isolation result, not silicon
evidence—claim full strict INT8 only after the compiled precision audit and
board evaluation pass.

To reproduce the observer-only weight behavior from earlier releases, opt out during preparation:

```python
qat_model = sima_prepare_qat_model(
    model, example_inputs, device='cuda', shift_aware=False
)
```

| function | purpose |
|---|---|
| `sima_qat.prepare(model, inputs, target="modalix", recipe="auto")` | Return the recommended lifecycle-managed QAT `nn.Module`. |
| `qat.calibrate(data, batches=None)` | Collect activation ranges without fake quantization. Omitted `batches` uses the recipe's qualified minimum; calibration records ordered sample IDs and shift-tier stability. |
| `qat.curriculum(step, total_steps, dropout_probability=None, dropout_decay_fraction=None)` | Progressively introduce activation rounding and optionally decay training-only QDrop to zero. Weights and frozen target grids remain strict. |
| `qat.loss(task_loss, feature_loss=None)` | Add optional intermediate-feature preservation and scale-normalized frozen-FP32 output preservation. |
| `qat.refine_ranges(data, evaluator, metric=..., factors=(1, 2, 4))` | Optionally search export-stable activation ranges on held-out task data; commit only an improvement and otherwise roll back. |
| `qat.freeze()` / `qat.validate()` / `qat.export(directory)` | Lock target grids before training, enforce structural gates, and create a content-bound ONNX bundle. |
| `sima_prepare_qat_model(model, inputs, device, shift_aware=True)` | Capture the model and insert SiMa fake-quant annotations. Power-of-two-aware weight QAT is the default; pass `False` for the legacy behavior. |
| `sima_freeze_qat(qat_model)` | Freeze observers and lock AFE-compatible weight scales before final fine-tuning. |
| `sima_finalize_qat_model(qat_model)` | Fold the trained scaffolding into an inference-only quantized graph. |
| `sima_export_onnx(qat_model, inputs, output_file, ...)` | Export the finalized model to an ONNX QuantizeLinear/DequantizeLinear graph. |

The quantizer has regression coverage for Conv2d, ConvTranspose2d, Linear,
Add, Multiply, average/max pooling, BatchNorm, Sigmoid, SiLU, Erf, slicing,
selection, unsqueeze, and Concat. Repeated-input Concat uses one shared INT8
grid so aligned C1-to-C16 output publication does not introduce a floating
requantization. Unsupported operators remain visible in the exported graph and
must be resolved in the model architecture before claiming strict INT8.

ONNX export runs on CPU by default. Use `export_device="cuda"` only when
reproducing an already-qualified accelerator-side constant-folding contract;
`device` controls where the returned model is restored after export.

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

## Neat artifact package

QAT extension bundles are published to Vulcan as independent SiMa.ai Neat artifacts:

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
  session.py                      # recommended lifecycle-managed customer API
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
