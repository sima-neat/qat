# Test suite

Directories define test scope. Markers describe runtime requirements or
release schedules; they are not used as a substitute for the hierarchy.

## Layout

| Path | Scope | Normal use |
|---|---|---|
| `integration/lifecycle/` | Public API, optimizer identity, checkpoints, stage guards, and device handling | Default and premerge |
| `integration/graph/` | FX fusion, quantization boundaries, ONNX Q/DQ, and ONNX Runtime | Default and premerge |
| `end_to_end/` | CIFAR training, finalization, export, and accuracy | Explicit slow CPU/GPU runs |
| `acceptance/model_compiler/` | Real pre-QAT and post-QAT compiler artifacts | Opt-in Model Compiler environment |

The default `pytest` command collects only `tests/integration`. The two smoke
tests are a marker-selected subset of lifecycle integration tests. There is no
`unit/` directory yet because the current tests exercise full QAT components
rather than isolated pure functions.

## Commands

Run the default integration suite:

```bash
activate-model-compiler
python -m pytest -q
```

Run only the smallest lifecycle checks:

```bash
python -m pytest -q -rs -m smoke tests/integration
```

Run end-to-end training explicitly:

```bash
python -m pytest -q tests/end_to_end/test_densenet.py
CUDA_VISIBLE_DEVICES=0 python -m pytest -q -rs tests/end_to_end/test_resnet50.py
```

Run compiler acceptance serially:

```bash
activate-model-compiler
SIMA_QAT_RUN_MODEL_COMPILER_TESTS=1 \
SIMA_QAT_MODEL_COMPILER_TARGET=modalix \
python -m pytest -q -s \
  --basetemp=build/model-compiler-pytest \
  -m model_compiler \
  tests/acceptance/model_compiler
```

Use `mlsoc` for Gen1 and `modalix` for Gen2.

## Artifacts and data

- Integration and compiler tests write explicit pytest temporary paths.
- End-to-end tests place relative ONNX and graph outputs under the repository's
  gitignored `exported_models/` directory.
- `tests/end_to_end/data/` is an ignored local CIFAR cache, not a tracked
  fixture. End-to-end tests may need network access when it is empty.
- `scripts/distill_cifar10.py` is a dataset-generation utility, not a test.
- A compiler `--basetemp` directory is cleared at the start of the next run;
  copy MPK release evidence elsewhere when it must be retained.
