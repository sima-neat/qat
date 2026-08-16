# Test suite

Directories define test scope. Markers describe runtime capabilities; they do
not replace the hierarchy or encode a release schedule.

## Layout

| Path | Contract | Invocation |
|---|---|---|
| `integration/lifecycle/` | Public API, optimizer identity, checkpoints, stage guards, and device handling | Default suite |
| `integration/graph/` | FX fusion, quantization boundaries, ONNX Q/DQ, and ONNX Runtime | Default suite |
| `integration/examples/` | Deterministic example artifact and path helpers | Default suite |
| `end_to_end/` | CIFAR-10 training, finalization, export, and accuracy | Explicit slow run |
| `acceptance/model_compiler/` | Real pre-QAT and post-QAT compiler artifacts | Opt-in Model Compiler run |

`pytest` defaults to `tests/integration`. There is no `unit/` directory because
the current tests exercise full QAT components rather than isolated pure
functions.

## Markers

| Marker | Meaning |
|---|---|
| `smoke` | Smallest installed API and lifecycle checks |
| `slow` | Long-running training or compilation |
| `gpu` | A visible CUDA device is required |
| `network` | The test may download an explicitly permitted missing cache |
| `model_compiler` | An activated Model Compiler environment is required |

Strict marker validation is enabled in `pytest.ini`. Add a marker there before
using a new one.

## Integration commands

Run the default suite inside the Model Compiler environment:

```bash
activate-model-compiler
python -m pytest -q
```

Run only the two smallest lifecycle checks:

```bash
python -m pytest -q -rs -m smoke tests/integration
```

The equivalent isolated contributor environments are:

```bash
python -m tox -e smoke
python -m tox -e integration
```

## End-to-end commands

End-to-end tests never change the process working directory. They read a
reusable CIFAR-10 cache and write model outputs to the test's pytest temporary
directory.

To permit the first download into the canonical cache:

```bash
SIMA_QAT_TEST_DATA_DIR=data \
SIMA_QAT_ALLOW_DATA_DOWNLOAD=1 \
python -m pytest -q -s \
  --basetemp=build/pytest/end-to-end/densenet \
  tests/end_to_end/test_densenet.py
```

Once cached, omit `SIMA_QAT_ALLOW_DATA_DOWNLOAD`:

```bash
SIMA_QAT_TEST_DATA_DIR=data \
python -m pytest -q -s \
  --basetemp=build/pytest/end-to-end/densenet \
  tests/end_to_end/test_densenet.py
```

The full ResNet50 gate needs a CUDA device visible to the active Python
process. Host `nvidia-smi` alone does not prove container access.

```bash
python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'

CUDA_VISIBLE_DEVICES=0 \
SIMA_QAT_TEST_DATA_DIR=data \
python -m pytest -q -s -rs \
  --basetemp=build/pytest/end-to-end/resnet50 \
  tests/end_to_end/test_resnet50.py
```

If the cache is missing, add `SIMA_QAT_ALLOW_DATA_DOWNLOAD=1` explicitly. A
missing cache without that opt-in skips instead of mutating the workspace or
using the network unexpectedly.

## Model Compiler acceptance

Run compiler acceptance serially from an activated Model Compiler environment:

```bash
activate-model-compiler
SIMA_QAT_RUN_MODEL_COMPILER_TESTS=1 \
SIMA_QAT_MODEL_COMPILER_TARGET=modalix \
python -m pytest -q -s \
  --basetemp=build/pytest/model-compiler/modalix \
  -m model_compiler \
  tests/acceptance/model_compiler
```

Use `mlsoc` for Gen1 and a matching
`build/pytest/model-compiler/mlsoc` base directory. Both the target variable and
matching final `--basetemp` component are required; acceptance fails before
compilation when either is missing or mislabeled. The tests intentionally
create independent fixtures:

- `test_pre_qat.py` exports float ONNX, applies Model Compiler PTQ, and compiles
  `pre_qat_mpk.tar.gz`;
- `test_post_qat.py` exports Q/DQ ONNX, validates learned quantization through
  compiler lowering and execution, and compiles `post_qat_mpk.tar.gz`.

Locate the archives and their `.sima`/ONNX intermediates with:

```bash
find build/pytest/model-compiler/modalix \
  -type f \
  \( -path '*/pre_qat/*' -o -path '*/post_qat/*' \) \
  -print
```

Pytest clears the selected `--basetemp` directory at the start of its next run.
Export retained evidence to the external qualification system before rerunning.
Compiler MPKs are test output and do not belong in `dist/`.

## Artifact locations

| Path | Contents | Lifetime |
|---|---|---|
| `build/pytest/integration/` | Default integration ONNX and graph scratch | Recreated by each integration run |
| `build/pytest/end-to-end/<suite>/` | Isolated training, checkpoint, and ONNX output | Recreated by the selected E2E command |
| `build/pytest/model-compiler/<target>/` | Target-labeled ONNX, `.sima`, and MPK evidence | Recreated only by that target command |
| `build/pytest-cache/` | Pytest collection and last-failure cache | Disposable |
| `build/pytest/tox/<env>/` | Tox-specific pytest temporary output | Disposable |
| `build/test-results/<env>.xml` | Tox JUnit reports | Disposable/report upload |
| `data/` | Shared CIFAR-10 download cache | Reusable local input |

Tests must use `tmp_path`/`tmp_path_factory` or an explicitly injected output
path. Do not add global `chdir`, implicit current-directory output, or a second
artifact root.
