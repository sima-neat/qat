# MNIST QAT example

This example trains an MNIST classifier with PyTorch Lightning, applies the
public SiMa QAT lifecycle, exports opset-17 Q/DQ ONNX, and validates the model
with CPU ONNX Runtime. It also supports a float training/export baseline.

Run every command below from the repository root. Defaults are derived from the
repository location and do not depend on the process working directory.

## Files

| File | Purpose |
|---|---|
| `mnist_lit.py` | Lightning module and QAT/float export hooks |
| `train.py` | Train, checkpoint, fully resume, and optionally export |
| `export_onnx.py` | Export a selected or latest checkpoint |
| `test_onnx.py` | Check ONNX and measure MNIST top-1 accuracy |

## Path contract

| Path | Contents |
|---|---|
| `data/mnist/` | Reusable torchvision MNIST cache |
| `build/examples/mnist/checkpoints/` | Lightning checkpoints |
| `build/examples/mnist/lightning/` | Lightning run state and logs |
| `build/examples/mnist/graphs/` | FX graph dumps |
| `build/examples/mnist/exports/` | Generated ONNX models |

`data/` is reusable input. Everything below `build/` is disposable output and
is ignored by Git. The Lightning module uses the repository-root
`build/examples/mnist` path even when imported from another working directory;
passing `output_dir` overrides it.

## Train and export

Activate Model Compiler, then run:

```bash
activate-model-compiler
python -m examples.mnist.train \
  --data data/mnist \
  --output-dir build/examples/mnist \
  --epochs 2 \
  --batch 32 \
  --workers 1 \
  --device cpu \
  --download \
  --export-on-end
```

A QAT run writes `build/examples/mnist/exports/mnist_qat.onnx`. Omit
`--download` after the cache exists. Add `--disable-qat` to train and export a
float baseline at `build/examples/mnist/exports/mnist_float.onnx`.

`--epochs`, `--batch`, and `--samples-limit` must be positive;
`--workers` must be nonnegative. Requesting `--device cuda` fails before model
or dataset setup when CUDA is unavailable.

## Resume full training state

```bash
python -m examples.mnist.train \
  --data data/mnist \
  --output-dir build/examples/mnist \
  --epochs 10 \
  --device cpu \
  --resume
```

`--resume` selects the newest checkpoint with the requested `mnist_qat_` or
`mnist_float_` prefix and supplies it to `Trainer.fit(ckpt_path=...)`, so
model tensors, optimizer/scheduler state, epoch, and global step are restored.
`--epochs` is the total target epoch count, not the number of additional
epochs. Use `--disable-qat` to select the float checkpoint namespace; QAT and float
runs under the same output directory cannot cross-resume. The command never searches the
current directory and fails clearly if no checkpoint exists.

## Export a checkpoint

PyTorch Lightning checkpoints use Python pickle. Load only checkpoints from a
trusted source; checkpoint deserialization is not a safe interchange format.

Export the latest checkpoint below the selected output directory on CPU:

```bash
python -m examples.mnist.export_onnx \
  --output-dir build/examples/mnist \
  --device cpu
```

The checkpoint is always mapped through CPU first, including checkpoints saved
from CUDA training. A QAT checkpoint creates
`build/examples/mnist/exports/mnist_checkpoint_qat.onnx`; a float checkpoint
creates `mnist_checkpoint_float.onnx`. Select a checkpoint explicitly for
reproducible automation:

```bash
python -m examples.mnist.export_onnx \
  --ckpt build/examples/mnist/checkpoints/mnist_qat_classifier_epoch=1.ckpt \
  --output-dir build/examples/mnist \
  --device cpu
```

## Validate ONNX

```bash
python -m examples.mnist.test_onnx \
  --onnx build/examples/mnist/exports/mnist_qat.onnx \
  --dsroot data/mnist \
  --samples-limit 1000 \
  --min-accuracy 0.95
```

All ONNX Runtime sessions explicitly use `CPUExecutionProvider`. Accuracy is a
fraction in `[0, 1]`; `--min-accuracy` makes the command exit nonzero when the
model misses the required threshold. `--samples-limit` must be positive. Add
`--download` only when validation is allowed to populate a missing cache.
Throughput is reported as `samples/s` without implying a precision mode.

## CLI defaults

| Command | Important defaults |
|---|---|
| `train.py` | `--epochs 10`, `--batch 16`, `--workers 1`, QAT enabled, CPU, repository `data/mnist` and `build/examples/mnist` |
| `export_onnx.py` | newest checkpoint under the output directory, CPU |
| `test_onnx.py` | newest file under the default export directory, full test split, no minimum-accuracy gate |

Use `python -m examples.mnist.<script> --help` as the authoritative option
reference.
