# ImageNet QAT example

This example fine-tunes a torchvision ImageNet classifier with PyTorch
Lightning, applies the SiMa QAT lifecycle, exports ONNX, and validates the
result with CPU ONNX Runtime. It also supports a float training/export
baseline.

Run every command below from the repository root. Dataset input is explicit;
generated output defaults to repository-root `build/examples/imagenet` and
does not depend on the working directory.

## Files

| File | Purpose |
|---|---|
| `imagenet_lit.py` | Lightning module, weight policy, and QAT/float export hooks |
| `imagenet_dataset.py` | ImageNet/Imagenette label mapping and balanced sampling |
| `train.py` | Train, checkpoint, fully resume, and optionally export |
| `export_onnx.py` | Verify and export a selected or latest model checkpoint |
| `test_onnx.py` | Check ONNX and measure top-1 accuracy on an ImageFolder split |

## Dataset layout

Pass an ImageFolder-style root containing both splits:

```text
<data-root>/
  train/
    <class-name>/
      image.JPEG
  val/
    <class-name>/
      image.JPEG
```

For a small smoke run, download Imagenette into the reusable repository data
root:

```bash
mkdir -p data
curl -L \
  -o data/imagenette2-160.tgz \
  https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz
tar -xzf data/imagenette2-160.tgz -C data
```

Imagenette WNID folders are remapped to their ImageNet-1K target IDs during
training and evaluation.

## Path contract

| Path | Contents |
|---|---|
| `data/imagenette2-160/` | Optional reusable smoke dataset |
| `data/torch/` | Optional torchvision pretrained-weight cache |
| `build/examples/imagenet/checkpoints/` | Lightning checkpoints |
| `build/examples/imagenet/lightning/` | Lightning run state and logs |
| `build/examples/imagenet/graphs/` | FX graph dumps |
| `build/examples/imagenet/exports/` | Generated ONNX models |

Everything under `build/` is disposable and ignored by Git.

## Fresh training and weight policy

A short CUDA QAT run on Imagenette:

```bash
TORCH_HOME=data/torch \
python -m examples.imagenet.train \
  --model resnet18 \
  --weights DEFAULT \
  --data data/imagenette2-160 \
  --output-dir build/examples/imagenet \
  --batch 64 \
  --device cuda \
  --workers 4 \
  --samples-limit 64 \
  --epochs 1 \
  --export-on-end
```

Fresh training defaults to `--weights none`, so it is offline and starts from
random weights. Request `--weights DEFAULT` explicitly to download the model's
default torchvision weights into `TORCH_HOME`, or pass a model-specific enum
member such as `IMAGENET1K_V1`. Checkpoint resume and checkpoint export always instantiate
with `weights=None`, because the checkpoint already contains every tensor;
they never trigger a pretrained-weight download.

A QAT run writes
`build/examples/imagenet/exports/resnet18_qat.onnx`. Add `--disable-qat` for a
float baseline, which writes `resnet18_float.onnx`.

`--epochs`, `--batch`, and `--samples-limit` must be positive;
`--workers` must be nonnegative. Requesting CUDA fails before model
construction, dataset loading, or a possible pretrained-weight download when
CUDA is unavailable.

## Resume full training state

```bash
python -m examples.imagenet.train \
  --model resnet18 \
  --data data/imagenette2-160 \
  --output-dir build/examples/imagenet \
  --epochs 10 \
  --device cuda \
  --resume
```

The latest checkpoint is selected only from filenames with the exact
`imagenet_resnet18_qat_classifier_` prefix (or the matching `float` prefix
when `--disable-qat` is used), and its saved model metadata is verified. `Trainer.fit(ckpt_path=...)` restores model tensors,
optimizer/scheduler state, epoch, and global step. `--epochs` is the total
target epoch count. QAT and float runs under the same output directory cannot cross-resume; add
`--disable-qat` to select the float namespace. `--weights` is ignored on resume.

## Export a checkpoint

PyTorch Lightning checkpoints use Python pickle. Load only checkpoints from a
trusted source; checkpoint deserialization is not a safe interchange format.

Export the latest verified `resnet18` checkpoint:

```bash
python -m examples.imagenet.export_onnx \
  --model resnet18 \
  --output-dir build/examples/imagenet \
  --device cpu
```

The checkpoint is mapped to CPU and the model is constructed without
pretrained weights, so a checkpoint saved on CUDA can be exported on a
CPU-only host without a download. The exporter verifies model metadata before
loading. A QAT checkpoint creates
`build/examples/imagenet/exports/resnet18_checkpoint_qat.onnx`; a float
checkpoint creates `resnet18_checkpoint_float.onnx`.

Select a checkpoint explicitly for reproducible automation:

```bash
python -m examples.imagenet.export_onnx \
  --model resnet18 \
  --ckpt build/examples/imagenet/checkpoints/imagenet_resnet18_qat_classifier_epoch=0.ckpt \
  --output-dir build/examples/imagenet \
  --device cpu
```

## Validate ONNX

```bash
python -m examples.imagenet.test_onnx \
  --onnx build/examples/imagenet/exports/resnet18_qat.onnx \
  --dsroot data/imagenette2-160 \
  --split val \
  --samples-limit 50 \
  --min-accuracy 0.50
```

All ONNX Runtime sessions explicitly use `CPUExecutionProvider`. Accuracy is a
fraction in `[0, 1]`; `--min-accuracy` exits nonzero when the model misses the
required threshold. `--samples-limit` must be positive. Throughput is reported
as `samples/s` without implying a precision mode.

## CLI defaults

| Command | Important defaults |
|---|---|
| `train.py` | `resnet18`, `--weights none`, QAT enabled, CPU, repository `build/examples/imagenet`; dataset required |
| `export_onnx.py` | model required, newest exact model-scoped checkpoint, CPU |
| `test_onnx.py` | ONNX path and dataset required, `val` split, full split, no accuracy gate |

Use `python -m examples.imagenet.<script> --help` as the authoritative option
reference.

## Backend boundary

The backend uses symbolic FX tracing and does not fall back to Dynamo. Custom
Python control flow must be symbolically traceable. Conv-BN and Conv-BN-ReLU
regions are folded at finalization. Qualify every additional torchvision or
customer model explicitly; choosing a model name does not by itself establish
SiMa compiler support.
