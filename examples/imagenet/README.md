# ImageNet Classifier Example

This example fine-tunes a pretrained torchvision ImageNet classifier with the SiMa QAT
prepare/finalize/export flow, then validates the exported ONNX model with ONNXRuntime.

Files:

```text
imagenet/
  README.md
  imagenet_lit.py      # Lightning module with QAT hooks
  train.py             # train / checkpoint / optional export
  export_onnx.py       # export the latest or selected checkpoint
  test_onnx.py         # validate an ONNX model on an ImageFolder split
```

## Dataset Layout

The scripts expect an ImageFolder-style dataset root:

```text
<data-root>/
  train/
    <class_name>/
      image.JPEG
  val/
    <class_name>/
      image.JPEG
```

For full ImageNet-2012, pass the ImageNet root with `-d /data/imagenet` or
`--dsroot /data/imagenet`. The training script reads `<data-root>/train`, and
`test_onnx.py --split val` reads `<data-root>/val`.

For a small smoke test, Imagenette already uses this layout and can be downloaded locally:

```bash
cd examples/imagenet
mkdir -p data
curl -L -o data/imagenette2-160.tgz https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz
tar -xzf data/imagenette2-160.tgz -C data
```

## Quick CUDA Smoke Test

```bash
cd examples/imagenet

python train.py --model resnet18 -d data/imagenette2-160 -b 64 --device cuda \
  --samples-limit 64 --epochs 1 --workers 4 --export-on-end
python export_onnx.py --model resnet18 --device cuda
python test_onnx.py --onnx exported_ckpt_resnet18_onnx_model.onnx \
  --dsroot data/imagenette2-160 --split val --samples-limit 50
```

For a longer smoke run, raise `--samples-limit` to `500`. The Imagenette WNID
folders are remapped to ImageNet-1K target IDs before training and evaluation.
For the full ImageNet dataset, use the same commands with `-d /data/imagenet`
and `--dsroot /data/imagenet`.

## QAT Flow

The Lightning module wires QAT into standard training hooks:

- `on_train_start()` calls `sima_prepare_qat_model()` to capture the torchvision model and insert QAT scaffolding.
- `on_train_end()` calls `sima_finalize_qat_model()` to convert the trained graph to inference-only fake-quant form.
- `on_fit_end()` calls `sima_export_onnx()` when `--export-on-end` is set.

`export_onnx.py` can also re-export the latest checkpoint after training.

## Training Arguments

Run `python train.py --help` for the current CLI.

| Argument | Default | Description |
|---|---:|---|
| `-e, --epochs` | `10` | Training epochs. |
| `-b, --batch` | `1` | Batch size. Use a larger value on CUDA for faster smoke runs. |
| `-d, --data` | `.` | Dataset root containing `train/` and `val/`. |
| `--device` | `cpu` | Lightning accelerator/device selector such as `cpu` or `cuda`. |
| `--model` | `resnet18` | Torchvision ImageNet model name. |
| `--samples-limit` | `1281167` | Limit training samples with class-balanced selection; useful for smoke tests. |
| `--workers` | `4` | DataLoader worker processes for train and validation. |
| `--export-on-end` | `False` | Export ONNX at the end of `trainer.fit()`. |
| `--disable-qat` | `False` | Train a float baseline instead of QAT. |
| `--resume` | `False` | Resume from the latest checkpoint under `checkpoints/`. |

## ONNX Validation Arguments

Run `python test_onnx.py --help` for the current CLI.

| Argument | Default | Description |
|---|---:|---|
| `--onnx` | latest ONNX | ONNX model to validate. |
| `--dsroot` | `.` | Dataset root containing split directories. |
| `--split` | `val` | Dataset split folder to evaluate. |
| `--samples-limit` | unset | Limit evaluation samples with class-balanced selection; useful for quick smoke tests. |
| `-v, --verbosity` | `INFO` | Logging level. |

Example:

```bash
python test_onnx.py --onnx exported_ckpt_resnet18_onnx_model.onnx --dsroot data/imagenette2-160 --split val --samples-limit 50
```

## Checkpoint Export Arguments

Run `python export_onnx.py --help` for the current CLI.

| Argument | Default | Description |
|---|---:|---|
| `--model` | required | Model name used to find the latest matching checkpoint. |
| `--device` | `cpu` | Device to restore the model to after CPU ONNX export. |
| `-c, --ckpt` | latest matching checkpoint | Explicit checkpoint path. |

Example:

```bash
python export_onnx.py --model resnet18 --device cuda
```

## Backend Notes

- QAT uses symbolic FX tracing and does not fall back to Dynamo. Custom Python
  control flow must be symbolically traceable.
- Conv-BN and Conv-BN-ReLU modules are folded during finalization.
- PyTorch 2.8 may emit FX quantization and legacy ONNX exporter deprecation
  warnings. The Model Compiler target remains PyTorch 2.3.1.
- Torchvision pretrained-constructor warnings do not affect QAT export.

## Debugging Tips

- Use `--samples-limit` on `train.py` and `test_onnx.py` for fast class-balanced smoke tests.
- Use `--workers 4` or lower if the dataloader is noisy or the machine has few CPU cores.
- For an FX graph object, `{fx_graph_model}.graph.print_tabular()` prints node-level details.
