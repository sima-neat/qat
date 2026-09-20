# YOLO26n pure-PyTorch QAT

This directory fine-tunes the COCO YOLO26n detector with SiMa QAT without the
Ultralytics trainer, dataset loader, model implementation, or loss at training
time. It reads COCO JSON directly and uses an ordinary PyTorch `DataLoader`,
optimizer, backward pass, and checkpoint.

The released `yolo26n.pt` file is a Python pickle containing Ultralytics class
references. `convert_checkpoint.py` therefore uses Ultralytics once to copy its
708 tensors into a safe, portable state-dict checkpoint. The converter verifies
strict key coverage and bit-exact raw predictions. `train.py` and every file it
imports contain no Ultralytics dependency.

The standalone implementation is fixed to the public COCO nano architecture:
80 classes, `reg_max=1`, strides 8/16/32, and 2,572,280 parameters. Architecture
or checkpoint licensing remains the user's responsibility; removing a runtime
dependency does not change the upstream model's license.

## 1. Convert the released checkpoint once

From the QAT repository root:

```bash
python examples/yolo26n_pytorch_qat/convert_checkpoint.py \
  --source /path/to/yolo26n.pt \
  --output /path/to/yolo26n_pure.pt \
  --ultralytics-source ../.model-audit-sources/ultralytics
```

An installed `ultralytics` package can be used instead by omitting
`--ultralytics-source`. The generated checkpoint contains only tensors and
primitive metadata and is loaded with `weights_only=True`.

## 2. Run a technical COCO128 smoke test

```bash
python examples/yolo26n_pytorch_qat/train.py \
  --weights /path/to/yolo26n_pure.pt \
  --images /project/ml_datasets/public_datasets/coco/coco128/images/train2017 \
  --annotations /project/ml_datasets/public_datasets/coco/coco128/annotations/instances_train2017.json \
  --output runs/yolo26n_coco128_qat \
  --device cuda \
  --batch-size 8 \
  --epochs 2 \
  --freeze-epoch 1
```

COCO128 only validates capture, gradients, optimizer updates, scale locking,
checkpointing, finalization, and export. It cannot establish detector accuracy.

## 3. Fine-tune on full COCO 2017

The full shared dataset is already the default. No YOLO `.txt` conversion is
needed:

```bash
python examples/yolo26n_pytorch_qat/train.py \
  --weights /path/to/yolo26n_pure.pt \
  --output runs/yolo26n_coco_qat \
  --device cuda \
  --batch-size 8 \
  --epochs 10 \
  --freeze-epoch 8 \
  --learning-rate 1e-5
```

The default data paths are:

```text
/project/ml_datasets/public_datasets/coco/coco_2017/train2017
/project/ml_datasets/public_datasets/coco/coco_2017/annotations/instances_train2017.json
```

The last two epochs in the example above train against locked, compiler-valid
quantization grids. `--freeze-epoch -1` defers locking until finalization, which
is useful only for diagnosis. AMP is opt-in with `--amp`; start without it when
validating a new GPU environment.

## Outputs

Each epoch writes a resumable QAT state-dict checkpoint under `checkpoints/`.
After the final epoch the trainer writes:

- `yolo26n_qat_training_outputs.onnx`: complete QDQ graph with both training heads.
- `yolo26n_qat_raw_heads.onnx`: six one-to-one QDQ heads in grouped order
  `bbox_0..2, class_logit_0..2`, ready for SiMa YOLO26 BoxDecode.

Use `--resume checkpoints/epoch_NNN.pt` to continue a run and `--no-export` for
training-only experiments.

## Slurm GPU smoke

`slurm_smoke.sbatch` requests one RTX 4000 from the `heavy` partition and runs
four real COCO128 samples through two epochs. The second epoch locks QAT grids;
the job then finalizes and verifies the raw-head ONNX artifact. Its checkpoint
input is expected at `runs/slurm_gpu_smoke/yolo26n_pure.pt`. Submit it only after
creating that portable checkpoint:

```bash
cd examples/yolo26n_pytorch_qat
mkdir -p logs
sbatch slurm_smoke.sbatch
```

Submit from this directory so Slurm can resolve the relative log and repository
paths. `QAT_REPO_ROOT`, `QAT_PYTHON_ENV`, `QAT_PYTHON_BIN`,
`QAT_SMOKE_RUN_DIR`, `COCO128_IMAGES`, and `COCO128_ANNOTATIONS` can override
the cluster defaults.

## Current production gates

The folder is a runnable single-GPU QAT path, not an accuracy sign-off. Before
shipping a checkpoint, add or run:

- FP32 versus finalized-QAT COCO `val2017` mAP comparison.
- A longer augmentation study; this initial fine-tuning loader intentionally
  implements only aspect-preserving letterbox resize and horizontal flip.
- Multi-GPU DDP if one-GPU throughput is insufficient.
- BoxDecode execution comparison against the PyTorch raw-head decoder.
