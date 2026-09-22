# YOLO26n pure-PyTorch QAT

This example fine-tunes the COCO YOLO26n detector with SiMa QAT using a plain
PyTorch model, data loader, loss, optimizer, and checkpoint. Ultralytics is
needed only once to convert the released pickle checkpoint into a portable
state dictionary; training and evaluation do not import it.

The implementation targets the public COCO nano architecture: 80 classes,
`reg_max=1`, strides 8/16/32, and 2,572,280 parameters. Users remain
responsible for the upstream architecture and checkpoint licenses.

Install the example dependencies in the QAT training environment:

```bash
python -m pip install -r examples/yolo26n_pytorch_qat/requirements.txt ultralytics
```

## Convert the checkpoint

```bash
python examples/yolo26n_pytorch_qat/convert_checkpoint.py \
  --source /path/to/yolo26n.pt \
  --output /path/to/yolo26n_pure.pt
```

The converter verifies strict tensor coverage and bit-exact raw predictions.
The resulting checkpoint contains tensors and primitive metadata and is loaded
with `weights_only=True`.

## Train

Run a short smoke test first with a small COCO subset, then use the same command
with the complete training dataset:

```bash
python examples/yolo26n_pytorch_qat/train.py \
  --weights /path/to/yolo26n_pure.pt \
  --images /path/to/coco/train2017 \
  --annotations /path/to/coco/annotations/instances_train2017.json \
  --output runs/yolo26n_qat \
  --device cuda \
  --batch-size 8 \
  --epochs 4 \
  --freeze-epoch 1
```

The example warms observers before `--freeze-epoch`, locks QAT grids at that
epoch, and trains the remaining epochs for recovery. Each epoch writes a
resumable checkpoint under `OUTPUT/checkpoints`. Use `--resume CHECKPOINT` to
continue a run and `--no-export` for training-only experiments.

After training, the script writes `yolo26n_qat_training_outputs.onnx`, a
complete QDQ graph containing both model heads.

## Evaluate

Compare the floating-point model with a calibrated, frozen pre-training QAT
baseline:

```bash
python examples/yolo26n_pytorch_qat/evaluate.py \
  --weights /path/to/yolo26n_pure.pt \
  --mode both \
  --val-images /path/to/coco/val2017 \
  --val-annotations /path/to/coco/annotations/instances_val2017.json \
  --calibration-images /path/to/coco/train2017 \
  --calibration-annotations /path/to/coco/annotations/instances_train2017.json \
  --output runs/yolo26n_accuracy \
  --device cuda
```

Evaluate a saved QAT checkpoint with the same native YOLO26 NMS-free
postprocessing:

```bash
python examples/yolo26n_pytorch_qat/evaluate.py \
  --weights /path/to/yolo26n_pure.pt \
  --qat-checkpoint runs/yolo26n_qat/checkpoints/epoch_003.pt \
  --mode qat-trained \
  --val-images /path/to/coco/val2017 \
  --val-annotations /path/to/coco/annotations/instances_val2017.json \
  --output runs/yolo26n_qat/eval \
  --device cuda
```

Evaluation writes predictions, per-variant COCO metrics, and `comparison.json`
to the selected output directory.
