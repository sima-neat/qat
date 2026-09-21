#!/usr/bin/env bash
#SBATCH --job-name=yolo26n-qat-full
#SBATCH --partition=heavy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

EXAMPLE_DIR="${SLURM_SUBMIT_DIR:?Submit this job from the example directory}"
WORKDIR="${QAT_REPO_ROOT:-$(cd "$EXAMPLE_DIR/../.." && pwd)}"
PYTHON_ENV="${QAT_PYTHON_ENV:-$WORKDIR/.venv}"
PYTHON_BIN="${QAT_PYTHON_BIN:-$PYTHON_ENV/bin/python}"
EVAL_DEPS="${QAT_EVAL_DEPS:-$WORKDIR/runs/coco_eval_deps}"
SLURM_HOME="${SLURM_HOME:-/project/sima/tools/opensource/slurm/24.11.3g}"
RUN_DIR="${QAT_FULL_RUN_DIR:-$WORKDIR/runs/yolo26n_coco_qat_4e_recovery}"
WEIGHTS="${QAT_YOLO26N_WEIGHTS:-$WORKDIR/runs/slurm_gpu_smoke/yolo26n_pure.pt}"
LEARNING_RATE="${QAT_LEARNING_RATE:-1e-5}"

cd "$WORKDIR"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$WORKDIR:$PYTHON_ENV/lib/python3.12/site-packages:$EVAL_DEPS${PYTHONPATH:+:$PYTHONPATH}"

hostname
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"$PYTHON_BIN" - <<'PY'
import pycocotools
import torch
import sima_qat

assert torch.cuda.is_available(), "CUDA is unavailable inside the Slurm allocation"
print("torch", torch.__version__)
print("cuda_runtime", torch.version.cuda)
print("cuda_device", torch.cuda.get_device_name(0))
print("pycocotools", pycocotools.__file__)
print("sima_qat", sima_qat.__file__)
PY

"$SLURM_HOME/bin/srun" "$PYTHON_BIN" examples/yolo26n_pytorch_qat/train.py \
  --weights "$WEIGHTS" \
  --output "$RUN_DIR/train" \
  --device cuda:0 \
  --image-size 640 \
  --batch-size 64 \
  --epochs 4 \
  --freeze-epoch 1 \
  --learning-rate "$LEARNING_RATE" \
  --workers 16 \
  --log-interval 200

for epoch in 000 001 002 003; do
  "$SLURM_HOME/bin/srun" "$PYTHON_BIN" examples/yolo26n_pytorch_qat/evaluate.py \
    --weights "$WEIGHTS" \
    --qat-checkpoint "$RUN_DIR/train/checkpoints/epoch_${epoch}.pt" \
    --output "$RUN_DIR/eval_epoch_${epoch}" \
    --mode qat-trained \
    --device cuda:0 \
    --image-size 640 \
    --batch-size 64 \
    --workers 16 \
    --log-interval 50
done

test -s "$RUN_DIR/train/yolo26n_qat_training_outputs.onnx"
test -s "$RUN_DIR/eval_epoch_003/comparison.json"
echo "FULL_QAT_OK $RUN_DIR"
