#!/usr/bin/env bash
#SBATCH --job-name=yolo26n-qat-eval
#SBATCH --partition=heavy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx4000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

EXAMPLE_DIR="${SLURM_SUBMIT_DIR:?Submit this job from the example directory}"
WORKDIR="${QAT_REPO_ROOT:-$(cd "$EXAMPLE_DIR/../.." && pwd)}"
PYTHON_ENV="${QAT_PYTHON_ENV:-$WORKDIR/.venv}"
PYTHON_BIN="${QAT_PYTHON_BIN:-$PYTHON_ENV/bin/python}"
EVAL_DEPS="${QAT_EVAL_DEPS:-$WORKDIR/runs/coco_eval_deps}"
SLURM_HOME="${SLURM_HOME:-/project/sima/tools/opensource/slurm/24.11.3g}"
RUN_DIR="${QAT_ACCURACY_RUN_DIR:-$WORKDIR/runs/yolo26n_accuracy}"
WEIGHTS="${QAT_YOLO26N_WEIGHTS:-$WORKDIR/runs/slurm_gpu_smoke/yolo26n_pure.pt}"

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

"$SLURM_HOME/bin/srun" "$PYTHON_BIN" examples/yolo26n_pytorch_qat/evaluate.py \
  --weights "$WEIGHTS" \
  --output "$RUN_DIR" \
  --mode both \
  --device cuda:0 \
  --image-size 640 \
  --batch-size 8 \
  --workers 8 \
  --calibration-limit 1024 \
  --log-interval 50

test -s "$RUN_DIR/comparison.json"
echo "ACCURACY_EVAL_OK $RUN_DIR/comparison.json"
