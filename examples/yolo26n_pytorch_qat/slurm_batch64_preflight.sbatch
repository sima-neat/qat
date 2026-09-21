#!/usr/bin/env bash
#SBATCH --job-name=yolo26n-b64-test
#SBATCH --partition=heavy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

EXAMPLE_DIR="${SLURM_SUBMIT_DIR:?Submit this job from the example directory}"
WORKDIR="${QAT_REPO_ROOT:-$(cd "$EXAMPLE_DIR/../.." && pwd)}"
PYTHON_ENV="${QAT_PYTHON_ENV:-$WORKDIR/../afe_env}"
PYTHON_BIN="${QAT_PYTHON_BIN:-$WORKDIR/../../../python312/bin/python3.12}"
SLURM_HOME="${SLURM_HOME:-/project/sima/tools/opensource/slurm/24.11.3g}"
WEIGHTS="${QAT_YOLO26N_WEIGHTS:-$WORKDIR/runs/slurm_gpu_smoke/yolo26n_pure.pt}"
RUN_DIR="$WORKDIR/runs/yolo26n_batch64_preflight"

cd "$WORKDIR"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$WORKDIR:$PYTHON_ENV/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"

nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
"$SLURM_HOME/bin/srun" "$PYTHON_BIN" examples/yolo26n_pytorch_qat/train.py \
  --weights "$WEIGHTS" \
  --output "$RUN_DIR" \
  --device cuda:0 \
  --image-size 640 \
  --batch-size 64 \
  --epochs 1 \
  --freeze-epoch -1 \
  --learning-rate 8e-5 \
  --workers 16 \
  --limit 128 \
  --log-interval 1 \
  --no-export
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
test -s "$RUN_DIR/checkpoints/epoch_000.pt"
echo "BATCH64_PREFLIGHT_OK"
