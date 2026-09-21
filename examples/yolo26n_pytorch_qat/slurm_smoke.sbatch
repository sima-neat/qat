#!/usr/bin/env bash
#SBATCH --job-name=yolo26n-qat-smoke
#SBATCH --partition=heavy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtx4000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

EXAMPLE_DIR="${SLURM_SUBMIT_DIR:?Submit this job from the example directory}"
WORKDIR="${QAT_REPO_ROOT:-$(cd "$EXAMPLE_DIR/../.." && pwd)}"
PYTHON_ENV="${QAT_PYTHON_ENV:-$WORKDIR/.venv}"
PYTHON_BIN="${QAT_PYTHON_BIN:-$PYTHON_ENV/bin/python}"
SLURM_HOME="${SLURM_HOME:-/project/sima/tools/opensource/slurm/24.11.3g}"
RUN_DIR="${QAT_SMOKE_RUN_DIR:-$WORKDIR/runs/slurm_gpu_smoke}"
WEIGHTS="$RUN_DIR/yolo26n_pure.pt"

cd "$WORKDIR"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$WORKDIR:$PYTHON_ENV/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"

hostname
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"$PYTHON_BIN" - <<'PY'
import onnx
import PIL
import inspect
import torch
import sima_qat

assert torch.cuda.is_available(), "CUDA is unavailable inside the Slurm allocation"
assert "dynamic_batch" in inspect.signature(
    sima_qat.sima_prepare_qat_model
).parameters, "The checkout's QAT API is not first on PYTHONPATH"
print("python_torch", torch.__version__)
print("cuda_runtime", torch.version.cuda)
print("cuda_device", torch.cuda.get_device_name(0))
print("onnx", onnx.__version__)
print("pillow", PIL.__version__)
print("sima_qat", sima_qat.__file__)
PY

"$SLURM_HOME/bin/srun" "$PYTHON_BIN" examples/yolo26n_pytorch_qat/train.py \
  --weights "$WEIGHTS" \
  --images "${COCO128_IMAGES:-/project/ml_datasets/public_datasets/coco/coco128/images/train2017}" \
  --annotations "${COCO128_ANNOTATIONS:-/project/ml_datasets/public_datasets/coco/coco128/annotations/instances_train2017.json}" \
  --output "$RUN_DIR/output" \
  --device cuda:0 \
  --image-size 160 \
  --batch-size 2 \
  --epochs 2 \
  --freeze-epoch 1 \
  --workers 2 \
  --limit 4 \
  --log-interval 1

test -s "$RUN_DIR/output/yolo26n_qat_training_outputs.onnx"
echo "GPU_SMOKE_OK $RUN_DIR/output/yolo26n_qat_training_outputs.onnx"
