#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source .venv/bin/activate

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONDONTWRITEBYTECODE=1

LOGDIR="runs/dreamerv3_env_diy$(date +%Y%m%d_%H%M%S)"

python world_model/dreamerv3/dreamerv3/main.py \
  --configs env_diy \
  --task env_diy_default \
  --logdir "$LOGDIR" \
  --run.steps 1000000 \
  --run.envs 4 \
  --batch_size 16 \
  --batch_length 64 \
  --report_length 32 \
  --run.train_ratio 32 \
  --jax.platform cuda \
  --jax.prealloc False \
  --jax.compute_dtype float32
