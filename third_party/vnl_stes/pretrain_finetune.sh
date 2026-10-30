#!/usr/bin/env bash

# If invoked as `sh pretrain_finetune.sh`, re-exec with bash.
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

# If your PyTorch build errors with:
#   undefined symbol: iJIT_NotifyEvent
# preload the shared ITT runtime in the active conda env.
if [[ -n "${CONDA_PREFIX:-}" && -f "${CONDA_PREFIX}/lib/libittnotify.so" ]]; then
  export LD_PRELOAD="${CONDA_PREFIX}/lib/libittnotify.so:${LD_PRELOAD:-}"
fi

export OMP_NUM_THREADS=1
export MKL_THREADING_LAYER=GNU
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Shared hyperparameters
MODEL_ARCH="rny008_gsm"
TEMP_ARCH="gru"
CLIP_LEN=64
BATCH_SIZE=8
NUM_WORKERS=4

# Use a single save dir so stage-2 can resume directly from stage-1 checkpoints.
SAVE_DIR="exp/e2espatial_pretrain_finetune"

echo "=== Stage 1: Pretrain on vnl_1.5 for 100 epochs ==="
python train_e2e_spatial.py vnl_1.5 data/vnl_1.5/frames_224p \
  -m "${MODEL_ARCH}" -t "${TEMP_ARCH}" \
  --clip_len "${CLIP_LEN}" --batch_size "${BATCH_SIZE}" \
  --num_epochs 100 \
  -s "${SAVE_DIR}" \
  --predict_location --num_workers "${NUM_WORKERS}" \
  --wandb_project e2espatial_pretrain

echo "=== Stage 2: Finetune on vnl_2.0 for +100 epochs (resume to epoch 200) ==="
python train_e2e_spatial.py vnl_2.0 data/vnl_2.0/frames_224p \
  -m "${MODEL_ARCH}" -t "${TEMP_ARCH}" \
  --clip_len "${CLIP_LEN}" --batch_size "${BATCH_SIZE}" \
  --num_epochs 200 \
  -s "${SAVE_DIR}" \
  --resume \
  --predict_location --num_workers "${NUM_WORKERS}" \
  --wandb_project e2espatial_finetune

echo "Done: pretrain + finetune complete."
