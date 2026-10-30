#!/usr/bin/env bash

# If invoked as `sh run.sh`, re-exec with bash.
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

# If your PyTorch build errors with:
#   undefined symbol: iJIT_NotifyEvent
# preload the shared ITT runtime we created in the conda env.
if [[ -n "${CONDA_PREFIX:-}" && -f "${CONDA_PREFIX}/lib/libittnotify.so" ]]; then
  export LD_PRELOAD="${CONDA_PREFIX}/lib/libittnotify.so:${LD_PRELOAD:-}"
fi

export OMP_NUM_THREADS=1
export MKL_THREADING_LAYER=GNU
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Full training.
python train_e2e_spatial.py vnl_2.0 data/vnl_2.0/frames_224p \
  -m rny008_gsm -t gru --clip_len 64 --batch_size 8 --num_epochs 150 \
  -s exp/e2espatial_vnl2 --predict_location --num_workers 4 \
  --wandb_project e2espatial_vnl2
