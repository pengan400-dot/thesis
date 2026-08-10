#!/usr/bin/env bash
set -euo pipefail
cd /root/shared-storage/TRAIL_RUL_MODEL_EXPERIMENT_20260808
source .venv/bin/activate

python experiments/exp_36_trail_dgr_prefix_replay_v3.py \
  --seeds 42 2024 3407 \
  --preflight-only \
  --output-dir results/trail_dgr_prefix_replay_v3
