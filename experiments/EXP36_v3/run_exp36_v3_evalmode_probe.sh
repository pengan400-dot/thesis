#!/usr/bin/env bash
set -euo pipefail
cd /root/shared-storage/TRAIL_RUL_MODEL_EXPERIMENT_20260808
source .venv/bin/activate

python experiments/exp_36_v3_evalmode_repro_probe.py
