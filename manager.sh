#!/usr/bin/env bash
set -Eeuo pipefail

MODE="${1:-help}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_ROOT="${MSTT_WORK_ROOT:-/root/workspace}"
SHARED_ROOT="${MSTT_SHARED_ROOT:-/root/shared-storage}"
RUN_ROOT="$WORK_ROOT/mstt_b456_confirmatory_runs"
DEV_ZIP="$SHARED_ROOT/xjtu_q2_prepared_csv.zip"
DEV_RAW="$RUN_ROOT/development_csv_raw"
DEV_PREPARED="$RUN_ROOT/development_prepared"
CONFIG="$ROOT/configs/confirmatory_protocol.json"
ARCHIVE_EXPECTED="$ROOT/manifests/library_archive_inventory_20260725.csv"
FREEZE_ZIP="$SHARED_ROOT/MSTT_RUL_Q2_Batch456_v0.1.0-freeze.zip"
RESULT_ZIP="$SHARED_ROOT/MSTT_RUL_Q2_Batch456_confirmatory_results.zip"
TRAIN_LOG="$WORK_ROOT/mstt_b456_freeze_train.log"
CALIBRATION_LOG="$WORK_ROOT/mstt_b456_calibration.log"
EVALUATION_LOG="$WORK_ROOT/mstt_b456_external_evaluation.log"

PYTHON_BIN="$ROOT/.venv/bin/python"

activate_env() {
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "[ERROR] Python environment not found. Run: bash manager.sh setup"
    exit 2
  fi
  export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
  export MPLCONFIGDIR="$RUN_ROOT/environment/matplotlib"
  mkdir -p "$MPLCONFIGDIR"
}

setup() {
  mkdir -p "$RUN_ROOT/environment"
  if [[ ! -x "$PYTHON_BIN" ]]; then
    python3 -m venv --system-site-packages "$ROOT/.venv"
    PYTHON_BIN="$ROOT/.venv/bin/python"
  fi
  "$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
  "$PYTHON_BIN" -m pip install -r "$ROOT/requirements.txt"
  "$PYTHON_BIN" - <<'PY'
import torch, numpy, pandas, scipy, sklearn, joblib, py7zr
print("[TORCH]", torch.__version__)
print("[CUDA]", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[GPU]", torch.cuda.get_device_name(0))
print("[PASS] imports")
PY
  "$PYTHON_BIN" -m pip freeze > "$RUN_ROOT/environment/pip_freeze.txt"
  "$PYTHON_BIN" - <<'PY' > "$RUN_ROOT/environment/runtime.json"
import json, platform, sys
try:
    import torch
    torch_info = {
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
except Exception as exc:
    torch_info = {"error": str(exc)}
print(json.dumps({
    "python": sys.version,
    "platform": platform.platform(),
    "torch": torch_info,
}, indent=2))
PY
  echo "[PASS] setup"
}

verify() {
  activate_env
  "$PYTHON_BIN" -m compileall -q "$ROOT/src" "$ROOT/tests"
  "$PYTHON_BIN" -m unittest discover -s "$ROOT/tests" -v
}

inventory() {
  activate_env
  "$PYTHON_BIN" "$ROOT/src/archive_gate.py" \
    --shared-root "$SHARED_ROOT" \
    --expected "$ARCHIVE_EXPECTED" \
    --output-dir "$RUN_ROOT/inventory"
}

prepare_dev() {
  activate_env
  test -f "$DEV_ZIP" || {
    echo "[ERROR] Missing $DEV_ZIP"
    echo "[NEXT] Upload the prior 31-cell CSV ZIP, or run:"
    echo "       bash manager.sh make_dev_zip /exact/path/to/the/31/source-CSVs"
    exit 2
  }
  rm -rf "$DEV_RAW" "$DEV_PREPARED"
  mkdir -p "$DEV_RAW"
  "$PYTHON_BIN" - "$DEV_ZIP" "$DEV_RAW" <<'PYEXTRACT'
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath

src = Path(sys.argv[1])
dst = Path(sys.argv[2])

with zipfile.ZipFile(src) as archive:
    bad = archive.testzip()
    if bad is not None:
        raise SystemExit(f"[ERROR] ZIP integrity failure: {bad}")

    for info in archive.infolist():
        normalized = info.filename.replace("\\\\", "/")
        parts = [
            part
            for part in PurePosixPath(normalized).parts
            if part not in ("", ".", "..")
        ]
        if not parts:
            continue

        target = dst.joinpath(*parts)

        if info.is_dir() or normalized.endswith("/"):
            target.mkdir(parents=True, exist_ok=True)
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info) as source, target.open("wb") as output:
            shutil.copyfileobj(source, output)

csv_count = len(list(dst.rglob("*.csv")))
print(f"[PASS] development ZIP extracted; CSV files={csv_count}")

if csv_count < 31:
    raise SystemExit(
        f"[ERROR] Expected at least 31 CSV files, got {csv_count}"
    )
PYEXTRACT
  "$PYTHON_BIN" "$ROOT/src/prepare_development.py" \
    --input-root "$DEV_RAW" \
    --output-dir "$DEV_PREPARED" \
    --smoothing-window 7
  echo "[PASS] development curves prepared"
}

make_dev_zip() {
  activate_env
  local source_dir="${2:-}"
  if [[ -z "$source_dir" || ! -d "$source_dir" ]]; then
    echo "Usage: bash manager.sh make_dev_zip /exact/path/to/the/31/source-CSVs"
    exit 2
  fi
  if [[ -e "$DEV_ZIP" ]]; then
    echo "[ERROR] Refusing to overwrite existing $DEV_ZIP"
    exit 2
  fi
  local build_root
  build_root="$(mktemp -d "$RUN_ROOT/development_zip_build.XXXXXX")"
  "$PYTHON_BIN" "$ROOT/src/prepare_development.py" \
    --input-root "$source_dir" \
    --output-dir "$build_root/prepared" \
    --smoothing-window 7
  "$PYTHON_BIN" "$ROOT/src/build_development_zip.py" \
    --prepared-dir "$build_root/prepared" \
    --output-zip "$DEV_ZIP"
}

smoke() {
  activate_env
  test -f "$RUN_ROOT/inventory/archive_gate.json" || {
    echo "[ERROR] Run inventory first"
    exit 2
  }
  test -f "$DEV_PREPARED/preflight_gate.json" || {
    echo "[ERROR] Run prepare_dev first"
    exit 2
  }
  rm -rf "$RUN_ROOT/smoke"
  "$PYTHON_BIN" "$ROOT/src/train_freeze_models.py" \
    --config "$CONFIG" \
    --prepared-dir "$DEV_PREPARED" \
    --output-dir "$RUN_ROOT/smoke" \
    --device auto \
    --quick-test
  echo "[PASS] smoke"
}

train_freeze() {
  activate_env
  test -f "$RUN_ROOT/inventory/archive_gate.json" || {
    echo "[ERROR] Run inventory first"
    exit 2
  }
  test -f "$DEV_PREPARED/preflight_gate.json" || {
    echo "[ERROR] Run prepare_dev first"
    exit 2
  }
  if pgrep -af '[t]rain_freeze_models.py' >/dev/null 2>&1; then
    echo "[ERROR] Frozen-model training is already running"
    pgrep -af '[t]rain_freeze_models.py'
    exit 2
  fi
  nohup "$PYTHON_BIN" "$ROOT/src/train_freeze_models.py" \
    --config "$CONFIG" \
    --prepared-dir "$DEV_PREPARED" \
    --output-dir "$RUN_ROOT/frozen_models" \
    --device auto \
    > "$TRAIN_LOG" 2>&1 < /dev/null &
  echo $! > "$RUN_ROOT/train_freeze.pid"
  echo "[STARTED] frozen-model training PID=$!"
  echo "[LOG] $TRAIN_LOG"
}

calibrate() {
  activate_env
  test -f "$RUN_ROOT/frozen_models/frozen_model_manifest.json" || {
    echo "[ERROR] Frozen-model training has not finished"
    exit 2
  }
  if pgrep -af '[c]rossfit_calibration.py' >/dev/null 2>&1; then
    echo "[ERROR] Calibration is already running"
    pgrep -af '[c]rossfit_calibration.py'
    exit 2
  fi
  nohup "$PYTHON_BIN" "$ROOT/src/crossfit_calibration.py" \
    --config "$CONFIG" \
    --prepared-dir "$DEV_PREPARED" \
    --frozen-model-root "$RUN_ROOT/frozen_models" \
    --output-dir "$RUN_ROOT/calibration" \
    --device auto \
    > "$CALIBRATION_LOG" 2>&1 < /dev/null &
  echo $! > "$RUN_ROOT/calibration.pid"
  echo "[STARTED] cross-fit calibration PID=$!"
  echo "[LOG] $CALIBRATION_LOG"
}

freeze_pack() {
  activate_env
  "$PYTHON_BIN" "$ROOT/src/freeze_pack.py" \
    --source-root "$ROOT" \
    --run-root "$RUN_ROOT" \
    --output-zip "$FREEZE_ZIP"
}

register_receipt() {
  activate_env
  COMMIT="${2:-}"
  DOI="${3:-}"
  if [[ -z "$COMMIT" || -z "$DOI" ]]; then
    echo "Usage: bash manager.sh register_receipt <40-char-commit> <Zenodo-version-DOI>"
    exit 2
  fi
  "$PYTHON_BIN" "$ROOT/src/register_freeze_receipt.py" \
    --commit "$COMMIT" \
    --doi "$DOI" \
    --freeze-zip "$FREEZE_ZIP" \
    --ready-json "$ROOT/freeze_artifacts/FREEZE_READY.json" \
    --output "$RUN_ROOT/freeze_receipt.json"
}

open_external() {
  activate_env
  test -f "$RUN_ROOT/freeze_receipt.json" || {
    echo "[ERROR] No freeze receipt. Publish v0.1.0-freeze and register its DOI first."
    exit 2
  }
  "$PYTHON_BIN" "$ROOT/src/prepare_external.py" \
    --config "$CONFIG" \
    --shared-root "$SHARED_ROOT" \
    --archive-inventory "$RUN_ROOT/inventory/runtime_archive_inventory.csv" \
    --receipt "$RUN_ROOT/freeze_receipt.json" \
    --freeze-zip "$FREEZE_ZIP" \
    --raw-dir "$RUN_ROOT/external_raw" \
    --output-dir "$RUN_ROOT/external_prepared"
}

evaluate() {
  activate_env
  test -f "$RUN_ROOT/external_prepared/external_preflight.json" || {
    echo "[ERROR] Run open_external first"
    exit 2
  }
  if pgrep -af '[e]valuate_external.py' >/dev/null 2>&1; then
    echo "[ERROR] External evaluation is already running"
    pgrep -af '[e]valuate_external.py'
    exit 2
  fi
  nohup "$PYTHON_BIN" "$ROOT/src/evaluate_external.py" \
    --config "$CONFIG" \
    --external-dir "$RUN_ROOT/external_prepared" \
    --frozen-model-root "$RUN_ROOT/frozen_models" \
    --calibration-dir "$RUN_ROOT/calibration" \
    --output-dir "$RUN_ROOT/external_evaluation" \
    --device auto \
    > "$EVALUATION_LOG" 2>&1 < /dev/null &
  echo $! > "$RUN_ROOT/evaluation.pid"
  echo "[STARTED] external evaluation PID=$!"
  echo "[LOG] $EVALUATION_LOG"
}

aggregate() {
  activate_env
  test -f "$RUN_ROOT/external_evaluation/evaluation_audit.json" || {
    echo "[ERROR] External evaluation has not finished"
    exit 2
  }
  "$PYTHON_BIN" "$ROOT/src/aggregate_results.py" \
    --config "$CONFIG" \
    --evaluation-dir "$RUN_ROOT/external_evaluation" \
    --output-dir "$RUN_ROOT/aggregate"
}

pack_results() {
  activate_env
  "$PYTHON_BIN" "$ROOT/src/pack_results.py" \
    --run-root "$RUN_ROOT" \
    --source-root "$ROOT" \
    --output-zip "$RESULT_ZIP"
}

status() {
  local status_python="$PYTHON_BIN"
  if [[ ! -x "$status_python" ]]; then
    status_python="$(command -v python3 || true)"
  fi
  echo "===== TIME ====="
  date
  echo "===== PROCESSES ====="
  pgrep -af '[t]rain_freeze_models.py|[c]rossfit_calibration.py|[e]valuate_external.py' \
    || echo "No active training/calibration/evaluation process"
  echo "===== GPU ====="
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader 2>/dev/null || true
  echo "===== GATES ====="
  for path in \
    "$RUN_ROOT/inventory/archive_gate.json" \
    "$RUN_ROOT/frozen_models/frozen_model_manifest.json" \
    "$RUN_ROOT/calibration/calibration_quantiles.json" \
    "$ROOT/freeze_artifacts/FREEZE_READY.json" \
    "$RUN_ROOT/freeze_receipt.json" \
    "$RUN_ROOT/external_prepared/external_preflight.json" \
    "$RUN_ROOT/external_evaluation/evaluation_audit.json" \
    "$RUN_ROOT/aggregate/aggregation_audit.json"; do
    if [[ -f "$path" ]]; then
      if [[ -z "$status_python" ]]; then
        echo "$(basename "$path"): present (Python unavailable for JSON status)"
        continue
      fi
      "$status_python" - "$path" <<'PY'
import json, pathlib, sys
p=pathlib.Path(sys.argv[1])
x=json.loads(p.read_text())
print(f"{p.name}: {x.get('status', 'UNKNOWN')}")
PY
    else
      echo "$(basename "$path"): pending"
    fi
  done
  echo "===== LATEST LOGS ====="
  for log in "$TRAIN_LOG" "$CALIBRATION_LOG" "$EVALUATION_LOG"; do
    echo "--- $log"
    tail -n 8 "$log" 2>/dev/null || true
  done
}

case "$MODE" in
  setup) setup ;;
  verify) verify ;;
  inventory) inventory ;;
  prepare_dev) prepare_dev ;;
  make_dev_zip) make_dev_zip "$@" ;;
  smoke) smoke ;;
  train_freeze) train_freeze ;;
  calibrate) calibrate ;;
  freeze_pack) freeze_pack ;;
  register_receipt) register_receipt "$@" ;;
  open_external) open_external ;;
  evaluate) evaluate ;;
  aggregate) aggregate ;;
  pack_results) pack_results ;;
  status) status ;;
  *)
    cat <<EOF
Usage:
  bash manager.sh setup
  bash manager.sh verify
  bash manager.sh inventory
  bash manager.sh make_dev_zip /exact/path/to/the/31/source-CSVs
  bash manager.sh prepare_dev
  bash manager.sh smoke
  bash manager.sh train_freeze
  bash manager.sh status
  bash manager.sh calibrate
  bash manager.sh freeze_pack
  bash manager.sh register_receipt <40-char-commit> <Zenodo-version-DOI>
  bash manager.sh open_external
  bash manager.sh evaluate
  bash manager.sh aggregate
  bash manager.sh pack_results
EOF
  ;;
esac
