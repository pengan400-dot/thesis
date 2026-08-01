#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_ROOT="${HUST_RUN_ROOT:-$PROJECT_ROOT/run_v030}"
HUST_ARCHIVE="${HUST_ARCHIVE:-/root/shared-storage/hust_data.zip}"
DEVICE="${HUST_DEVICE:-auto}"

CONFIG="$PROJECT_ROOT/configs/hust_confirmatory_protocol_v0.3.0.json"
DEV_CONFIG="$PROJECT_ROOT/configs/soh_development_protocol_v0.2.0.json"
XJTU_ZIP="$PROJECT_ROOT/inputs/xjtu_q2_prepared_csv.zip"

INVENTORY="$RUN_ROOT/00_hust_structure_inventory"
XJTU_SOURCE="$RUN_ROOT/01_xjtu_source"
DEVELOPMENT="$RUN_ROOT/02_xjtu_soh_development"
MODELS="$RUN_ROOT/03_xjtu_soh_frozen_models"
CALIBRATION="$RUN_ROOT/04_hust_calibration"
CONFIRMATION="$RUN_ROOT/confirmation"
AGGREGATE="$RUN_ROOT/aggregate"
FREEZES="$RUN_ROOT/freezes"
LOGS="$RUN_ROOT/logs"

EVALUATOR_ZIP="$FREEZES/MSTT_RUL_v0.3.0_HUST_evaluator_pre_calibration_freeze.zip"
EVALUATOR_MANIFEST="$EVALUATOR_ZIP.manifest.json"
EVALUATOR_RECEIPT="$FREEZES/evaluator_freeze_receipt.json"
CALIBRATION_ZIP="$FREEZES/MSTT_RUL_v0.3.0_HUST_calibration_pre_confirmation_freeze.zip"
CALIBRATION_MANIFEST="$CALIBRATION_ZIP.manifest.json"
CALIBRATION_RECEIPT="$FREEZES/calibration_freeze_receipt.json"
RESULT_ZIP="$RUN_ROOT/MSTT_RUL_v0.3.0_HUST_confirmatory_one_shot_results.zip"

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$RUN_ROOT/matplotlib-cache}"

mkdir -p "$RUN_ROOT" "$FREEZES" "$LOGS" "$MPLCONFIGDIR"

fail() {
  echo "[FAIL] $*" >&2
  exit 1
}

need_file() {
  [ -f "$1" ] || fail "missing file: $1"
}

python_cli() {
  "$PYTHON_BIN" -m hust_v030.cli "$@"
}

check_hust_archive() {
  need_file "$HUST_ARCHIVE"
  [ "$(basename -- "$HUST_ARCHIVE")" = "hust_data.zip" ] || \
    fail "registered archive name must be hust_data.zip: $HUST_ARCHIVE"
}

show_help() {
  cat <<'EOF'
MSTT_RUL v0.3.0 HUST one-shot manager

Required environment (only if defaults do not fit):
  HUST_ARCHIVE=/absolute/path/hust_data.zip
  HUST_RUN_ROOT=/absolute/path/new_run_directory
  HUST_DEVICE=auto|cpu|cuda
  PYTHON_BIN=python3

Commands, in frozen order:
  install                         Install Python dependencies.
  check                           Check imports and print fixed paths.
  test                            Run synthetic structure/parser/statistics tests.
  inventory                       Structure-only inventory; no pickle unpickling.
  prepare_development             Prepare bundled XJTU B1+B3 SOH development data.
  smoke                           Two-epoch one-model smoke test; never formal weights.
  train                           Train all 2 models x 3 seeds in foreground.
  train_bg                        Train formally in background; safe exact-job resume.
  status                          Show formal training manifest/log state.
  freeze_evaluator                Freeze code, protocol, receipts, and formal weights.
  register_evaluator COMMIT URL   Bind evaluator freeze to 40-char commit and immutable URL/DOI.
  open_calibration                Unpickle only the frozen 20-cell calibration arm.
  calibrate                       Calibrate 80%/90% simultaneous bands; no model update.
  freeze_calibration              Freeze calibration before confirmation access.
  register_calibration COMMIT URL Bind calibration freeze to commit and immutable URL/DOI.
  confirm                         One-shot 57-cell confirmation in foreground.
  confirm_bg                      One-shot confirmation in background.
  confirm_resume                  Resume only an interrupted exact frozen confirmation.
  aggregate                       Cell-first inference, tables, and plots.
  pack                            Build the final result ZIP and SHA-256 sidecar.
  paths                           Print all important paths.

Never run open_calibration before the evaluator receipt exists. Never run confirm
before the calibration receipt exists. Do not delete one_shot_state.json after a
failed confirmation; use confirm_resume with the exact same files.
EOF
}

show_paths() {
  printf '%s\n' \
    "PROJECT_ROOT=$PROJECT_ROOT" \
    "HUST_ARCHIVE=$HUST_ARCHIVE" \
    "RUN_ROOT=$RUN_ROOT" \
    "MODELS=$MODELS" \
    "EVALUATOR_ZIP=$EVALUATOR_ZIP" \
    "EVALUATOR_RECEIPT=$EVALUATOR_RECEIPT" \
    "CALIBRATION_ZIP=$CALIBRATION_ZIP" \
    "CALIBRATION_RECEIPT=$CALIBRATION_RECEIPT" \
    "RESULT_ZIP=$RESULT_ZIP"
}

confirm_arguments() {
  printf '%s\0' \
    --config "$CONFIG" \
    --dev-config "$DEV_CONFIG" \
    --archive "$HUST_ARCHIVE" \
    --inventory "$INVENTORY" \
    --evaluator-zip "$EVALUATOR_ZIP" \
    --evaluator-manifest "$EVALUATOR_MANIFEST" \
    --evaluator-receipt "$EVALUATOR_RECEIPT" \
    --calibration-zip "$CALIBRATION_ZIP" \
    --calibration-manifest "$CALIBRATION_MANIFEST" \
    --calibration-receipt "$CALIBRATION_RECEIPT" \
    --calibration "$CALIBRATION" \
    --models "$MODELS" \
    --output "$CONFIRMATION" \
    --device "$DEVICE" \
    --project-root "$PROJECT_ROOT" \
    --trust-official-pickle
}

ACTION="${1:-help}"
shift || true

case "$ACTION" in
  help|-h|--help)
    show_help
    ;;
  paths)
    show_paths
    ;;
  install)
    "$PYTHON_BIN" -m pip install -r "$PROJECT_ROOT/requirements.txt"
    ;;
  check)
    "$PYTHON_BIN" - <<'PY'
import joblib, matplotlib, numpy, pandas, scipy, sklearn, torch
print("[PASS] imports")
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("numpy", numpy.__version__, "pandas", pandas.__version__)
print("scipy", scipy.__version__, "sklearn", sklearn.__version__)
PY
    show_paths
    ;;
  test)
    "$PYTHON_BIN" -m unittest discover -s "$PROJECT_ROOT/tests" -v
    ;;
  inventory)
    check_hust_archive
    python_cli inventory \
      --config "$CONFIG" \
      --archive "$HUST_ARCHIVE" \
      --output "$INVENTORY"
    ;;
  prepare_development)
    [ ! -e "$EVALUATOR_ZIP" ] || fail "evaluator is already frozen; development files are locked"
    python_cli extract-xjtu --source-zip "$XJTU_ZIP" --output "$XJTU_SOURCE"
    "$PYTHON_BIN" -m mstt_soh.pipeline prepare-development \
      --config "$DEV_CONFIG" \
      --source "$XJTU_SOURCE" \
      --output "$DEVELOPMENT"
    cp "$XJTU_SOURCE/xjtu_input_receipt.json" "$DEVELOPMENT/xjtu_input_receipt.json"
    ;;
  smoke)
    "$PYTHON_BIN" -m mstt_soh.pipeline train \
      --config "$DEV_CONFIG" \
      --development "$DEVELOPMENT" \
      --output "$RUN_ROOT/smoke_only_not_formal" \
      --device "$DEVICE" \
      --quick-test
    ;;
  train)
    [ ! -e "$EVALUATOR_ZIP" ] || fail "evaluator is already frozen; model files are locked"
    "$PYTHON_BIN" -m mstt_soh.pipeline train \
      --config "$DEV_CONFIG" \
      --development "$DEVELOPMENT" \
      --output "$MODELS" \
      --device "$DEVICE"
    ;;
  train_bg)
    [ ! -e "$EVALUATOR_ZIP" ] || fail "evaluator is already frozen; model files are locked"
    nohup "$PYTHON_BIN" -m mstt_soh.pipeline train \
      --config "$DEV_CONFIG" \
      --development "$DEVELOPMENT" \
      --output "$MODELS" \
      --device "$DEVICE" \
      >"$LOGS/formal_training.log" 2>&1 &
    printf '%s\n' "$!" >"$LOGS/formal_training.pid"
    echo "[STARTED] pid=$! log=$LOGS/formal_training.log"
    ;;
  status)
    if [ -f "$MODELS/frozen_model_manifest.json" ]; then
      "$PYTHON_BIN" - "$MODELS/frozen_model_manifest.json" <<'PY'
import json, sys
p=json.load(open(sys.argv[1], encoding="utf-8"))
print("manifest_status=", p.get("status"))
print("quick_test=", p.get("quick_test"))
print("formal_jobs=", len(p.get("jobs", [])))
PY
    else
      echo "manifest_status=NOT_COMPLETE"
    fi
    if [ -f "$LOGS/formal_training.log" ]; then
      tail -n 30 "$LOGS/formal_training.log"
    fi
    ;;
  freeze_evaluator)
    check_hust_archive
    [ ! -e "$CONFIRMATION/one_shot_state.json" ] || \
      fail "confirmation was already opened; evaluator freeze is too late"
    python_cli freeze-evaluator \
      --config "$CONFIG" \
      --dev-config "$DEV_CONFIG" \
      --archive "$HUST_ARCHIVE" \
      --inventory "$INVENTORY" \
      --development "$DEVELOPMENT" \
      --models "$MODELS" \
      --project-root "$PROJECT_ROOT" \
      --output-zip "$EVALUATOR_ZIP" \
      --calibration-preflight "$CALIBRATION/calibration_preflight.json"
    ;;
  register_evaluator)
    [ "$#" -eq 2 ] || fail "usage: ./manager.sh register_evaluator 40_CHAR_COMMIT IMMUTABLE_URL_OR_DOI"
    python_cli register-freeze \
      --kind evaluator \
      --archive "$EVALUATOR_ZIP" \
      --manifest "$EVALUATOR_MANIFEST" \
      --commit "$1" \
      --locator "$2" \
      --output "$EVALUATOR_RECEIPT" \
      --calibration-preflight "$CALIBRATION/calibration_preflight.json" \
      --confirmation-state "$CONFIRMATION/one_shot_state.json"
    ;;
  open_calibration)
    check_hust_archive
    need_file "$EVALUATOR_RECEIPT"
    [ ! -e "$CONFIRMATION/one_shot_state.json" ] || \
      fail "confirmation was already opened"
    python_cli open-calibration \
      --config "$CONFIG" \
      --archive "$HUST_ARCHIVE" \
      --inventory "$INVENTORY" \
      --output "$CALIBRATION" \
      --evaluator-zip "$EVALUATOR_ZIP" \
      --evaluator-manifest "$EVALUATOR_MANIFEST" \
      --evaluator-receipt "$EVALUATOR_RECEIPT" \
      --models "$MODELS" \
      --project-root "$PROJECT_ROOT" \
      --trust-official-pickle
    ;;
  calibrate)
    python_cli calibrate \
      --config "$CONFIG" \
      --dev-config "$DEV_CONFIG" \
      --archive "$HUST_ARCHIVE" \
      --inventory "$INVENTORY" \
      --evaluator-zip "$EVALUATOR_ZIP" \
      --evaluator-manifest "$EVALUATOR_MANIFEST" \
      --evaluator-receipt "$EVALUATOR_RECEIPT" \
      --models "$MODELS" \
      --calibration "$CALIBRATION" \
      --device "$DEVICE" \
      --project-root "$PROJECT_ROOT"
    ;;
  freeze_calibration)
    [ ! -e "$CONFIRMATION/one_shot_state.json" ] || \
      fail "confirmation was already opened; calibration freeze is too late"
    python_cli freeze-calibration \
      --config "$CONFIG" \
      --archive "$HUST_ARCHIVE" \
      --inventory "$INVENTORY" \
      --evaluator-zip "$EVALUATOR_ZIP" \
      --evaluator-manifest "$EVALUATOR_MANIFEST" \
      --evaluator-receipt "$EVALUATOR_RECEIPT" \
      --models "$MODELS" \
      --calibration "$CALIBRATION" \
      --project-root "$PROJECT_ROOT" \
      --output-zip "$CALIBRATION_ZIP"
    ;;
  register_calibration)
    [ "$#" -eq 2 ] || fail "usage: ./manager.sh register_calibration 40_CHAR_COMMIT IMMUTABLE_URL_OR_DOI"
    python_cli register-freeze \
      --kind calibration \
      --archive "$CALIBRATION_ZIP" \
      --manifest "$CALIBRATION_MANIFEST" \
      --commit "$1" \
      --locator "$2" \
      --output "$CALIBRATION_RECEIPT" \
      --calibration-preflight "$CALIBRATION/calibration_preflight.json" \
      --confirmation-state "$CONFIRMATION/one_shot_state.json"
    ;;
  confirm|confirm_bg|confirm_resume)
    check_hust_archive
    need_file "$EVALUATOR_RECEIPT"
    need_file "$CALIBRATION_RECEIPT"
    mapfile -d '' CONFIRM_ARGS < <(confirm_arguments)
    if [ "$ACTION" = "confirm_resume" ]; then
      python_cli confirm "${CONFIRM_ARGS[@]}" --resume
    elif [ "$ACTION" = "confirm_bg" ]; then
      nohup "$PYTHON_BIN" -m hust_v030.cli confirm "${CONFIRM_ARGS[@]}" \
        >"$LOGS/one_shot_confirmation.log" 2>&1 &
      printf '%s\n' "$!" >"$LOGS/one_shot_confirmation.pid"
      echo "[STARTED-ONE-SHOT] pid=$! log=$LOGS/one_shot_confirmation.log"
    else
      python_cli confirm "${CONFIRM_ARGS[@]}"
    fi
    ;;
  aggregate)
    python_cli aggregate \
      --config "$CONFIG" \
      --confirmation "$CONFIRMATION" \
      --output "$AGGREGATE"
    ;;
  pack)
    python_cli pack \
      --config "$CONFIG" \
      --inventory "$INVENTORY" \
      --evaluator-manifest "$EVALUATOR_MANIFEST" \
      --evaluator-receipt "$EVALUATOR_RECEIPT" \
      --calibration-manifest "$CALIBRATION_MANIFEST" \
      --calibration-receipt "$CALIBRATION_RECEIPT" \
      --calibration "$CALIBRATION" \
      --confirmation "$CONFIRMATION" \
      --aggregate "$AGGREGATE" \
      --project-root "$PROJECT_ROOT" \
      --output-zip "$RESULT_ZIP"
    ;;
  *)
    fail "unknown command: $ACTION (run ./manager.sh help)"
    ;;
esac
