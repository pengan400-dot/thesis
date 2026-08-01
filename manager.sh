#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHARED_ROOT="${MSTT_SHARED_ROOT:-/root/shared-storage}"
WORK_ROOT="${MSTT_WORK_ROOT:-/root/workspace}"
RUN_ROOT="${MSTT_RUN_ROOT:-${WORK_ROOT}/MSTT_RUL_v020_SOH_BIT_run}"
XJTU_PREPARED_DIR="${XJTU_PREPARED_DIR:-${WORK_ROOT}/MSTT_RUL_Q2_Batch456_confirmatory_20260725/development_prepared}"
HNEI_ARCHIVE="${HNEI_ARCHIVE:-${SHARED_ROOT}/BatteryLife.zip}"
BIT_SOURCE="${BIT_SOURCE:-${SHARED_ROOT}/kw34hhw7xg-3.zip}"
BIT_SCHEMA_MAPPING="${BIT_SCHEMA_MAPPING:-${PROJECT_ROOT}/amendments/v0.2.1-BIT-structural-feasibility/configs/bit_schema_mapping_v0.2.1.yaml}"
BIT_MAPPING_RESOLUTION="${PROJECT_ROOT}/amendments/v0.2.1-BIT-structural-feasibility/AUDIT/CYCLE_MAPPING_PASS_OFFSET20_20260801/bit_cycle_mapping_resolution_v0.2.1.json"
BIT_MAPPING_RECEIPT="${SHARED_ROOT}/BIT_v0.2.1_offset20_mapping_commit_receipt.txt"

CONFIG="${PROJECT_ROOT}/configs/soh_development_protocol_v0.2.0.json"
BIT_CONFIG="${PROJECT_ROOT}/configs/soh_development_protocol_v0.2.1_BIT_amendment.json"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
PIPELINE=(env "PYTHONPATH=${PROJECT_ROOT}/src" "${PYTHON}" -m mstt_soh.pipeline)
CONFORMAL=(env "PYTHONPATH=${PROJECT_ROOT}/src" "${PYTHON}" -m mstt_soh.conformal)
DEV_FREEZER=(env "PYTHONPATH=${PROJECT_ROOT}/src" "${PYTHON}" -m mstt_soh.development_freeze)
BIT_EVALUATOR=(env "PYTHONPATH=${PROJECT_ROOT}/src" "${PYTHON}" "${PROJECT_ROOT}/scripts/bit_v021_evaluator.py")

ENVIRONMENT="${RUN_ROOT}/00_environment"
DEVELOPMENT="${RUN_ROOT}/01_xjtu_soh_development"
SMOKE_MODELS="${RUN_ROOT}/02_smoke/models"
SMOKE_CONFORMAL="${RUN_ROOT}/02_smoke/conformal"
MODELS="${RUN_ROOT}/03_xjtu_soh_models"
CONFORMAL_DIR="${RUN_ROOT}/04_xjtu_crossfit_conformal"
HNEI_INVENTORY="${RUN_ROOT}/05_hnei_inventory"
DEVELOPMENT_RECEIPT_DIR="${RUN_ROOT}/06_development_freeze_receipt"
DEVELOPMENT_RECEIPT="${DEVELOPMENT_RECEIPT_DIR}/receipt.json"
HNEI_EXTERNAL="${RUN_ROOT}/07_hnei_prepared"
HNEI_EVALUATION="${RUN_ROOT}/08_hnei_evaluation"
HNEI_AGGREGATE="${RUN_ROOT}/09_hnei_aggregate"
HNEI_DECISION_DIR="${RUN_ROOT}/10_hnei_decision"
HNEI_DECISION="${HNEI_DECISION_DIR}/hnei_version_decision.json"
BIT_PREFLIGHT="${RUN_ROOT}/11_bit_structure_preflight"
BIT_RECEIPT_DIR="${RUN_ROOT}/12_bit_registration_receipt"
BIT_RECEIPT="${BIT_RECEIPT_DIR}/receipt.json"
BIT_V021_VERIFY_DIR="${RUN_ROOT}/16_bit_v021_pre_freeze_verify"
BIT_V021_VERIFY_RECEIPT="${BIT_V021_VERIFY_DIR}/pre_freeze_verify_receipt.json"
BIT_EVALUATION="${RUN_ROOT}/14_bit_evaluation_LOCKED"
BIT_EVALUATOR_VERIFY_DIR="${RUN_ROOT}/17_bit_v021_evaluator_source_verify"
BIT_EVALUATOR_VERIFY_RECEIPT="${BIT_EVALUATOR_VERIFY_DIR}/evaluator_source_verify_receipt.json"
BIT_EVALUATOR_SOURCE_DIR="${RUN_ROOT}/18_bit_v021_evaluator_source_freeze"
BIT_EVALUATOR_SOURCE_RECEIPT="${BIT_EVALUATOR_SOURCE_DIR}/evaluator_source_freeze_receipt.json"
BIT_RAW_DIR="${BIT_EVALUATION}/00_raw_capacity"
BIT_RAW_CAPACITY="${BIT_RAW_DIR}/bit_raw_capacity_v0.2.1.csv"
BIT_PREPARED="${BIT_EVALUATION}/01_prepared"
BIT_PREDICTIONS="${BIT_EVALUATION}/02_evaluation"
BIT_AGGREGATE="${BIT_EVALUATION}/03_aggregate"
BIT_PACK_STAGING="${RUN_ROOT}/15_bit_result_pack_staging"
BIT_PAIRING_MANIFEST="${PROJECT_ROOT}/amendments/v0.2.1-BIT-structural-feasibility/AUDIT/STRUCTURE_MAPPING_EVIDENCE_20260801/cell_boundary_inventory.csv"
BIT_MAPPING_MANIFEST="${PROJECT_ROOT}/amendments/v0.2.1-BIT-structural-feasibility/AUDIT/CYCLE_MAPPING_PASS_OFFSET20_20260801/bit_cycle_mapping_manifest_v0.2.1.csv"

DEV_FREEZE_ZIP="${SHARED_ROOT}/MSTT_RUL_v0.2.0_SOH_development_freeze.zip"
DEV_FREEZE_MANIFEST="${SHARED_ROOT}/MSTT_RUL_v0.2.0_SOH_development_freeze.manifest.json"
HNEI_RESULT_ZIP="${SHARED_ROOT}/MSTT_RUL_v0.2.0_SOH_HNEI_exploratory_results.zip"
BIT_FREEZE_ZIP="${SHARED_ROOT}/MSTT_RUL_v0.2.1_BIT_structural_feasibility_pre_model_evaluation_freeze.zip"
BIT_FREEZE_MANIFEST="${SHARED_ROOT}/MSTT_RUL_v0.2.1_BIT_structural_feasibility_pre_model_evaluation_freeze.manifest.json"
BIT_EVALUATOR_SOURCE_ZIP="${SHARED_ROOT}/MSTT_RUL_v0.2.1_BIT_evaluator_source_pre_capacity_freeze.zip"
BIT_EVALUATOR_SOURCE_MANIFEST="${SHARED_ROOT}/MSTT_RUL_v0.2.1_BIT_evaluator_source_pre_capacity_freeze.manifest.json"
BIT_RESULT_ZIP="${SHARED_ROOT}/MSTT_RUL_v0.2.1_BIT_confirmatory_results.zip"

export MPLCONFIGDIR="${RUN_ROOT}/matplotlib_cache"
mkdir -p "${MPLCONFIGDIR}" "${SHARED_ROOT}"

usage() {
  sed -n '1,280p' "${PROJECT_ROOT}/README_CN.md"
}

require_python() {
  if [[ ! -x "${PYTHON}" ]]; then
    echo "[ERROR] 先运行: bash manager.sh setup" >&2
    exit 2
  fi
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "[ERROR] 缺少文件: $1" >&2
    exit 2
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "[ERROR] 缺少目录: $1" >&2
    exit 2
  fi
}

require_hnei_unchanged() {
  require_file "${HNEI_DECISION}"
  "${PYTHON}" - "${HNEI_DECISION}" <<'PY'
import json
import pathlib
import sys

receipt = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if receipt.get("status") != "PASS_V0_2_0_UNCHANGED":
    raise SystemExit(
        "[ERROR] HNEI 已触发模型修改；当前 v0.2.0 不得接触 BIT。"
    )
PY
}

command="${1:-help}"
case "${command}" in
  setup)
    python3 -m venv "${PROJECT_ROOT}/.venv"
    "${PROJECT_ROOT}/.venv/bin/pip" install --upgrade pip
    "${PROJECT_ROOT}/.venv/bin/pip" install \
      -r "${PROJECT_ROOT}/environment/requirements.txt"
    echo "[PASS] Python 环境完成"
    ;;

  verify)
    require_python
    "${PYTHON}" "${PROJECT_ROOT}/scripts/verify_contract_no_torch.py" \
      --config "${CONFIG}"
    "${PYTHON}" -m py_compile \
      "${PROJECT_ROOT}"/src/mstt_soh/*.py \
      "${PROJECT_ROOT}"/scripts/*.py
    PYTHONPATH="${PROJECT_ROOT}/src" \
      "${PYTHON}" -m unittest discover \
      -s "${PROJECT_ROOT}/tests" -v
    "${PYTHON}" "${PROJECT_ROOT}/scripts/capture_environment.py" \
      --output-dir "${ENVIRONMENT}"
    "${PYTHON}" -m unittest discover \
      -s "${PROJECT_ROOT}/tests" \
      -p "test_bit_v021_freeze_contract.py" -v
    mkdir -p "${BIT_V021_VERIFY_DIR}"
    "${PYTHON}" "${PROJECT_ROOT}/scripts/verify_bit_v021_pre_freeze.py" \
      --config "${BIT_CONFIG}" \
      --schema-mapping "${BIT_SCHEMA_MAPPING}" \
      --mapping-resolution "${BIT_MAPPING_RESOLUTION}" \
      --mapping-receipt "${BIT_MAPPING_RECEIPT}" \
      --project-root "${PROJECT_ROOT}" \
      --output "${BIT_V021_VERIFY_RECEIPT}"
    echo "[PASS] 合同、语法、模型身份、单元测试和环境记录通过"
    ;;

  prepare_dev)
    require_python
    require_dir "${XJTU_PREPARED_DIR}"
    "${PIPELINE[@]}" prepare-development \
      --config "${CONFIG}" \
      --source "${XJTU_PREPARED_DIR}" \
      --output "${DEVELOPMENT}"
    ;;

  smoke)
    require_python
    require_file "${DEVELOPMENT}/development_preflight.json"
    "${PIPELINE[@]}" train \
      --config "${CONFIG}" \
      --development "${DEVELOPMENT}" \
      --output "${SMOKE_MODELS}" \
      --device auto \
      --quick-test
    "${CONFORMAL[@]}" \
      --config "${CONFIG}" \
      --development "${DEVELOPMENT}" \
      --output "${SMOKE_CONFORMAL}" \
      --device auto \
      --quick-test
    echo "[PASS] 训练和 cross-fit conformal 冒烟链通过；不用于论文"
    ;;

  train_soh)
    require_python
    require_file "${DEVELOPMENT}/development_preflight.json"
    "${PIPELINE[@]}" train \
      --config "${CONFIG}" \
      --development "${DEVELOPMENT}" \
      --output "${MODELS}" \
      --device auto
    ;;

  calibrate_soh)
    require_python
    require_file "${DEVELOPMENT}/development_preflight.json"
    "${CONFORMAL[@]}" \
      --config "${CONFIG}" \
      --development "${DEVELOPMENT}" \
      --output "${CONFORMAL_DIR}" \
      --device auto
    ;;

  freeze_development)
    require_python
    require_file "${ENVIRONMENT}/runtime_environment.json"
    require_file "${DEVELOPMENT}/development_preflight.json"
    require_file "${MODELS}/frozen_model_manifest.json"
    require_file "${CONFORMAL_DIR}/calibration_manifest.json"
    "${DEV_FREEZER[@]}" \
      --config "${CONFIG}" \
      --project-root "${PROJECT_ROOT}" \
      --development "${DEVELOPMENT}" \
      --models "${MODELS}" \
      --conformal "${CONFORMAL_DIR}" \
      --runtime-environment "${ENVIRONMENT}" \
      --output-zip "${DEV_FREEZE_ZIP}"
    ;;

  hnei_inventory)
    require_python
    require_file "${DEV_FREEZE_ZIP}"
    require_file "${DEVELOPMENT_RECEIPT}"
    require_file "${HNEI_ARCHIVE}"
    "${PIPELINE[@]}" inventory \
      --config "${CONFIG}" \
      --archive "${HNEI_ARCHIVE}" \
      --output "${HNEI_INVENTORY}"
    ;;

  register_development)
    require_python
    if [[ "$#" -ne 4 ]]; then
      echo "用法: bash manager.sh register_development <40位commit> <GitHub release URL> <Zenodo版本DOI>" >&2
      exit 2
    fi
    require_file "${DEV_FREEZE_ZIP}"
    require_file "${DEV_FREEZE_MANIFEST}"
    mkdir -p "${DEVELOPMENT_RECEIPT_DIR}"
    "${PIPELINE[@]}" register \
      --config "${CONFIG}" \
      --freeze-zip "${DEV_FREEZE_ZIP}" \
      --freeze-manifest "${DEV_FREEZE_MANIFEST}" \
      --evaluation "${HNEI_EVALUATION}" \
      --commit "$2" \
      --github-release "$3" \
      --doi "$4" \
      --output "${DEVELOPMENT_RECEIPT}"
    ;;

  hnei_open)
    require_python
    require_file "${HNEI_ARCHIVE}"
    require_file "${HNEI_INVENTORY}/inventory.json"
    require_file "${DEV_FREEZE_ZIP}"
    require_file "${DEVELOPMENT_RECEIPT}"
    "${PIPELINE[@]}" prepare-hnei \
      --config "${CONFIG}" \
      --archive "${HNEI_ARCHIVE}" \
      --inventory "${HNEI_INVENTORY}" \
      --freeze-zip "${DEV_FREEZE_ZIP}" \
      --receipt "${DEVELOPMENT_RECEIPT}" \
      --output "${HNEI_EXTERNAL}"
    ;;

  hnei_evaluate)
    require_python
    require_file "${HNEI_EXTERNAL}/external_preflight.json"
    require_file "${MODELS}/frozen_model_manifest.json"
    require_file "${DEV_FREEZE_ZIP}"
    require_file "${DEVELOPMENT_RECEIPT}"
    "${PIPELINE[@]}" evaluate \
      --config "${CONFIG}" \
      --external "${HNEI_EXTERNAL}" \
      --models "${MODELS}" \
      --freeze-zip "${DEV_FREEZE_ZIP}" \
      --receipt "${DEVELOPMENT_RECEIPT}" \
      --output "${HNEI_EVALUATION}" \
      --device auto
    ;;

  hnei_aggregate)
    require_python
    require_file "${HNEI_EVALUATION}/evaluation_audit.json"
    "${PIPELINE[@]}" aggregate \
      --config "${CONFIG}" \
      --evaluation "${HNEI_EVALUATION}" \
      --output "${HNEI_AGGREGATE}"
    ;;

  hnei_pack_results)
    require_python
    require_file "${HNEI_AGGREGATE}/aggregate_audit.json"
    "${PIPELINE[@]}" pack \
      --inventory "${HNEI_INVENTORY}" \
      --development "${DEVELOPMENT}" \
      --models "${MODELS}" \
      --external "${HNEI_EXTERNAL}" \
      --evaluation "${HNEI_EVALUATION}" \
      --aggregate "${HNEI_AGGREGATE}" \
      --receipt "${DEVELOPMENT_RECEIPT}" \
      --output-zip "${HNEI_RESULT_ZIP}"
    ;;

  hnei_decision)
    require_python
    if [[ "$#" -lt 3 ]]; then
      echo "用法: bash manager.sh hnei_decision <unchanged|modified> '<理由>'" >&2
      exit 2
    fi
    require_file "${HNEI_RESULT_ZIP}"
    require_file "${HNEI_AGGREGATE}/aggregate_audit.json"
    require_file "${DEV_FREEZE_ZIP}"
    mkdir -p "${HNEI_DECISION_DIR}"
    "${PYTHON}" "${PROJECT_ROOT}/scripts/record_hnei_decision.py" \
      --decision "$2" \
      --hnei-results "${HNEI_RESULT_ZIP}" \
      --aggregate-audit "${HNEI_AGGREGATE}/aggregate_audit.json" \
      --development-freeze "${DEV_FREEZE_ZIP}" \
      --rationale "$3" \
      --output "${HNEI_DECISION}"
    ;;

  bit_preflight)
    require_python
    require_hnei_unchanged
    require_file "${BIT_SOURCE}"
    "${PYTHON}" "${PROJECT_ROOT}/scripts/bit_structure_preflight.py" \
      --source "${BIT_SOURCE}" \
      --out "${BIT_PREFLIGHT}"
    echo "[STOP] 不要运行任何 BIT 模型。先核对三份结构报告并解析 schema。"
    ;;

  bit_freeze)
    require_python
    require_hnei_unchanged
    require_file "${BIT_CONFIG}"
    require_file "${BIT_SCHEMA_MAPPING}"
    require_file "${BIT_MAPPING_RESOLUTION}"
    require_file "${BIT_MAPPING_RECEIPT}"
    require_file "${BIT_V021_VERIFY_RECEIPT}"
    require_file "${DEV_FREEZE_ZIP}"
    require_file "${DEV_FREEZE_MANIFEST}"
    require_file "${BIT_PREFLIGHT}/bit_structural_preflight.json"
    "${PYTHON}" "${PROJECT_ROOT}/scripts/freeze_bit_protocol.py" \
      --config "${BIT_CONFIG}" \
      --schema-mapping "${BIT_SCHEMA_MAPPING}" \
      --preflight-dir "${BIT_PREFLIGHT}" \
      --development-freeze "${DEV_FREEZE_ZIP}" \
      --development-manifest "${DEV_FREEZE_MANIFEST}" \
      --hnei-decision "${HNEI_DECISION}" \
      --mapping-resolution "${BIT_MAPPING_RESOLUTION}" \
      --mapping-receipt "${BIT_MAPPING_RECEIPT}" \
      --verify-receipt "${BIT_V021_VERIFY_RECEIPT}" \
      --project-root "${PROJECT_ROOT}" \
      --output-zip "${BIT_FREEZE_ZIP}"
    ;;

  bit_register)
    require_python
    if [[ "$#" -ne 6 ]]; then
      echo "用法: bash manager.sh bit_register <40位commit> <GitHub release URL> <OSF registration URL> <Zenodo版本DOI> <保留参数>" >&2
      echo "最后一个参数固定写 CONFIRM，防止误触。" >&2
      exit 2
    fi
    if [[ "$6" != "CONFIRM" ]]; then
      echo "[ERROR] 最后一个参数必须是 CONFIRM" >&2
      exit 2
    fi
    require_file "${BIT_FREEZE_ZIP}"
    require_file "${BIT_FREEZE_MANIFEST}"
    mkdir -p "${BIT_RECEIPT_DIR}"
    "${PYTHON}" "${PROJECT_ROOT}/scripts/register_bit_freeze_receipt.py" \
      --freeze-zip "${BIT_FREEZE_ZIP}" \
      --freeze-manifest "${BIT_FREEZE_MANIFEST}" \
      --commit "$2" \
      --github-release "$3" \
      --osf-registration "$4" \
      --zenodo-doi "$5" \
      --output "${BIT_RECEIPT}" \
      --evaluation-dir "${BIT_EVALUATION}"
    echo "[PASS] BIT 注册收据已生成。本代码包仍不含 BIT 模型评估命令。"
    ;;


  bit_eval_verify)
    require_python
    require_file "${BIT_RECEIPT}"
    require_file "${BIT_FREEZE_MANIFEST}"
    "${PYTHON}" -m py_compile \
      "${PROJECT_ROOT}/scripts/bit_v021_evaluator.py" \
      "${PROJECT_ROOT}/scripts/verify_bit_evaluator_source_v0_2_1.py" \
      "${PROJECT_ROOT}/scripts/freeze_bit_evaluator_source_v0_2_1.py" \
      "${PROJECT_ROOT}/tests/test_bit_v021_evaluator_contract.py"
    PYTHONPATH="${PROJECT_ROOT}/src" \
      "${PYTHON}" -m unittest discover \
      -s "${PROJECT_ROOT}/tests" \
      -p "test_bit_v021_evaluator_contract.py" -v
    mkdir -p "${BIT_EVALUATOR_VERIFY_DIR}"
    "${PYTHON}" "${PROJECT_ROOT}/scripts/verify_bit_evaluator_source_v0_2_1.py" \
      --project-root "${PROJECT_ROOT}" \
      --registration-receipt "${BIT_RECEIPT}" \
      --freeze-manifest "${BIT_FREEZE_MANIFEST}" \
      --evaluation-dir "${BIT_EVALUATION}" \
      --output "${BIT_EVALUATOR_VERIFY_RECEIPT}"
    ;;

  bit_eval_source_freeze)
    require_python
    require_file "${BIT_EVALUATOR_VERIFY_RECEIPT}"
    require_file "${BIT_RECEIPT}"
    require_file "${BIT_FREEZE_MANIFEST}"
    mkdir -p "${BIT_EVALUATOR_SOURCE_DIR}"
    "${PYTHON}" "${PROJECT_ROOT}/scripts/freeze_bit_evaluator_source_v0_2_1.py" \
      --project-root "${PROJECT_ROOT}" \
      --verify-receipt "${BIT_EVALUATOR_VERIFY_RECEIPT}" \
      --registration-receipt "${BIT_RECEIPT}" \
      --freeze-manifest "${BIT_FREEZE_MANIFEST}" \
      --evaluation-dir "${BIT_EVALUATION}" \
      --output-zip "${BIT_EVALUATOR_SOURCE_ZIP}" \
      --output-manifest "${BIT_EVALUATOR_SOURCE_MANIFEST}" \
      --output-receipt "${BIT_EVALUATOR_SOURCE_RECEIPT}"
    ;;

  bit_parse)
    require_python
    require_file "${BIT_EVALUATOR_SOURCE_RECEIPT}"
    require_file "${BIT_SOURCE}"
    require_file "${BIT_PAIRING_MANIFEST}"
    require_file "${BIT_MAPPING_MANIFEST}"
    "${BIT_EVALUATOR[@]}" parse \
      --project-root "${PROJECT_ROOT}" \
      --bit-config "${BIT_CONFIG}" \
      --development-config "${CONFIG}" \
      --registration-receipt "${BIT_RECEIPT}" \
      --source-receipt "${BIT_EVALUATOR_SOURCE_RECEIPT}" \
      --source-freeze-zip "${BIT_EVALUATOR_SOURCE_ZIP}" \
      --freeze-manifest "${BIT_FREEZE_MANIFEST}" \
      --archive "${BIT_SOURCE}" \
      --parser "${PROJECT_ROOT}/scripts/parse_bit_capacity_v0_2_1.py" \
      --pairing-manifest "${BIT_PAIRING_MANIFEST}" \
      --mapping-manifest "${BIT_MAPPING_MANIFEST}" \
      --output "${BIT_RAW_CAPACITY}"
    ;;

  bit_prepare)
    require_python
    require_file "${BIT_RAW_CAPACITY}"
    "${BIT_EVALUATOR[@]}" prepare \
      --project-root "${PROJECT_ROOT}" \
      --bit-config "${BIT_CONFIG}" \
      --development-config "${CONFIG}" \
      --registration-receipt "${BIT_RECEIPT}" \
      --source-receipt "${BIT_EVALUATOR_SOURCE_RECEIPT}" \
      --source-freeze-zip "${BIT_EVALUATOR_SOURCE_ZIP}" \
      --freeze-manifest "${BIT_FREEZE_MANIFEST}" \
      --raw-capacity-csv "${BIT_RAW_CAPACITY}" \
      --output "${BIT_PREPARED}"
    ;;

  bit_evaluate)
    require_python
    require_file "${BIT_PREPARED}/preparation_audit.json"
    require_file "${MODELS}/frozen_model_manifest.json"
    "${BIT_EVALUATOR[@]}" evaluate \
      --project-root "${PROJECT_ROOT}" \
      --bit-config "${BIT_CONFIG}" \
      --development-config "${CONFIG}" \
      --registration-receipt "${BIT_RECEIPT}" \
      --source-receipt "${BIT_EVALUATOR_SOURCE_RECEIPT}" \
      --source-freeze-zip "${BIT_EVALUATOR_SOURCE_ZIP}" \
      --freeze-manifest "${BIT_FREEZE_MANIFEST}" \
      --prepared "${BIT_PREPARED}" \
      --models "${MODELS}" \
      --output "${BIT_PREDICTIONS}" \
      --device auto
    ;;

  bit_aggregate)
    require_python
    require_file "${BIT_PREDICTIONS}/evaluation_audit.json"
    "${BIT_EVALUATOR[@]}" aggregate \
      --project-root "${PROJECT_ROOT}" \
      --bit-config "${BIT_CONFIG}" \
      --development-config "${CONFIG}" \
      --registration-receipt "${BIT_RECEIPT}" \
      --source-receipt "${BIT_EVALUATOR_SOURCE_RECEIPT}" \
      --source-freeze-zip "${BIT_EVALUATOR_SOURCE_ZIP}" \
      --freeze-manifest "${BIT_FREEZE_MANIFEST}" \
      --prepared "${BIT_PREPARED}" \
      --evaluation "${BIT_PREDICTIONS}" \
      --output "${BIT_AGGREGATE}"
    ;;

  bit_pack_results)
    require_python
    require_file "${BIT_AGGREGATE}/aggregate_audit.json"
    "${BIT_EVALUATOR[@]}" pack \
      --project-root "${PROJECT_ROOT}" \
      --bit-config "${BIT_CONFIG}" \
      --development-config "${CONFIG}" \
      --registration-receipt "${BIT_RECEIPT}" \
      --source-receipt "${BIT_EVALUATOR_SOURCE_RECEIPT}" \
      --source-freeze-zip "${BIT_EVALUATOR_SOURCE_ZIP}" \
      --freeze-manifest "${BIT_FREEZE_MANIFEST}" \
      --prepared "${BIT_PREPARED}" \
      --evaluation "${BIT_PREDICTIONS}" \
      --aggregate "${BIT_AGGREGATE}" \
      --models "${MODELS}" \
      --package-root "${BIT_PACK_STAGING}" \
      --output-zip "${BIT_RESULT_ZIP}"
    ;;

  status)
    for item in \
      "${ENVIRONMENT}/runtime_environment.json" \
      "${DEVELOPMENT}/development_preflight.json" \
      "${SMOKE_MODELS}/frozen_model_manifest.json" \
      "${SMOKE_CONFORMAL}/calibration_manifest.json" \
      "${MODELS}/frozen_model_manifest.json" \
      "${CONFORMAL_DIR}/calibration_manifest.json" \
      "${DEV_FREEZE_ZIP}" \
      "${DEVELOPMENT_RECEIPT}" \
      "${HNEI_INVENTORY}/inventory.json" \
      "${HNEI_EXTERNAL}/external_preflight.json" \
      "${HNEI_EVALUATION}/evaluation_audit.json" \
      "${HNEI_AGGREGATE}/aggregate_audit.json" \
      "${HNEI_RESULT_ZIP}" \
      "${HNEI_DECISION}" \
      "${BIT_PREFLIGHT}/bit_structural_preflight.json" \
      "${BIT_FREEZE_ZIP}" \
      "${BIT_RECEIPT}" \
      "${BIT_EVALUATOR_VERIFY_RECEIPT}" \
      "${BIT_EVALUATOR_SOURCE_ZIP}" \
      "${BIT_EVALUATOR_SOURCE_RECEIPT}" \
      "${BIT_PREPARED}/preparation_audit.json" \
      "${BIT_PREDICTIONS}/evaluation_audit.json" \
      "${BIT_AGGREGATE}/aggregate_audit.json" \
      "${BIT_RESULT_ZIP}"
    do
      if [[ -e "${item}" ]]; then
        echo "[YES] ${item}"
      else
        echo "[NO ] ${item}"
      fi
    done
    ;;

  help|-h|--help)
    usage
    ;;

  *)
    echo "[ERROR] 未知命令: ${command}" >&2
    usage >&2
    exit 2
    ;;
esac
