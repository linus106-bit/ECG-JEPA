#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# ECG-JEPA Finetune — Single GPU
# =============================================================================

# -----------------------------------------------------------------------------
# 설정 (자주 수정하는 항목)
# -----------------------------------------------------------------------------

# eval config 이름 (configs/eval/ 참고)
#   linear                → linear probing (인코더 고정)
#   finetune              → end-to-end fine-tuning
#   finetune_after_linear → linear 학습 후 fine-tuning
CONFIG="finetune"

# pretrain 체크포인트 경로 또는 pretrain config yaml 경로 (필수)
ENCODER="pretrain/ViTS_ptbxl/checkpoint.pt"

# 출력 디렉토리 (체크포인트 및 예측 결과 저장 위치)
OUT_DIR="finetune/${CONFIG}"

# 평가 task: all | diagnostic | subdiagnostic | superdiagnostic | form | rhythm | ST-MEM
TASK="all"

# validation fold (1-10)
VAL_FOLD=9

# test fold (1-10)
TEST_FOLD=10

# 정밀도: float32 | bfloat16
AMP="bfloat16"

# dataset type: ecg | har
DATASET_TYPE="ecg"

# 데이터 경로 (HF dataset 디렉토리 경로)
#   방법 A (권장): configs/eval/<CONFIG>.yaml 의 dataset.data_dir 에서 지정
#   방법 B: 아래 변수를 직접 채우면 yaml 설정을 덮어씀
DATA_DIR=""

# =============================================================================
# 이 아래는 수정하지 않아도 됩니다
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

mkdir -p "${OUT_DIR}"

CMD=(
  python finetune.py
  --encoder      "${ENCODER}"
  --config       "${CONFIG}"
  --out          "${OUT_DIR}"
  --dataset-type "${DATASET_TYPE}"
  --task         "${TASK}"
  --val-fold     "${VAL_FOLD}"
  --test-fold    "${TEST_FOLD}"
  --amp          "${AMP}"
)

[[ -n "${DATA_DIR}" ]] && CMD+=(--data-dir "${DATA_DIR}")

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="${OUT_DIR}/train_${TIMESTAMP}.log"
echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Command: ${CMD[*]}"
echo "Logging to: ${LOG_FILE}"
{ echo "=== Started at: $(date '+%Y-%m-%d %H:%M:%S') ==="; "${CMD[@]}"; } 2>&1 | tee "${LOG_FILE}"
