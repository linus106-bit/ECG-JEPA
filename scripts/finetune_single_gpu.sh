#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# ECG-JEPA Finetune — Single GPU
# =============================================================================

# 어디서든 실행 가능하도록 프로젝트 루트로 이동
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

# -----------------------------------------------------------------------------
# [1] CONFIG 선택
#     사용 가능한 eval config (configs/eval/ 참고):
#       linear                → linear probing (인코더 고정)
#       finetune              → end-to-end fine-tuning
#       finetune_after_linear → linear 학습 후 fine-tuning
# -----------------------------------------------------------------------------

CONFIG="finetune"

# -----------------------------------------------------------------------------
# [2] 데이터 설정
# -----------------------------------------------------------------------------

# PTB-XL 데이터 디렉토리 (필수)
DATA_DIR="/path/to/ptb-xl"

# .npy dump 파일 경로 (비워두면 <DATA_DIR>.npy 로 자동 탐색)
DUMP=""

# -----------------------------------------------------------------------------
# [3] 인코더 설정
# -----------------------------------------------------------------------------

# pretrain 체크포인트 경로 또는 pretrain config yaml 경로 (필수)
ENCODER="pretrain/ViTS_ptbxl/checkpoint.pt"

# -----------------------------------------------------------------------------
# [4] 학습 설정
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# [5] 실행
# -----------------------------------------------------------------------------

mkdir -p "${OUT_DIR}"

CMD=(
  python finetune.py
  --data-dir "${DATA_DIR}"
  --encoder  "${ENCODER}"
  --config   "${CONFIG}"
  --out      "${OUT_DIR}"
  --task     "${TASK}"
  --val-fold  "${VAL_FOLD}"
  --test-fold "${TEST_FOLD}"
  --amp      "${AMP}"
)

[[ -n "${DUMP}" ]] && CMD+=(--dump "${DUMP}")

echo "Command: ${CMD[*]}"
"${CMD[@]}"
