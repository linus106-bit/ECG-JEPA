#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# ECG-JEPA Finetune — Multi GPU (torchrun, single node)
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
# [2] GPU 설정
# -----------------------------------------------------------------------------

# 사용할 GPU 수 (서버의 실제 GPU 수 이하로 설정)
NUM_GPUS=8

# 사용할 GPU ID 지정 (전체 사용 시 주석 처리)
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# -----------------------------------------------------------------------------
# [3] 데이터 설정
#     데이터 경로는 두 가지 방법 중 하나로 지정합니다.
#
#     방법 A (권장): eval config yaml 파일 안에서 지정
#       configs/eval/<CONFIG>.yaml 에 아래 내용을 추가:
#         dataset:
#           data_dir: /path/to/ptb-xl
#           dump: /path/to/ptb-xl.npy   # 생략 시 <data_dir>.npy 로 자동 탐색
#
#     방법 B: 아래 변수로 직접 지정 (yaml 설정을 덮어씀)
# -----------------------------------------------------------------------------

# PTB-XL 데이터 디렉토리 (비워두면 yaml의 dataset.data_dir 사용)
DATA_DIR=""

# .npy dump 파일 경로 (비워두면 yaml의 dataset.dump 또는 <DATA_DIR>.npy 로 자동 탐색)
DUMP=""

# -----------------------------------------------------------------------------
# [4] 인코더 설정
# -----------------------------------------------------------------------------

# pretrain 체크포인트 경로 또는 pretrain config yaml 경로 (필수)
ENCODER="pretrain/ViTS_ptbxl/checkpoint.pt"

# -----------------------------------------------------------------------------
# [5] 학습 설정
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
# [6] 실행
# -----------------------------------------------------------------------------

mkdir -p "${OUT_DIR}"

CMD=(
  torchrun
    --standalone
    --nproc_per_node="${NUM_GPUS}"
  finetune.py
  --encoder  "${ENCODER}"
  --config   "${CONFIG}"
  --out      "${OUT_DIR}"
  --task     "${TASK}"
  --val-fold  "${VAL_FOLD}"
  --test-fold "${TEST_FOLD}"
  --amp      "${AMP}"
)

[[ -n "${DATA_DIR}" ]] && CMD+=(--data-dir "${DATA_DIR}")
[[ -n "${DUMP}" ]] && CMD+=(--dump "${DUMP}")

LOG_FILE="${OUT_DIR}/train.log"
echo "Command: ${CMD[*]}"
echo "Logging to: ${LOG_FILE}"
"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
