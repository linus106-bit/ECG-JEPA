#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# ECG-JEPA Pretrain — Single GPU
# =============================================================================

# 어디서든 실행 가능하도록 프로젝트 루트로 이동
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

# -----------------------------------------------------------------------------
# [1] CONFIG 선택
#     데이터셋 경로와 비율은 모두 yaml 파일 안에서 관리합니다.
#     여기서는 어떤 yaml을 쓸지만 선택하세요.
#
#     사용 가능한 config (configs/pretrain/ 참고):
#       ptb-xl 단독 → ViTXS_ptbxl | ViTS_ptbxl
#       전체 데이터  → ViTXS_all   | ViTS_all   | ViTB_all
# -----------------------------------------------------------------------------

CONFIG="ViTS_ptbxl"

# -----------------------------------------------------------------------------
# [2] 학습 설정
# -----------------------------------------------------------------------------

# 출력 디렉토리 (체크포인트 저장 위치)
OUT_DIR="pretrain/${CONFIG}"

# 정밀도: float32 | bfloat16
AMP="bfloat16"

# 이전 체크포인트에서 재개할 경우 경로를 지정 (없으면 빈 문자열)
CHECKPOINT=""

# torch.compile 사용 여부: true | false
COMPILE=false

# -----------------------------------------------------------------------------
# [3] 실행
# -----------------------------------------------------------------------------

mkdir -p "${OUT_DIR}"

CMD=(
  python pretrain.py
  --config "${CONFIG}"
  --out    "${OUT_DIR}"
  --amp    "${AMP}"
)

[[ -n "${CHECKPOINT}" ]] && CMD+=(--chkpt "${CHECKPOINT}")
[[ "${COMPILE}" == "true" ]] && CMD+=(--compile)

LOG_FILE="${OUT_DIR}/train.log"
echo "Command: ${CMD[*]}"
echo "Logging to: ${LOG_FILE}"
"${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
