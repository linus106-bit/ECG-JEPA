#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# ECG-JEPA Pretrain — Multi GPU (torchrun, single node)
# =============================================================================

# -----------------------------------------------------------------------------
# 설정 (자주 수정하는 항목)
# -----------------------------------------------------------------------------

# pretrain config 이름 (configs/pretrain/ 참고)
#   ptb-xl 단독 → ViTXS_ptbxl | ViTS_ptbxl
#   전체 데이터  → ViTXS_all   | ViTS_all   | ViTB_all
CONFIG="ViTS_ptbxl"

# 사용할 GPU 수 (서버의 실제 GPU 수 이하로 설정)
NUM_GPUS=8

# 사용할 GPU ID 지정 (전체 사용 시 주석 처리)
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 출력 디렉토리 (체크포인트 저장 위치)
OUT_DIR="pretrain/${CONFIG}"

# 이전 체크포인트에서 재개할 경우 경로를 지정 (없으면 빈 문자열)
CHECKPOINT=""

# 정밀도: float32 | bfloat16
AMP="bfloat16"

# torch.compile 사용 여부: true | false
COMPILE=false

# =============================================================================
# 이 아래는 수정하지 않아도 됩니다
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

mkdir -p "${OUT_DIR}"

CMD=(
  torchrun
    --standalone
    --nproc_per_node="${NUM_GPUS}"
  pretrain.py
    --config "${CONFIG}"
    --out    "${OUT_DIR}"
    --amp    "${AMP}"
)

[[ -n "${CHECKPOINT}" ]] && CMD+=(--chkpt "${CHECKPOINT}")
[[ "${COMPILE}" == "true" ]] && CMD+=(--compile)

TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="${OUT_DIR}/train_${TIMESTAMP}.log"
echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Command: ${CMD[*]}"
echo "Logging to: ${LOG_FILE}"
{ echo "=== Started at: $(date '+%Y-%m-%d %H:%M:%S') ==="; "${CMD[@]}"; } 2>&1 | tee "${LOG_FILE}"
