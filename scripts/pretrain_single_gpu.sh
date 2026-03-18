#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# ECG-JEPA Pretrain — Single GPU
# =============================================================================

# -----------------------------------------------------------------------------
# [1] 데이터 경로 설정
#     보유한 데이터셋의 주석을 해제하고 경로를 채워주세요.
#     config의 datasets: 항목과 키가 일치해야 합니다.
# -----------------------------------------------------------------------------

DATA_ARGS=(
  "ptb-xl=/path/to/ptb-xl.npy"
  # "mimic-iv-ecg=/path/to/mimic-iv-ecg.npy"
  # "code-15=/path/to/code-15.npy"
  # "georgia=/path/to/georgia.npy"
  # "chapman-shaoxing=/path/to/chapman-shaoxing.npy"
  # "ningbo=/path/to/ningbo.npy"
  # "cpsc=/path/to/cpsc.npy"
  # "cpsc-extra=/path/to/cpsc-extra.npy"
  # "ptb=/path/to/ptb.npz"
  # "st-petersburg=/path/to/st-petersburg.npy"
)

# -----------------------------------------------------------------------------
# [2] 학습 설정
# -----------------------------------------------------------------------------

# config 이름 (configs/pretrain/ 아래 yaml 파일명, 확장자 제외)
#   ptb-xl 단독:  ViTXS_ptbxl | ViTS_ptbxl
#   전체 데이터:  ViTXS_all   | ViTS_all   | ViTB_all
CONFIG="ViTS_ptbxl"

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
  --data  "${DATA_ARGS[@]}"
  --config "${CONFIG}"
  --out    "${OUT_DIR}"
  --amp    "${AMP}"
)

[[ -n "${CHECKPOINT}" ]] && CMD+=(--chkpt "${CHECKPOINT}")
[[ "${COMPILE}" == "true" ]] && CMD+=(--compile)

echo "Command: ${CMD[*]}"
"${CMD[@]}"
