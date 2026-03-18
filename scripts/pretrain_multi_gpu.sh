#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# ECG-JEPA Pretrain — Multi GPU (torchrun, single node)
# =============================================================================

# 어디서든 실행 가능하도록 프로젝트 루트로 이동
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

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
# [2] GPU 설정
# -----------------------------------------------------------------------------

# 사용할 GPU 수 (서버의 실제 GPU 수 이하로 설정)
NUM_GPUS=8

# 사용할 GPU ID 지정 (전체 사용 시 주석 처리)
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# -----------------------------------------------------------------------------
# [3] 학습 설정
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
# [4] 실행
# -----------------------------------------------------------------------------

mkdir -p "${OUT_DIR}"

CMD=(
  torchrun
    --standalone
    --nproc_per_node="${NUM_GPUS}"
  pretrain.py
    --data  "${DATA_ARGS[@]}"
    --config "${CONFIG}"
    --out    "${OUT_DIR}"
    --amp    "${AMP}"
)

[[ -n "${CHECKPOINT}" ]] && CMD+=(--chkpt "${CHECKPOINT}")
[[ "${COMPILE}" == "true" ]] && CMD+=(--compile)

echo "Command: ${CMD[*]}"
"${CMD[@]}"
