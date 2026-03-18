#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# ECG-JEPA Pretrain — Multi GPU (torchrun, single node)
# =============================================================================

# 어디서든 실행 가능하도록 프로젝트 루트로 이동
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

# -----------------------------------------------------------------------------
# [1] CONFIG 선택
#     CONFIG에 지정한 yaml 파일이 사용할 데이터셋과 비율을 결정합니다.
#     DATA_ARGS는 그 yaml에 명시된 데이터셋의 파일 경로를 제공하는 역할입니다.
#
#     규칙:
#       - yaml의 datasets: 에 있는 키는 DATA_ARGS에 반드시 존재해야 합니다. (없으면 에러)
#       - DATA_ARGS에 yaml에 없는 키를 넣어도 무시됩니다.
#       → CONFIG와 DATA_ARGS는 항상 함께 맞춰서 수정하세요.
#
#     사용 가능한 config (configs/pretrain/ 참고):
#       ptb-xl 단독 → ViTXS_ptbxl | ViTS_ptbxl
#       전체 데이터  → ViTXS_all   | ViTS_all   | ViTB_all
# -----------------------------------------------------------------------------

CONFIG="ViTS_ptbxl"

# -----------------------------------------------------------------------------
# [2] 데이터 경로 설정
#     위에서 선택한 CONFIG의 yaml datasets: 항목에 있는 키만 활성화하세요.
#
#     ViTS_ptbxl / ViTXS_ptbxl 사용 시: ptb-xl 만 필요
#     ViTS_all / ViTB_all 사용 시:       아래 전부 필요
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
# [3] GPU 설정
# -----------------------------------------------------------------------------

# 사용할 GPU 수 (서버의 실제 GPU 수 이하로 설정)
NUM_GPUS=8

# 사용할 GPU ID 지정 (전체 사용 시 주석 처리)
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# -----------------------------------------------------------------------------
# [4] 학습 설정
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
# [5] 실행
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
