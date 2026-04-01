#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# ECG-JEPA Finetune — Multi GPU (torchrun, single node)
# =============================================================================
# 사용법: bash scripts/finetune_multi_gpu.sh [CONFIG]
#
# 모든 학습 설정(out_dir, amp, encoder, task 등)은 YAML의 run: 섹션에서 관리합니다.
# 이 스크립트에서 수정할 항목:
#   CONFIG   — eval config 이름 (configs/eval/ 참고)
#   NUM_GPUS — 사용할 GPU 수
# =============================================================================

CONFIG="${1:-configs/eval/finetune.yaml}"
NUM_GPUS=8

# 사용할 GPU ID 지정 (전체 사용 시 주석 처리)
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

cd "$(dirname "${BASH_SOURCE[0]}")/.."
torchrun --standalone --nproc_per_node="${NUM_GPUS}" finetune.py --config "${CONFIG}"
