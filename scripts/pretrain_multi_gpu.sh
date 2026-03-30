#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# ECG-JEPA Pretrain — Multi GPU (torchrun, single node)
# =============================================================================
# Usage: bash scripts/pretrain_multi_gpu.sh [CONFIG]
#
# All settings (num_gpus, out_dir, amp, compile, checkpoint) are read from
# the run: section of the config YAML. Edit the YAML instead of this script.
#
# CONFIG: pretrain config name (configs/pretrain/ 참고)
#   ptb-xl 단독 → ViTXS_ptbxl | ViTS_ptbxl
#   전체 데이터  → ViTXS_all   | ViTS_all   | ViTB_all
# =============================================================================

CONFIG="${1:-ViTS_ptbxl}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

# Read num_gpus and out_dir from YAML run: section
_YAML=$(find configs/pretrain -name "${CONFIG}.yaml" | head -1)
if [[ -z "${_YAML}" ]]; then
  echo "Error: config file not found for '${CONFIG}' in configs/pretrain/" >&2
  exit 1
fi
NUM_GPUS=$(python -c "import yaml; d=yaml.safe_load(open('${_YAML}')); print(d.get('run',{}).get('num_gpus',8))")
OUT_DIR=$(python -c "import yaml; d=yaml.safe_load(open('${_YAML}')); print(d.get('run',{}).get('out_dir','pretrain/${CONFIG}'))")

mkdir -p "${OUT_DIR}"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="${OUT_DIR}/train_${TIMESTAMP}.log"
echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Config:     ${_YAML}"
echo "Logging to: ${LOG_FILE}"
{
  echo "=== Started at: $(date '+%Y-%m-%d %H:%M:%S') ==="
  torchrun --standalone --nproc_per_node="${NUM_GPUS}" pretrain.py --config "${CONFIG}"
} 2>&1 | tee "${LOG_FILE}"
