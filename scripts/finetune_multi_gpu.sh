#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# ECG-JEPA Finetune — Multi GPU (torchrun, single node)
# =============================================================================
# Usage: bash scripts/finetune_multi_gpu.sh [CONFIG]
#
# All settings (num_gpus, out_dir, amp, encoder, task, dataset_type) are read
# from the run: section of the config YAML. Edit the YAML instead of this script.
#
# CONFIG: eval config name (configs/eval/ 참고)
#   linear                → linear probing (인코더 고정)
#   finetune              → end-to-end fine-tuning
#   finetune_after_linear → linear 학습 후 fine-tuning
#   har_linear            → HAR linear evaluation
# =============================================================================

CONFIG="${1:-finetune}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

# Read num_gpus and out_dir from YAML run: section
_YAML="configs/eval/${CONFIG}.yaml"
if [[ ! -f "${_YAML}" ]]; then
  echo "Error: config file not found: ${_YAML}" >&2
  exit 1
fi
NUM_GPUS=$(python -c "import yaml; d=yaml.safe_load(open('${_YAML}')); print(d.get('run',{}).get('num_gpus',8))")
OUT_DIR=$(python -c "import yaml; d=yaml.safe_load(open('${_YAML}')); print(d.get('run',{}).get('out_dir','finetune/${CONFIG}'))")

mkdir -p "${OUT_DIR}"
TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"
LOG_FILE="${OUT_DIR}/train_${TIMESTAMP}.log"
echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Config:     ${_YAML}"
echo "Logging to: ${LOG_FILE}"
{
  echo "=== Started at: $(date '+%Y-%m-%d %H:%M:%S') ==="
  torchrun --standalone --nproc_per_node="${NUM_GPUS}" finetune.py --config "${CONFIG}"
} 2>&1 | tee "${LOG_FILE}"
