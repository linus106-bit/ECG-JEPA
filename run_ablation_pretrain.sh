#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# ECG-JEPA Ablation Study - Pretrain Pipeline
# =============================================================================
# 모든 ablation config에 대해 순차적으로 pretrain 수행
# Usage: bash run_ablation_pretrain.sh [NUM_GPUS]
# =============================================================================

NUM_GPUS="${1:-8}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/configs/pretrain/ViT/ablation"
LOG_DIR="${SCRIPT_DIR}/logs/ablation/pretrain"

mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "ECG-JEPA Ablation Pretrain Pipeline"
echo "=========================================="
echo "NUM_GPUS: ${NUM_GPUS}"
echo "CONFIG_DIR: ${CONFIG_DIR}"
echo "LOG_DIR: ${LOG_DIR}"
echo "=========================================="

# 모든 ablation config 파일 목록 가져오기
mapfile -t CONFIGS < <(find "${CONFIG_DIR}" -name "*.yaml" -type f | sort)

echo "Found ${#CONFIGS[@]} ablation configs"
echo ""

# 각 config에 대해 pretrain 수행
for config in "${CONFIGS[@]}"; do
    config_name=$(basename "${config}" .yaml)
    log_file="${LOG_DIR}/${config_name}_$(date +%Y%m%d_%H%M%S).log"

    echo "----------------------------------------"
    echo "Running pretrain: ${config_name}"
    echo "Config: ${config}"
    echo "Log: ${log_file}"
    echo "Started at: $(date)"
    echo "----------------------------------------"

    # Pretrain 실행
    bash "${SCRIPT_DIR}/scripts/pretrain_multi_gpu.sh" "${config}" "${NUM_GPUS}" 2>&1 | tee "${log_file}"

    exit_code=${PIPESTATUS[0]}

    if [ ${exit_code} -eq 0 ]; then
        echo "✓ Completed: ${config_name}"
    else
        echo "✗ Failed: ${config_name} (exit code: ${exit_code})"
        # 실패해도 다음 config 계속 실행
    fi

    echo "Finished at: $(date)"
    echo ""
done

echo "=========================================="
echo "All pretrain experiments completed!"
echo "Logs saved to: ${LOG_DIR}"
echo "=========================================="
