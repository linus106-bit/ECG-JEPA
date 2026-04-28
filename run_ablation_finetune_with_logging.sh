#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# ECG-JEPA Ablation Study - Downstream (Finetune) Pipeline with Result Logging
# =============================================================================
# Pretrain된 checkpoint들을 사용하여 downstream task 수행 후 결과를 JSON으로 기록
# Usage: bash run_ablation_finetune_with_logging.sh [NUM_GPUS] [DATASET]
# =============================================================================

NUM_GPUS="${1:-8}"
DATASET="${2:-ptb-xl}"  # ptb-xl or capture24
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRETRAIN_DIR="${SCRIPT_DIR}/results/pretrain"
FINETUNE_CONFIG="${SCRIPT_DIR}/configs/eval/linear.yaml"
LOG_DIR="${SCRIPT_DIR}/logs/ablation/finetune"
RESULTS_JSON="${SCRIPT_DIR}/results/ablation_results.json"

mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "ECG-JEPA Ablation Downstream Pipeline"
echo "=========================================="
echo "NUM_GPUS: ${NUM_GPUS}"
echo "DATASET: ${DATASET}"
echo "PRETRAIN_DIR: ${PRETRAIN_DIR}"
echo "LOG_DIR: ${LOG_DIR}"
echo "RESULTS_JSON: ${RESULTS_JSON}"
echo "=========================================="

# Dataset에 따른 설정
if [ "${DATASET}" = "ptb-xl" ]; then
    DATA_DIR="hf/ptb-xl"
    TASK="all"
elif [ "${DATASET}" = "capture24" ]; then
    DATA_DIR="/group-volume/datasets/openDB/Capture24"
    TASK="har"
    FINETUNE_CONFIG="${SCRIPT_DIR}/configs/eval/har_linear.yaml"
else
    echo "Error: Unknown dataset ${DATASET}"
    echo "Usage: bash run_ablation_finetune_with_logging.sh [NUM_GPUS] [DATASET]"
    echo "  DATASET: ptb-xl or capture24"
    exit 1
fi

# Pretrain 결과 디렉토리에서 ablation checkpoint 찾기
mapfile -t CHECKPOINT_DIRS < <(find "${PRETRAIN_DIR}" -maxdepth 1 -type d -name "m*" | sort)

if [ ${#CHECKPOINT_DIRS[@]} -eq 0 ]; then
    echo "Error: No ablation checkpoint directories found in ${PRETRAIN_DIR}"
    echo "Looking for directories matching pattern: m*"
    exit 1
fi

echo "Found ${#CHECKPOINT_DIRS[@]} checkpoint directories"
echo ""

# 각 checkpoint에 대해 finetune 수행
for ckpt_dir in "${CHECKPOINT_DIRS[@]}"; do
    exp_name=$(basename "${ckpt_dir}")

    # 가장 최근 checkpoint 찾기 (chkpt_*.pt 중 가장 큰 step)
    latest_ckpt=$(find "${ckpt_dir}" -name "chkpt_*.pt" -type f | sort -V | tail -1)

    if [ -z "${latest_ckpt}" ]; then
        echo "Warning: No checkpoint found in ${ckpt_dir}, skipping..."
        continue
    fi

    log_file="${LOG_DIR}/${exp_name}_${DATASET}_$(date +%Y%m%d_%H%M%S).log"

    echo "----------------------------------------"
    echo "Running finetune: ${exp_name}"
    echo "Checkpoint: ${latest_ckpt}"
    echo "Dataset: ${DATASET}"
    echo "Log: ${log_file}"
    echo "Started at: $(date)"
    echo "----------------------------------------"

    # Finetune config 업데이트 (백업 후 수정)
    CONFIG_BACKUP="${FINETUNE_CONFIG}.backup"
    cp "${FINETUNE_CONFIG}" "${CONFIG_BACKUP}"

    python -c "
import yaml

config_path = '${FINETUNE_CONFIG}'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

config['run']['encoder'] = '${latest_ckpt}'
config['run']['out_dir'] = 'results/finetune/ablation/${DATASET}/${exp_name}'
config['run']['task'] = '${TASK}'
config['dataset']['data_dir'] = '${DATA_DIR}'

with open(config_path, 'w') as f:
    yaml.dump(config, f)
"

    # Finetune 실행
    bash "${SCRIPT_DIR}/scripts/finetune_multi_gpu.sh" "${FINETUNE_CONFIG}" 2>&1 | tee "${log_file}"
    exit_code=${PIPESTATUS[0]}

    # Config 복원
    mv "${CONFIG_BACKUP}" "${FINETUNE_CONFIG}"

    if [ ${exit_code} -eq 0 ]; then
        echo "✓ Completed: ${exp_name}"

        # 결과 수집
        echo "Collecting results..."
        python "${SCRIPT_DIR}/collect_experiment_results.py" \
            --base-dir results/finetune/ablation \
            --output-json results/ablation_results.json \
            --output-markdown results/ablation_results.md
    else
        echo "✗ Failed: ${exp_name} (exit code: ${exit_code})"
    fi

    echo "Finished at: $(date)"
    echo ""
done

echo "=========================================="
echo "All finetune experiments completed!"
echo "Logs saved to: ${LOG_DIR}"
echo "Results saved to: ${RESULTS_JSON}"
echo "=========================================="

# 최종 결과 요약
echo ""
echo "Generating final summary..."
python "${SCRIPT_DIR}/collect_experiment_results.py" \
    --base-dir results/finetune/ablation \
    --output-json results/ablation_results.json \
    --output-markdown results/ablation_results.md

echo ""
echo "✓ All done! Check results/ablation_results.md for summary."
