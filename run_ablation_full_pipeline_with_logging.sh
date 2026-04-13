#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# ECG-JEPA Ablation Study - Full Pipeline with Result Logging
# =============================================================================
# 각 ablation config에 대해 pretrain → finetune을 순차적으로 수행하고 결과 기록
# Usage: bash run_ablation_full_pipeline_with_logging.sh [NUM_GPUS] [DATASET]
# =============================================================================

NUM_GPUS="${1:-8}"
DATASET="${2:-ptb-xl}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/configs/pretrain/ViT/ablation"
LOG_DIR="${SCRIPT_DIR}/logs/ablation/full_pipeline"
RESULTS_JSON="${SCRIPT_DIR}/results/ablation_results.json"

# Export NUM_GPUS for child scripts
export NUM_GPUS

mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "ECG-JEPA Ablation Full Pipeline"
echo "=========================================="
echo "NUM_GPUS: ${NUM_GPUS}"
echo "DATASET: ${DATASET}"
echo "CONFIG_DIR: ${CONFIG_DIR}"
echo "LOG_DIR: ${LOG_DIR}"
echo "RESULTS_JSON: ${RESULTS_JSON}"
echo "=========================================="

# Dataset에 따른 설정
if [ "${DATASET}" = "ptb-xl" ]; then
    DATA_DIR="hf/ptb-xl"
    TASK="all"
    FINETUNE_CONFIG="${SCRIPT_DIR}/configs/eval/linear.yaml"
elif [ "${DATASET}" = "capture24" ]; then
    DATA_DIR="/group-volume/datasets/openDB/Capture24"
    TASK="har"
    FINETUNE_CONFIG="${SCRIPT_DIR}/configs/eval/har_linear.yaml"
else
    echo "Error: Unknown dataset ${DATASET}"
    echo "Usage: bash run_ablation_full_pipeline_with_logging.sh [NUM_GPUS] [DATASET]"
    echo "  DATASET: ptb-xl or capture24"
    exit 1
fi

# 모든 ablation config 파일 목록 가져오기
mapfile -t CONFIGS < <(find "${CONFIG_DIR}" -name "*.yaml" -type f | sort)

echo "Found ${#CONFIGS[@]} ablation configs"
echo ""

# 실험 결과 추적용 JSON 초기화
python -c "
import json
from pathlib import Path
results_file = Path('${RESULTS_JSON}')
results_file.parent.mkdir(parents=True, exist_ok=True)
if not results_file.exists():
    with open(results_file, 'w') as f:
        json.dump({'metadata': {}, 'experiments': {}}, f, indent=2)
"

# 각 config에 대해 pretrain → finetune 수행
for config in "${CONFIGS[@]}"; do
    config_name=$(basename "${config}" .yaml)
    pretrain_log="${LOG_DIR}/${config_name}_pretrain_$(date +%Y%m%d_%H%M%S).log"
    finetune_log="${LOG_DIR}/${config_name}_finetune_$(date +%Y%m%d_%H%M%S).log"

    echo "=========================================="
    echo "Experiment: ${config_name}"
    echo "=========================================="

    # ========================================
    # Step 1: Pretrain
    # ========================================
    echo "Step 1: Pretrain"
    echo "Config: ${config}"
    echo "Log: ${pretrain_log}"
    echo "Started at: $(date)"

    bash "${SCRIPT_DIR}/scripts/pretrain_multi_gpu.sh" "${config}" 2>&1 | tee "${pretrain_log}"
    pretrain_exit=${PIPESTATUS[0]}

    if [ ${pretrain_exit} -ne 0 ]; then
        echo "✗ Pretrain failed: ${config_name} (exit code: ${pretrain_exit})"
        echo "Skipping finetune for this config..."
        continue
    fi

    echo "✓ Pretrain completed: ${config_name}"
    echo "Finished at: $(date)"
    echo ""

    # ========================================
    # Step 2: Find latest checkpoint
    # ========================================
    # Pretrain 결과 디렉토리 찾기 (config 이름과 매칭)
    pretrain_result_dir="${SCRIPT_DIR}/results/pretrain/${config_name}"

    if [ ! -d "${pretrain_result_dir}" ]; then
        echo "Warning: Pretrain result directory not found: ${pretrain_result_dir}"
        echo "Trying to find checkpoint in default location..."

        # config 파일에서 out_dir 읽기
        pretrain_result_dir=$(python -c "
import yaml
with open('${config}', 'r') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('run', {}).get('out_dir', ''))
" 2>/dev/null || echo "")

        if [ -z "${pretrain_result_dir}" ] || [ ! -d "${pretrain_result_dir}" ]; then
            echo "Error: Could not find pretrain result directory"
            continue
        fi
    fi

    # 가장 최근 checkpoint 찾기
    latest_ckpt=$(find "${pretrain_result_dir}" -name "chkpt_*.pt" -type f | sort -V | tail -1)

    if [ -z "${latest_ckpt}" ]; then
        echo "Error: No checkpoint found in ${pretrain_result_dir}"
        continue
    fi

    echo "Using checkpoint: ${latest_ckpt}"
    echo ""

    # ========================================
    # Step 3: Finetune
    # ========================================
    echo "Step 2: Finetune"
    echo "Dataset: ${DATASET}"
    echo "Log: ${finetune_log}"
    echo "Started at: $(date)"

    # Finetune config 백업 및 업데이트
    CONFIG_BACKUP="${FINETUNE_CONFIG}.backup"
    cp "${FINETUNE_CONFIG}" "${CONFIG_BACKUP}"

    python -c "
import yaml

config_path = '${FINETUNE_CONFIG}'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

config['run']['encoder'] = '${latest_ckpt}'
config['run']['out_dir'] = 'results/finetune/ablation/${DATASET}/${config_name}'
config['run']['task'] = '${TASK}'
config['dataset']['data_dir'] = '${DATA_DIR}'

with open(config_path, 'w') as f:
    yaml.dump(config, f)
"

    bash "${SCRIPT_DIR}/scripts/finetune_multi_gpu.sh" "${FINETUNE_CONFIG}" 2>&1 | tee "${finetune_log}"
    finetune_exit=${PIPESTATUS[0]}

    # Config 복원
    mv "${CONFIG_BACKUP}" "${FINETUNE_CONFIG}"

    if [ ${finetune_exit} -eq 0 ]; then
        echo "✓ Finetune completed: ${config_name}"

        # 결과 수집 및 업데이트
        echo "Collecting and updating results..."
        python "${SCRIPT_DIR}/collect_experiment_results.py" \
            --base-dir results/finetune/ablation \
            --output-json results/ablation_results.json \
            --output-markdown results/ablation_results.md
    else
        echo "✗ Finetune failed: ${config_name} (exit code: ${finetune_exit})"
    fi

    echo "Finished at: $(date)"
    echo "=========================================="
    echo ""
done

echo "=========================================="
echo "All experiments completed!"
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
