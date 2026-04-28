#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# ECG-JEPA Ablation Study - Downstream (Finetune) Pipeline
# =============================================================================
# Pretrain된 checkpoint들을 사용하여 downstream task 수행 후 결과를 자동 집계
# 3가지 finetune 방식 지원: linear, finetune, 2stage
# Usage: bash run_ablation_finetune.sh [NUM_GPUS] [DATASET] [MODE]
# =============================================================================

NUM_GPUS="${1:-8}"
DATASET="${2:-ptb-xl}"  # ptb-xl or capture24
MODE="${3:-all}"  # linear, finetune, 2stage, or all (default: all)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRETRAIN_DIR="${SCRIPT_DIR}/results/pretrain"
LOG_DIR="${SCRIPT_DIR}/logs/ablation/finetune"
RESULTS_JSON="${SCRIPT_DIR}/results/ablation_results.json"

mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "ECG-JEPA Ablation Downstream Pipeline"
echo "=========================================="
echo "NUM_GPUS: ${NUM_GPUS}"
echo "DATASET: ${DATASET}"
echo "MODE: ${MODE}"
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
else
    echo "Error: Unknown dataset ${DATASET}"
    echo "Usage: bash run_ablation_finetune.sh [NUM_GPUS] [DATASET] [MODE]"
    echo "  DATASET: ptb-xl or capture24"
    echo "  MODE: linear, finetune, 2stage, or all"
    exit 1
fi

# Finetune 방식 설정
FINETUNE_MODES=()
if [ "${MODE}" = "all" ]; then
    FINETUNE_MODES=("linear" "finetune" "2stage")
elif [ "${MODE}" = "linear" ]; then
    FINETUNE_MODES=("linear")
elif [ "${MODE}" = "finetune" ]; then
    FINETUNE_MODES=("finetune")
elif [ "${MODE}" = "2stage" ]; then
    FINETUNE_MODES=("2stage")
else
    echo "Error: Unknown mode ${MODE}"
    echo "Usage: bash run_ablation_finetune.sh [NUM_GPUS] [DATASET] [MODE]"
    echo "  MODE: linear, finetune, 2stage, or all"
    exit 1
fi

echo "Finetune modes: ${FINETUNE_MODES[*]}"
echo ""

# Pretrain 결과 디렉토리에서 ablation checkpoint 찾기
# 예: results/pretrain/ViTB_capture24/m0.25_b5_p10/
mapfile -t CHECKPOINT_DIRS < <(find "${PRETRAIN_DIR}" -maxdepth 2 -type d -name "m*" | sort)

if [ ${#CHECKPOINT_DIRS[@]} -eq 0 ]; then
    echo "Error: No ablation checkpoint directories found in ${PRETRAIN_DIR}"
    echo "Looking for directories matching pattern: */m*"
    exit 1
fi

echo "Found ${#CHECKPOINT_DIRS[@]} checkpoint directories"
echo ""

# 결과 수집 스크립트 확인
COLLECT_SCRIPT="${SCRIPT_DIR}/collect_experiment_results.py"
if [ ! -f "${COLLECT_SCRIPT}" ]; then
    echo "Error: Result collection script not found: ${COLLECT_SCRIPT}"
    exit 1
fi

# 각 finetune 방식에 대해 수행
for FINETUNE_MODE in "${FINETUNE_MODES[@]}"; do
    echo "=========================================="
    echo "Finetune Mode: ${FINETUNE_MODE}"
    echo "=========================================="
    echo ""

    # Finetune config 선택
    if [ "${FINETUNE_MODE}" = "linear" ]; then
        if [ "${DATASET}" = "ptb-xl" ]; then
            BASE_CONFIG="${SCRIPT_DIR}/configs/eval/linear.yaml"
        else
            BASE_CONFIG="${SCRIPT_DIR}/configs/eval/har_linear.yaml"
        fi
    elif [ "${FINETUNE_MODE}" = "finetune" ]; then
        BASE_CONFIG="${SCRIPT_DIR}/configs/eval/finetune.yaml"
    elif [ "${FINETUNE_MODE}" = "2stage" ]; then
        BASE_CONFIG="${SCRIPT_DIR}/configs/eval/finetune_after_linear.yaml"
    fi

    # 각 checkpoint에 대해 finetune 수행
    for ckpt_dir in "${CHECKPOINT_DIRS[@]}"; do
        exp_name=$(basename "${ckpt_dir}")

        # 가장 최근 checkpoint 찾기 (chkpt_*.pt 중 가장 큰 step)
        latest_ckpt=$(find "${ckpt_dir}" -name "chkpt_*.pt" -type f | sort -V | tail -1)

        if [ -z "${latest_ckpt}" ]; then
            echo "Warning: No checkpoint found in ${ckpt_dir}, skipping..."
            continue
        fi

        log_file="${LOG_DIR}/${exp_name}_${DATASET}_${FINETUNE_MODE}_$(date +%Y%m%d_%H%M%S).log"

        echo "----------------------------------------"
        echo "Running finetune: ${exp_name} (${FINETUNE_MODE})"
        echo "Checkpoint: ${latest_ckpt}"
        echo "Dataset: ${DATASET}"
        echo "Mode: ${FINETUNE_MODE}"
        echo "Log: ${log_file}"
        echo "Started at: $(date)"
        echo "----------------------------------------"

        # Config 백업
        CONFIG_BACKUP="${BASE_CONFIG}.backup"
        cp "${BASE_CONFIG}" "${CONFIG_BACKUP}"

        # Finetune config 업데이트
        python -c "
import yaml

config_path = '${BASE_CONFIG}'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

config['run']['encoder'] = '${latest_ckpt}'
config['run']['out_dir'] = 'results/finetune/ablation/${DATASET}/${FINETUNE_MODE}/${exp_name}'
config['run']['task'] = '${TASK}'
config['dataset']['data_dir'] = '${DATA_DIR}'

with open(config_path, 'w') as f:
    yaml.dump(config, f)
"

        # Finetune 실행
        bash "${SCRIPT_DIR}/scripts/finetune_multi_gpu.sh" "${BASE_CONFIG}" 2>&1 | tee "${log_file}"
        exit_code=${PIPESTATUS[0]}

        # Config 복원
        mv "${CONFIG_BACKUP}" "${BASE_CONFIG}"

        if [ ${exit_code} -eq 0 ]; then
            echo "✓ Completed: ${exp_name} (${FINETUNE_MODE})"

            # 결과 수집 및 집계
            echo "Collecting and aggregating results..."
            python "${COLLECT_SCRIPT}" \
                --base-dir results/finetune/ablation \
                --output-json results/ablation_results.json \
                --output-markdown results/ablation_results.md

            # 현재 실험 결과 미리보기
            result_file="results/finetune/ablation/${DATASET}/${FINETUNE_MODE}/${exp_name}/${TASK}_eval_results.json"
            if [ -f "${result_file}" ]; then
                echo "Result preview:"
                python -c "
import json
with open('${result_file}', 'r') as f:
    data = json.load(f)
    if data.get('single_label'):
        print(f\"  Test F1: {data.get('test_f1', 'N/A'):.4f}\")
        print(f\"  Test Acc: {data.get('test_acc', 'N/A'):.4f}\")
    else:
        print(f\"  Test AUC: {data.get('test_auc', 'N/A'):.4f}\")
    print(f\"  Best Step: {data.get('best_epoch_or_step', 'N/A')}\")
"
            fi
        else
            echo "✗ Failed: ${exp_name} (${FINETUNE_MODE}) (exit code: ${exit_code})"
        fi

        echo "Finished at: $(date)"
        echo ""
    done
done

echo "=========================================="
echo "All finetune experiments completed!"
echo "Logs saved to: ${LOG_DIR}"
echo "Results saved to: ${RESULTS_JSON}"
echo "=========================================="

# 최종 결과 요약
echo ""
echo "Generating final summary..."
python "${COLLECT_SCRIPT}" \
    --base-dir results/finetune/ablation \
    --output-json results/ablation_results.json \
    --output-markdown results/ablation_results.md

echo ""
echo "✓ All done! Check results/ablation_results.md for summary."
