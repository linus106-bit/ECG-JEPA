# ECG-JEPA Ablation Experiment Guide

이 문서는 ECG-JEPA의 ablation 실험을 수행하기 위한 가이드입니다.

## 📋 목차

- [개요](#개요)
- [실험 설정](#실험-설정)
- [스크립트 사용법](#스크립트-사용법)
- [추천 워크플로우](#추천-워크플로우)
- [결과 파일 위치](#결과-파일-위치)
- [팁 및 문제 해결](#팁-및-문제-해결)

---

## 개요

### 실험 구성

- **총 54개의 ablation config**
  - Capture24용: 27개
  - PTB-XL용: 27개

- **Ablation 파라미터**
  - `m` (encoder_momentum): 0.25, 0.35, 0.45
  - `b` (min_block_size): 5, 10, 12
  - `p` (patch_size): 10, 20, 25

- **모델 아키텍처**: ViT-Base (ViTB)
  - dim: 768
  - depth: 12
  - num_heads: 12

### 데이터셋

| Dataset | Channels | Sampling Rate | Task |
|---------|----------|---------------|------|
| Capture24 | 3 (x, y, z) | 100 Hz | HAR (Human Activity Recognition) |
| PTB-XL | 12 (ECG leads) | 100 Hz | ECG Classification |

---

## 실험 설정

### Config 파일 위치

```
configs/pretrain/ViT/ablation/
├── m0.25_b5_p10.yaml          # Capture24용
├── m0.25_b5_p10_ptb-xl.yaml   # PTB-XL용
├── m0.25_b5_p20.yaml
├── m0.25_b5_p20_ptb-xl.yaml
└── ... (총 54개)
```

### Checkpoint 저장 구조

```
results/pretrain/
├── ViTB_capture24/
│   ├── m0.25_b5_p10/
│   │   ├── chkpt_10000.pt
│   │   ├── chkpt_20000.pt
│   │   └── train_*.log
│   ├── m0.25_b5_p20/
│   └── ... (27개)
└── ViTB_ptb-xl/
    ├── m0.25_b5_p10/
    ├── m0.25_b5_p20/
    └── ... (27개)
```

---

## 스크립트 사용법

### 1. Pretrain만 수행

모든 ablation config에 대해 pretrain만 수행합니다.

```bash
# 기본 사용법 (8 GPU)
bash run_ablation_pretrain.sh 8

# GPU 수 지정
bash run_ablation_pretrain.sh 4
```

**동작:**
- `configs/pretrain/ViT/ablation/` 내 모든 config 파일 (54개)에 대해 순차적으로 pretrain
- Checkpoint 저장: `results/pretrain/ViTB_{dataset}/{config_name}/`
- 로그 저장: `logs/ablation/pretrain/`

---

### 2. Finetune만 수행

이미 pretrain된 checkpoint들을 사용하여 downstream task를 수행합니다.

**3가지 Finetune 방식:**

1. **Linear Evaluation** (`linear`)
   - Encoder frozen, classifier만 학습
   - 빠르고 baseline 성능 확인용
   - `frozen: True`

2. **Full Finetuning** (`finetune`)
   - 전체 모델 fine-tuning
   - 더 높은 성능 기대
   - `frozen: False`

3. **2-Stage Finetuning** (`2stage`)
   - Linear evaluation 후 full finetuning
   - Linear checkpoint에서 시작해서 전체 모델 fine-tuning
   - 최고 성능 기대
   - `frozen: False`

**사용법:**

```bash
# 모든 finetune 방식 (linear, finetune, 2stage)
bash run_ablation_finetune.sh 8 ptb-xl all

# 특정 방식만
bash run_ablation_finetune.sh 8 ptb-xl linear
bash run_ablation_finetune.sh 8 ptb-xl finetune
bash run_ablation_finetune.sh 8 ptb-xl 2stage

# Capture24로 finetune
bash run_ablation_finetune.sh 8 capture24 all
```

**동작:**
- `results/pretrain/` 내 모든 checkpoint 디렉토리를 찾음
- 각 checkpoint의 최신 checkpoint를 사용해서 finetune
- 결과 저장: `results/finetune/ablation/{dataset}/{mode}/{config_name}/`
- **각 실험 완료 후 자동으로 결과 집계**
  - `results/ablation_results.json`
  - `results/ablation_results.md`

---

### 3. 전체 파이프라인 (Pretrain + Finetune)

각 config에 대해 pretrain → finetune을 순차적으로 수행합니다.

```bash
# PTB-XL로 전체 실험
bash run_ablation_full_pipeline.sh 8 ptb-xl

# Capture24로 전체 실험
bash run_ablation_full_pipeline.sh 8 capture24
```

**동작:**
- 각 config에 대해: Pretrain → Finetune → 결과 집계
- 실험 하나씩 순차적으로 완료
- **각 실험 완료 후 바로 결과 확인 가능**

---

### 4. 결과 수집만 수행

이미 완료된 실험들의 결과를 다시 수집합니다.

```bash
python collect_experiment_results.py \
    --base-dir results/finetune/ablation \
    --output-json results/ablation_results.json \
    --output-markdown results/ablation_results.md
```

---

## 추천 워크플로우

### 옵션 A: 데이터셋별로 순차 실행 (권장) ⭐

한 데이터셋을 완료하고 다음 데이터셋으로 넘어가는 방식입니다.

```bash
# 1. Capture24로 모든 ablation pretrain + finetune
bash run_ablation_full_pipeline.sh 8 capture24

# 2. PTB-XL로 모든 ablation pretrain + finetune
bash run_ablation_full_pipeline.sh 8 ptb-xl
```

**장점:**
- 한 데이터셋의 모든 결과를 먼저 확보
- 중간에 문제 발생 시 다른 데이터셋에 영향 없음

---

### 옵션 B: Pretrain 먼저 모두 완료, 그 다음 Finetune

모든 pretrain을 완료한 후 finetune을 수행합니다.

```bash
# 1. 모든 ablation pretrain (Capture24 + PTB-XL)
bash run_ablation_pretrain.sh 8

# 2. Capture24로 finetune
bash run_ablation_finetune.sh 8 capture24

# 3. PTB-XL로 finetune
bash run_ablation_finetune.sh 8 ptb-xl
```

**장점:**
- Pretrain 단계에서 문제가 있으면 finetune 전에 수정 가능
- GPU 자원을 효율적으로 사용

---

## 결과 파일 위치

### Pretrain 결과

```
results/pretrain/
├── ViTB_capture24/
│   ├── m0.25_b5_p10/
│   │   ├── chkpt_10000.pt
│   │   ├── chkpt_20000.pt
│   │   └── train_*.log
│   ├── m0.25_b5_p20/
│   └── ... (27개)
└── ViTB_ptb-xl/
    ├── m0.25_b5_p10/
    └── ... (27개)
```

### Finetune 결과

```
results/finetune/ablation/
├── ptb-xl/
│   ├── linear/
│   │   ├── m0.25_b5_p10/
│   │   │   ├── all_eval_results.json      # 평가 결과
│   │   │   ├── all_predictions.npz        # 예측값
│   │   │   ├── all_best_chkpt.pt          # Best checkpoint
│   │   │   └── train_*.log                # Training log
│   │   └── ... (27개)
│   ├── finetune/
│   │   ├── m0.25_b5_p10/
│   │   └── ... (27개)
│   └── 2stage/
│       ├── m0.25_b5_p10/
│       └── ... (27개)
└── capture24/
    ├── linear/
    ├── finetune/
    └── 2stage/
```

### 집계 결과

```
results/
├── ablation_results.json    # 모든 실험 결과 (JSON)
└── ablation_results.md       # 결과 요약 테이블 (Markdown)
```

**ablation_results.json 예시:**

```json
{
  "metadata": {
    "collected_at": "2026-04-06T14:30:00",
    "base_dir": "results/finetune/ablation"
  },
  "experiments": {
    "ptb-xl": {
      "linear": {
        "m0.25_b5_p10": {
          "all": {
            "dataset_type": "ecg",
            "single_label": false,
            "test_auc": 0.9187,
            "val_auc": 0.9234,
            "best_epoch_or_step": 35,
            "ablation_params": {
              "encoder_momentum": 0.25,
              "min_block_size": 5,
              "patch_size": 10
            }
          }
        }
      },
      "finetune": {
        "m0.25_b5_p10": {
          "all": {
            "test_auc": 0.9245,
            "val_auc": 0.9301,
            ...
          }
        }
      },
      "2stage": {
        "m0.25_b5_p10": {
          "all": {
            "test_auc": 0.9312,
            "val_auc": 0.9367,
            ...
          }
        }
      }
    },
    "capture24": {
      "linear": {
        "m0.25_b5_p10": {
          "har": {
            "dataset_type": "har",
            "single_label": true,
            "test_f1": 0.8765,
            "test_acc": 0.8912,
            "val_f1": 0.8823
          }
        }
      },
      "finetune": { ... },
      "2stage": { ... }
    }
  }
}
```

**ablation_results.md 예시:**

```markdown
# Ablation Study Results

Collected at: 2026-04-06T14:30:00

## Dataset: ptb-xl

### Finetune Mode: linear

#### Task: all

| Config | Momentum | Block Size | Patch Size | Val AUC | Test AUC | Best Step |
|--------|----------|------------|------------|---------|----------|-----------|
| m0.25_b5_p10 | 0.25 | 5 | 10 | 0.9234 | 0.9187 | 35 |
| m0.25_b5_p20 | 0.25 | 5 | 20 | 0.9301 | 0.9256 | 42 |

### Finetune Mode: finetune

#### Task: all

| Config | Momentum | Block Size | Patch Size | Val AUC | Test AUC | Best Step |
|--------|----------|------------|------------|---------|----------|-----------|
| m0.25_b5_p10 | 0.25 | 5 | 10 | 0.9301 | 0.9245 | 38 |
| m0.25_b5_p20 | 0.25 | 5 | 20 | 0.9367 | 0.9312 | 45 |

### Finetune Mode: 2stage

#### Task: all

| Config | Momentum | Block Size | Patch Size | Val AUC | Test AUC | Best Step |
|--------|----------|------------|------------|---------|----------|-----------|
| m0.25_b5_p10 | 0.25 | 5 | 10 | 0.9367 | 0.9312 | 40 |
| m0.25_b5_p20 | 0.25 | 5 | 20 | 0.9423 | 0.9378 | 47 |
```

---

## 팁 및 문제 해결

### 특정 config만 실행하기

```bash
# 특정 config만 pretrain
bash scripts/pretrain_multi_gpu.sh configs/pretrain/ViT/ablation/m0.25_b5_p10.yaml

# 특정 checkpoint로 finetune
python -c "
import yaml
config = yaml.safe_load(open('configs/eval/linear.yaml'))
config['run']['encoder'] = 'results/pretrain/ViTB_capture24/m0.25_b5_p10/chkpt_50000.pt'
config['run']['out_dir'] = 'results/finetune/test'
yaml.dump(config, open('configs/eval/linear.yaml', 'w'))
"
bash scripts/finetune_multi_gpu.sh configs/eval/linear.yaml
```

### 진행 상황 확인

```bash
# 로그 실시간 확인
tail -f logs/ablation/full_pipeline/m0.25_b5_p10_pretrain_*.log

# 결과 확인
cat results/ablation_results.md

# 특정 실험 결과 확인
cat results/finetune/ablation/ptb-xl/m0.25_b5_p10/all_eval_results.json
```

### 실패한 실험 재실행

실험이 실패해도 스크립트는 다음 config로 계속 진행합니다. 실패한 실험만 다시 실행하려면:

```bash
# 특정 config만 다시 실행
bash scripts/pretrain_multi_gpu.sh configs/pretrain/ViT/ablation/m0.25_b5_p10.yaml
```

### GPU 메모리 부족 시

```bash
# GPU 수 줄이기
bash run_ablation_full_pipeline.sh 4 ptb-xl

# 또는 특정 GPU만 사용
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash run_ablation_full_pipeline.sh 4 ptb-xl
```

### 중간에 중단된 경우

스크립트는 이미 완료된 checkpoint를 자동으로 찾아서 사용합니다. 중단된 지점에서 다시 시작하면 됩니다.

```bash
# 그냥 다시 실행
bash run_ablation_full_pipeline.sh 8 ptb-xl
```

---

## 실험 결과 분석

### JSON 결과에서 특정 metric 추출

```python
import json

with open('results/ablation_results.json', 'r') as f:
    results = json.load(f)

# PTB-XL 결과에서 best model 찾기
ptbxl_results = results['experiments']['ptb-xl']
best_config = max(ptbxl_results.items(),
                  key=lambda x: x[1]['all']['test_auc'])
print(f"Best config: {best_config[0]}")
print(f"Test AUC: {best_config[1]['all']['test_auc']:.4f}")
```

### Markdown 결과 확인

```bash
# 전체 결과 테이블 확인
cat results/ablation_results.md

# 특정 데이터셋만 확인
grep -A 20 "Dataset: ptb-xl" results/ablation_results.md
```

---

## 문의 및 이슈

문제가 발생하면 다음을 확인하세요:

1. 로그 파일: `logs/ablation/`
2. Checkpoint 존재 여부: `results/pretrain/`
3. GPU 메모리: `nvidia-smi`
4. 디스크 공간: `df -h`

---

**마지막 업데이트**: 2026-04-06
