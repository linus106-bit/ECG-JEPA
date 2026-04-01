# ECG-JEPA GUIDELINES (Updated Workflow)

이 문서는 현재 저장소 상태(스크립트/설정 파일 기준)에 맞춘 **실행 가이드**입니다.
기존 README의 일부 예시는 구버전 `.npy dump` 중심이라, 여기서는 현재 권장되는
**HuggingFace Dataset(parquet shard) 기반 파이프라인**을 기준으로 정리합니다.

---

## 1) 환경 준비

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> 참고: `scripts/convert_to_hf_dataset.py`를 사용하려면 `datasets` 패키지가 필요합니다.
> 기본 `requirements.txt`에 없다면 다음을 추가로 설치하세요.

```bash
pip install datasets
```

---

## 2) 데이터 준비 전략 (권장)

현재 학습 코드(`pretrain.py`, `finetune.py`)는 다음 입력 형식을 지원합니다.

- **권장:** HuggingFace dataset 디렉토리 (`load_dataset(data_dir)`로 로드)
- 레거시 지원: `.npy`, `.npz`

실제로는 HF 포맷이 split/train/val/test 관리와 멀티데이터셋 조합에서 가장 편합니다.

### 2.1 원본 데이터 다운로드

저장소 내 스크립트 사용:

```bash
bash scripts/download_datasets.sh /path/to/raw-data
```

이 스크립트는 PTB-XL, MIMIC-IV-ECG, CinC2021 묶음, CODE-15를 다운로드합니다.

### 2.2 HF dataset으로 변환

핵심 스크립트:

- `scripts/convert_to_hf_dataset.py`

예시:

```bash
# PTB-XL (finetune/eval용 train/val/test split 생성)
python -m scripts.convert_to_hf_dataset \
  --data-dir /path/to/raw-data/ptb-xl \
  --dataset ptb-xl \
  --task all \
  --out /path/to/hf/ptb-xl \
  --verbose

# MIMIC-IV-ECG (pretrain용 train split)
python -m scripts.convert_to_hf_dataset \
  --data-dir /path/to/raw-data/mimic-iv-ecg \
  --dataset mimic-iv-ecg \
  --out /path/to/hf/mimic-iv-ecg \
  --verbose

# Capture-24 (HAR train/test split)
python -m scripts.convert_to_hf_dataset \
  --data-dir /path/to/raw-data/capture24 \
  --dataset capture-24 \
  --out /path/to/hf/capture24 \
  --verbose
```

### 2.3 전처리(선택)

구버전 dump(`.npy/.npz`)를 쓰는 경우에만 `scripts/preprocess_data_dump.py`가 직접 필요합니다.
HF 변환 파이프라인에서는 일반적으로 `--normalize` 옵션 또는 학습 시 내부 전처리를 활용하세요.

```bash
# 레거시 dump 후 후처리 예시
python scripts/preprocess_data_dump.py \
  --data /path/to/data.npy \
  --interpolate-nans \
  --remove-baseline-wander
```

---

## 3) Pretrain 실행 (현재 방식)

### 3.1 설정 파일 준비

`configs/pretrain/**/*.yaml`에서 아래를 먼저 수정:

- `run.amp`, `run.compile`, `run.checkpoint`
- `datasets.<name>.path` (HF dataset 디렉토리 경로)
- 필요 시 `split`, `weight`

예: `configs/pretrain/ViT/ViTS_mimic.yaml`

### 3.2 단일 GPU

```bash
python pretrain.py --config configs/pretrain/ViT/ViTS_mimic.yaml
```

### 3.3 멀티 GPU

```bash
# 스크립트 기본값: NUM_GPUS=8
bash scripts/pretrain_multi_gpu.sh configs/pretrain/ViT/ViTS_mimic.yaml
```

또는 직접:

```bash
torchrun --standalone --nproc_per_node=8 pretrain.py --config configs/pretrain/ViT/ViTS_mimic.yaml
```

---

## 4) Finetune / Linear evaluation 실행

### 4.1 설정 파일 준비

`configs/eval/*.yaml`에서 다음을 확인:

- `run.encoder` (pretrain checkpoint 또는 encoder config)
- `run.dataset_type` (`ecg` 또는 `har`)
- `run.task` (ECG면 `all/diagnostic/.../ST-MEM`, HAR면 `har`)
- `dataset.data_dir` (또는 CLI `--data-dir`로 override)

### 4.2 실행 예시 (ECG)

```bash
python finetune.py \
  --config configs/eval/linear.yaml \
  --data-dir /path/to/hf/ptb-xl \
  --encoder /path/to/pretrain/checkpoint.pt
```

```bash
python finetune.py \
  --config configs/eval/finetune.yaml \
  --data-dir /path/to/hf/ptb-xl \
  --encoder /path/to/pretrain/checkpoint.pt
```

### 4.3 실행 예시 (HAR / Capture-24)

```bash
python finetune.py \
  --config configs/eval/har_linear.yaml \
  --data-dir /path/to/hf/capture24 \
  --encoder /path/to/pretrain/checkpoint.pt
```

### 4.4 멀티 GPU

```bash
bash scripts/finetune_multi_gpu.sh configs/eval/finetune.yaml
```

---

## 5) Ablation YAML 자동 생성

아래 스크립트로 `max_keep_ratio`, `min_block_size`, `patch_size` 조합 YAML을 일괄 생성할 수 있습니다.

```bash
python scripts/generate_ablation_yamls.py \
  --template configs/pretrain/ViT/ViTS_mimic.yaml \
  --output-dir configs/pretrain/ViT/ablation \
  --max-keep-ratios 0.20 0.25 0.30 \
  --min-block-sizes 8 10 12 \
  --patch-sizes 20 25
```

생성 파일명 형식:

- `m{max_keep_ratio}_b{min_block_size}_p{patch_size}.yaml`

예:

- `m0.25_b10_p25.yaml`

---

## 6) 실무 팁

- config 이름 문자열(`ViTS_mimic`) 대신, **명시적 경로**(`configs/pretrain/ViT/ViTS_mimic.yaml`)를 쓰면 혼동이 적습니다.
- 대규모 pretrain에서는 먼저 소규모 subset/HF shard로 smoke test 후 본 실행을 권장합니다.
- `run.checkpoint` 또는 `--chkpt`로 재시작 시, 출력 디렉토리를 명확히 분리해 실험 추적성을 유지하세요.
- PTB-XL eval은 반드시 `convert_to_hf_dataset.py`로 만든 `train/val/test` split을 사용하세요.

---

## 7) 자주 생기는 문제

- `ModuleNotFoundError: datasets`
  - `pip install datasets`
- `Config file not found`
  - `--config`에 실제 파일 경로를 전달했는지 확인
- `Dataset does not exist`
  - pretrain config의 `datasets.*.path`가 HF dataset 디렉토리인지 확인
- finetune에서 split 오류
  - ECG 데이터는 `train/val/test` split이 있어야 함 (변환 스크립트로 재생성)
