# ECG-JEPA

Official implementation of the paper ["Self-Supervised Pre-Training with Joint-Embedding Predictive Architecture Boosts ECG Classification Performance"](https://arxiv.org/abs/2410.13867).

The current codebase supports:

- JEPA pre-training for ECG signals with ViT, CNN, and Mamba encoders
- Fine-tuning / linear evaluation for ECG classification on PTB-XL
- Single-label evaluation for HAR (Capture-24) and PPG-style data (SDB)
- Hugging Face dataset directories as the primary data format
- Legacy `.npy` / `.npz` dumps for backward compatibility in pre-training

## Results

We pre-train models with the [JEPA](https://arxiv.org/abs/2301.08243) framework and evaluate them on PTB-XL. The table below reports average test AUC over 10 runs, with standard deviation in brackets.

| Model         | Method                    | All \[Fine-tune\] | All \[Linear\] | Source                                                                          |
|---------------|---------------------------|-------------------|----------------|---------------------------------------------------------------------------------|
| inception1d   | Random Init               | 0.925(08)         | —              | [helme/ecg_ptbxl_benchmarking](https://github.com/helme/ecg_ptbxl_benchmarking) |
| xresnet1d50   | Random Init               | 0.924(05)         | 0.721(16)      | [hhi-aml/ecg-selfsupervised](https://github.com/hhi-aml/ecg-selfsupervised)     |
| 4FC+2LSTM+2FC | Random Init               | 0.932(03)         | 0.711(07)      | [hhi-aml/ecg-selfsupervised](https://github.com/hhi-aml/ecg-selfsupervised)     |
| ViT-B         | Random Init               | 0.837(17)         | 0.867(05)      | This repository                                                                 |
| ViT-S         | Random Init               | 0.883(04)         | 0.833(06)      | This repository                                                                 |
| ViT-XS        | Random Init               | 0.911(04)         | 0.815(10)      | This repository                                                                 |
| xresnet1d50   | SimCLR                    | 0.927(03)         | 0.883(03)      | [hhi-aml/ecg-selfsupervised](https://github.com/hhi-aml/ecg-selfsupervised)     |
| xresnet1d50   | BYOL                      | 0.929(02)         | 0.878(02)      | [hhi-aml/ecg-selfsupervised](https://github.com/hhi-aml/ecg-selfsupervised)     |
| 4FC+2LSTM+2FC | CPC (CinC2020)            | 0.942(01)         | 0.927(01)      | [hhi-aml/ecg-selfsupervised](https://github.com/hhi-aml/ecg-selfsupervised)     |
| 4FC+2LSTM+2FC | CPC (CinC2020 w/o PTB-XL) | 0.940(02)         | 0.919(01)      | [hhi-aml/ecg-selfsupervised](https://github.com/hhi-aml/ecg-selfsupervised)     |
| S4            | CPC (CinC2021)            | **0.945(02)**     | -              | [tmehari/ssm_ecg](https://github.com/tmehari/ssm_ecg)                           |
| ViT-B         | JEPA (All)                | 0.940(01)         | 0.935(01)      | This repository                                                                 |
| ViT-S         | JEPA (All)                | **0.945(01)**     | 0.938(02)      | This repository                                                                 |
| ViT-S         | JEPA (MIMIC-IV-ECG)       | 0.944(01)         | **0.940(02)**  | This repository                                                                 |
| ViT-S         | JEPA (PTB-XL)             | 0.930(01)         | 0.926(02)      | This repository                                                                 |
| ViT-XS        | JEPA (All)                | 0.939(00)         | 0.933(02)      | This repository                                                                 |
| ViT-XS        | JEPA (MIMIC-IV-ECG)       | 0.943(01)         | 0.933(03)      | This repository                                                                 |
| ViT-XS        | JEPA (PTB-XL)             | 0.940(01)         | 0.931(02)      | This repository                                                                 |

The next table compares against [ST-MEM](https://arxiv.org/abs/2402.09450) on the PTB-XL superdiagnostic single-label setup.

| Model         | Method                 | Superdiagnostic<br>(Single Label) \[Fine-tune\] | Superdiagnostic<br>(Single Label) \[Linear\] | Source                           |
|---------------|------------------------|-------------------------------------------------|----------------------------------------------|----------------------------------|
| ViT-B         | MoCo v3                | 0.913(02)                                       | 0.739(06)                                    | https://arxiv.org/abs/2402.09450 |
| ViT-B         | CMSC                   | 0.877(03)                                       | 0.797(38)                                    | https://arxiv.org/abs/2402.09450 |
| ViT-B         | MTAE                   | 0.910(01)                                       | 0.807(06)                                    | https://arxiv.org/abs/2402.09450 |
| ViT-B         | MTAE+RLM               | 0.911(04)                                       | 0.806(05)                                    | https://arxiv.org/abs/2402.09450 |
| ViT-B         | MLAE                   | 0.915(01)                                       | 0.779(08)                                    | https://arxiv.org/abs/2402.09450 |
| ViT-B         | ST-MEM                 | 0.933(03)                                       | 0.838(11)                                    | https://arxiv.org/abs/2402.09450 |
| 4FC+2LSTM+2FC | CPC (w/ entire PTB-XL) | 0.934(02)                                       | —                                            | https://arxiv.org/abs/2402.09450 |
| ViT-B         | JEPA (All)             | 0.928(03)                                       | 0.920(02)                                    | This repository                  |
| ViT-S         | JEPA (All)             | **0.935(02)**                                   | **0.928(03)**                                | This repository                  |
| ViT-S         | JEPA (MIMIC-IV-ECG)    | 0.932(02)                                       | 0.921(03)                                    | This repository                  |
| ViT-S         | JEPA (PTB-XL)          | 0.929(02)                                       | 0.917(01)                                    | This repository                  |
| ViT-XS        | JEPA (All)             | 0.930(01)                                       | 0.924(02)                                    | This repository                  |
| ViT-XS        | JEPA (MIMIC-IV-ECG)    | 0.928(02)                                       | 0.920(02)                                    | This repository                  |
| ViT-XS        | JEPA (PTB-XL)          | 0.928(02)                                       | 0.919(02)                                    | This repository                  |

## Installation

```bash
mamba create -n ecg-jepa python=3.13
mamba activate ecg-jepa
pip install -r requirements.txt
pip install datasets
```

Notes:

- The repository uses the Hugging Face `datasets` package at runtime, but it is not listed in `requirements.txt`.

## Data Format

The current implementation expects Hugging Face dataset directories for normal use.

- Pre-training input: HF dataset directory or legacy `.npy` / `.npz`
- ECG fine-tuning input: HF dataset directory with `train` / `val` / `test`
- HAR / PPG fine-tuning input: HF dataset directory with at least `train` / `test`

Expected sample layout:

- `data`: channels-first array shaped `(num_channels, channel_size)`
- `label`: multi-hot list for ECG, integer class for HAR / PPG, or `-1` for unlabeled pre-train data

## Supported Dataset IDs

Dataset identifiers are defined in `data/datasets/__init__.py`:

- `capture-24`
- `chapman-shaoxing`
- `cpsc`
- `cpsc-extra`
- `georgia`
- `ningbo`
- `ptb`
- `st-petersburg`
- `code-15`
- `mimic-iv-ecg`
- `ptb-xl`
- `sdb`

`scripts/convert_to_hf_dataset.py` currently converts these raw datasets to HF format:

- `ptb-xl`
- `capture-24`
- `chapman-shaoxing`
- `cpsc`
- `cpsc-extra`
- `georgia`
- `ningbo`
- `ptb`
- `st-petersburg`
- `code-15`
- `mimic-iv-ecg`

`sdb` is expected as a user-provided HF dataset and is not converted by the script above.

## Download And Convert Data

### 1. Download raw ECG datasets

```bash
bash scripts/download_datasets.sh /path/to/raw-data
```

This script downloads:

- PTB-XL
- MIMIC-IV-ECG
- PhysioNet Challenge 2021 subsets
- CODE-15

Capture-24 and SDB are not downloaded by this helper script. Provide them separately if you use HAR / PPG evaluation.

### 2. Convert raw data to HF dataset directories

PTB-XL for ECG fine-tuning:

```bash
python -m scripts.convert_to_hf_dataset \
  --data-dir /path/to/raw-data/ptb-xl \
  --dataset ptb-xl \
  --task all \
  --out /path/to/hf/ptb-xl \
  --verbose
```

MIMIC-IV-ECG for pre-training:

```bash
python -m scripts.convert_to_hf_dataset \
  --data-dir /path/to/raw-data/mimic-iv-ecg \
  --dataset mimic-iv-ecg \
  --out /path/to/hf/mimic-iv-ecg \
  --verbose
```

Capture-24 for HAR evaluation:

```bash
python -m scripts.convert_to_hf_dataset \
  --data-dir /path/to/raw-data/capture24 \
  --dataset capture-24 \
  --out /path/to/hf/capture24 \
  --verbose
```

Optional normalization during conversion:

```bash
python -m scripts.convert_to_hf_dataset \
  --data-dir /path/to/raw-data/mimic-iv-ecg \
  --dataset mimic-iv-ecg \
  --out /path/to/hf/mimic-iv-ecg-normalized \
  --normalize \
  --verbose
```

## Pre-Training

Pre-train configs live in:

- `configs/pretrain/ViT`
- `configs/pretrain/CNN`
- `configs/pretrain/Mamba`

Important details:

- Pass `--config` as an actual YAML file path
- Set `datasets.<name>.path` inside the YAML to your dataset directory
- Dataset sampling `weight` values must sum to `1`
- `run.checkpoint` or `--chkpt` resumes training
- Checkpoints are saved as `chkpt_<step>.pt`

Example: single-dataset ViT pre-training on MIMIC-IV-ECG

```bash
python pretrain.py \
  --config configs/pretrain/ViT/ViTS_mimic.yaml \
  --out results/pretrain/ViTS_mimic \
  --amp bfloat16
```

Example: multi-GPU pre-training

```bash
torchrun --standalone --nproc_per_node=8 pretrain.py \
  --config configs/pretrain/ViT/ViTS_mimic.yaml
```

The repository also includes a helper wrapper:

```bash
bash scripts/pretrain_multi_gpu.sh configs/pretrain/ViT/ViTS_mimic.yaml
```

If you want a smaller encoder-only checkpoint for downstream use:

```bash
python scripts/minify_pretrained_checkpoint.py \
  --checkpoint results/pretrain/ViTS_mimic/chkpt_10000.pt
```

## Fine-Tuning And Evaluation

Evaluation configs live in `configs/eval`.

Key behavior of `finetune.py`:

- `--encoder` accepts either a pre-train checkpoint or a pre-train config YAML
- If `--encoder` is a YAML file, the encoder is initialized from scratch
- `--dataset-type` can be `ecg`, `har`, or `ppg`; if omitted, it is inferred from `--data-dir`
- ECG tasks use ROC-AUC
- HAR / PPG tasks use macro F1 and accuracy
- Best checkpoint is saved as `<task>_best_chkpt.pt`
- Metrics are saved as `<task>_eval_results.json`
- Predictions are saved as `<task>_predictions.npz`

### Linear evaluation on PTB-XL

```bash
python finetune.py \
  --config configs/eval/linear.yaml \
  --data-dir /path/to/hf/ptb-xl \
  --encoder /path/to/pretrain/chkpt_10000.pt \
  --task all
```

### End-to-end fine-tuning from a linear checkpoint

```bash
python finetune.py \
  --config configs/eval/finetune_after_linear.yaml \
  --data-dir /path/to/hf/ptb-xl \
  --encoder /path/to/linear-output/all_best_chkpt.pt \
  --task all
```

### Direct fine-tuning on PTB-XL

```bash
python finetune.py \
  --config configs/eval/finetune.yaml \
  --data-dir /path/to/hf/ptb-xl \
  --encoder /path/to/pretrain/chkpt_10000.pt \
  --task all
```

### ST-MEM-style single-label evaluation on PTB-XL

```bash
python finetune.py \
  --config configs/eval/finetune.yaml \
  --data-dir /path/to/hf/ptb-xl \
  --encoder /path/to/pretrain/chkpt_10000.pt \
  --task ST-MEM
```

### HAR evaluation on Capture-24

```bash
python finetune.py \
  --config configs/eval/har_linear.yaml \
  --data-dir /path/to/hf/capture24 \
  --encoder /path/to/pretrain/chkpt_10000.pt \
  --dataset-type har
```

### Multi-GPU evaluation

```bash
torchrun --standalone --nproc_per_node=8 finetune.py \
  --config configs/eval/finetune.yaml \
  --data-dir /path/to/hf/ptb-xl \
  --encoder /path/to/pretrain/chkpt_10000.pt
```

The repository also includes a helper wrapper:

```bash
bash scripts/finetune_multi_gpu.sh configs/eval/finetune.yaml
```

## Legacy Compatibility

Some legacy paths are still supported:

- Pre-training from `.npy` or `.npz` dumps
- Capture-24 evaluation from legacy dump prefixes
- `scripts/preprocess_data_dump.py` for old dump-based preprocessing

For new experiments, prefer HF dataset directories.

## Common Issues

- `Config file not found`
  - Use a full YAML path such as `configs/pretrain/ViT/ViTS_mimic.yaml`
- `ModuleNotFoundError: datasets`
  - Install the missing package with `pip install datasets`
- `Dataset does not exist`
  - Check `datasets.<name>.path` inside the pre-train YAML
- ECG fine-tuning split errors
  - Recreate PTB-XL with `scripts/convert_to_hf_dataset.py` so `train/val/test` exist

## Archived Repository

@ZIB: https://git.zib.de/bzfweima/ecg-jepa
