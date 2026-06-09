# ECG-JEPA

Official implementation of the paper ["Self-Supervised Pre-Training with Joint-Embedding Predictive Architecture Boosts ECG Classification Performance"](https://arxiv.org/abs/2410.13867).

The current codebase supports:

- JEPA pre-training for ECG signals with ViT, CNN, and Mamba encoders
- Fine-tuning / linear evaluation for ECG classification on PTB-XL
- Single-label evaluation for HAR (Capture-24) and PPG-style data (SDB)
- Hugging Face dataset directories as the primary data format
- Legacy `.npy` / `.npz` dumps for backward compatibility in pre-training

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

The repository also includes a helper wrapper: (match NUM_GPUS of scripts/pretrain_multi_gpu.sh)

```bash
bash scripts/pretrain_multi_gpu.sh configs/pretrain/ViT/ViTS_mimic.yaml
```

If you want a smaller encoder-only checkpoint for downstream use:

```bash
python scripts/minify_pretrained_checkpoint.py \
  --checkpoint results/pretrain/ViTS_mimic/chkpt_10000.pt
```

## Standalone Encoder Inference And ONNX Export

Run PyTorch encoder inference directly from a full pre-training checkpoint, a
minified checkpoint, or a fine-tuned checkpoint. If the checkpoint is a
fine-tuned classifier checkpoint, the script also prints classifier
probabilities for six labels by default: `AFIB`, `1AVB`, `2AVB`, `SVTAC`,
`PAC`, and `PVC`; pass `--label-names` to override them. `--normalize auto`
uses the fine-tuned checkpoint's saved train-set mean/std preprocessing when
available.

```bash
python scripts/infer_ecg_encoder.py \
  --checkpoint results/pretrain/ViTS_mimic/chkpt_10000.pt \
  --input /path/to/ecg.npy \
  --output /path/to/embeddings.npz \
  --length-mode center-crop \
  --normalize auto
```

Export the trained 500 Hz model to ONNX. Install optional ONNX dependencies
first with `pip install onnx onnxruntime`. For fine-tuned classifier checkpoints,
`--export-target auto` exports the full classifier by default, with input name
`ecg` and outputs `logits` and `probabilities`. For pre-training checkpoints,
the exporter emits encoder `token_embeddings`. A sidecar metadata JSON is
written next to the ONNX file by default, including saved preprocessing stats
and eval crop settings. For fine-tuned ECG configs this matches eval by running
2.5-second crops with 1.25-second stride, averaging crop logits, then applying
sigmoid.

```bash
python scripts/export_ecg_encoder_onnx.py \
  --checkpoint /path/to/fine-tuned/all_best_chkpt.pt \
  --output /path/to/ecg_classifier.onnx
```

Run ONNX Runtime inference from the exported ONNX model:

```bash
python scripts/infer_ecg_encoder_onnx.py \
  --onnx /path/to/ecg_classifier.onnx \
  --input /path/to/ecg.npy \
  --output /path/to/onnx_predictions.npz \
  --length-mode center-crop \
  --normalize auto
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
