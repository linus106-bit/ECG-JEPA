"""Convert Capture-24 HuggingFace dataset to .npy files for ECG-JEPA pipeline.

Usage:
  python scripts/prepare_capture24.py --data-dir /path/to/capture24 --out /path/to/output

The script expects a HuggingFace dataset directory with train/test splits.
Each sample has shape (3, 1000) at 100Hz and an integer label (0-9).

Output files:
  capture24_train.npy         (N_train, 1000, 3) - channels last
  capture24_train_labels.npy  (N_train,) int
  capture24_val.npy           (N_val, 1000, 3)
  capture24_val_labels.npy    (N_val,) int
  capture24_test.npy          (N_test, 1000, 3)
  capture24_test_labels.npy   (N_test,) int
"""

import argparse
from os import path, makedirs

import numpy as np
from datasets import load_from_disk

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', required=True, help='path to HuggingFace dataset directory')
parser.add_argument('--out', default='.', help='output directory')
parser.add_argument('--val-ratio', type=float, default=0.2,
                    help='fraction of training data to use for validation')
parser.add_argument('--seed', type=int, default=42, help='random seed for validation split')
args = parser.parse_args()

makedirs(args.out, exist_ok=True)

print(f'Loading dataset from {args.data_dir}')
dataset = load_from_disk(args.data_dir)

# extract train split
train_ds = dataset['train']
train_data = np.array(train_ds['input_values'], dtype=np.float32)  # (N, 3, 1000)
train_labels = np.array(train_ds['label'], dtype=np.int64)

# extract test split
test_ds = dataset['test']
test_data = np.array(test_ds['input_values'], dtype=np.float32)  # (N, 3, 1000)
test_labels = np.array(test_ds['label'], dtype=np.int64)

# transpose to channels-last: (N, 3, 1000) -> (N, 1000, 3)
train_data = train_data.transpose(0, 2, 1)
test_data = test_data.transpose(0, 2, 1)

# create validation split from training data
rng = np.random.RandomState(args.seed)
num_train = len(train_data)
num_val = int(num_train * args.val_ratio)
indices = rng.permutation(num_train)
val_indices = indices[:num_val]
train_indices = indices[num_val:]

val_data = train_data[val_indices]
val_labels = train_labels[val_indices]
train_data = train_data[train_indices]
train_labels = train_labels[train_indices]

# compute per-channel mean and std from training data
# shape: (N, 1000, 3) -> mean/std over (N, 1000)
mean = train_data.mean(axis=(0, 1))
std = train_data.std(axis=(0, 1))
print(f'Train mean (per channel): {mean.tolist()}')
print(f'Train std  (per channel): {std.tolist()}')
print(f'Update data/datasets/capture24.py with these values.')

# save
print(f'Train: {train_data.shape}, Val: {val_data.shape}, Test: {test_data.shape}')

np.save(path.join(args.out, 'capture24_train.npy'), train_data.astype(np.float16))
np.save(path.join(args.out, 'capture24_train_labels.npy'), train_labels)
np.save(path.join(args.out, 'capture24_val.npy'), val_data.astype(np.float16))
np.save(path.join(args.out, 'capture24_val_labels.npy'), val_labels)
np.save(path.join(args.out, 'capture24_test.npy'), test_data.astype(np.float16))
np.save(path.join(args.out, 'capture24_test_labels.npy'), test_labels)

print('Done.')
