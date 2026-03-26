"""Convert raw datasets to HuggingFace Dataset format.

Each dataset is saved with columns: ['id', 'data', 'label']
  - id: unique identifier (str or int)
  - data: 2D array (channel_size, num_channels) channels last, float16
  - label: int (single-label) or list[int] (multi-hot multi-label), or -1 if no label

Splits:
  - PTB-XL: train (folds 1-8), val (fold 9), test (fold 10)
  - Capture-24: train, test (from original HF dataset)
  - Other ECG datasets: train only (pretrain-only datasets, no labels)

Usage:
  python scripts/convert_to_hf_dataset.py --data-dir /path/to/ptb-xl --dataset ptb-xl --out /path/to/output
  python scripts/convert_to_hf_dataset.py --data-dir /path/to/capture24 --dataset capture-24 --out /path/to/output
  python scripts/convert_to_hf_dataset.py --data-dir /path/to/mimic-iv-ecg --dataset mimic-iv-ecg --out /path/to/output
"""

import argparse
from os import path

import numpy as np
from datasets import Dataset, DatasetDict

from data.datasets import DATASETS
from data.datasets.capture24 import Capture24
from data.datasets.code_15 import CODE15
from data.datasets.ptb_xl import PTB_XL
from data.utils import load_raw_data, load_raw_variable_data

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', required=True, help='path to raw data directory')
parser.add_argument('--dataset', choices=list(DATASETS), required=True, help='dataset type')
parser.add_argument('--out', required=True, help='output directory for HF dataset')
parser.add_argument('--verbose', action='store_true', help='verbose mode')
args = parser.parse_args()


def convert_ptb_xl(data_dir, out_dir, verbose=False):
  """Convert PTB-XL to HF dataset with train/val/test splits based on strat_fold."""
  import pandas as pd

  record_names = PTB_XL.find_records(data_dir)
  data = load_raw_data(record_names, verbose=verbose)

  labels_df = pd.read_csv(path.join(data_dir, 'ptbxl_database.csv'), index_col='ecg_id')
  scp_codes_str = labels_df.scp_codes.values.tolist()
  strat_folds = labels_df.strat_fold.values.tolist()
  ecg_ids = labels_df.index.tolist()

  splits = {}
  for split_name, fold_values in [('train', range(1, 9)), ('val', [9]), ('test', [10])]:
    fold_set = set(fold_values)
    indices = [i for i, fold in enumerate(strat_folds) if fold in fold_set]
    split_data = {
      'id': [str(ecg_ids[i]) for i in indices],
      'data': [data[i].tolist() for i in indices],
      'label': [scp_codes_str[i] for i in indices],
      'strat_fold': [strat_folds[i] for i in indices],
    }
    splits[split_name] = Dataset.from_dict(split_data)
    print(f'  {split_name}: {len(indices)} samples')

  dataset_dict = DatasetDict(splits)
  dataset_dict.save_to_disk(out_dir)

  # Copy scp_statements.csv needed for label aggregation during finetuning
  import shutil
  scp_file = path.join(data_dir, 'scp_statements.csv')
  if path.isfile(scp_file):
    shutil.copy2(scp_file, path.join(out_dir, 'scp_statements.csv'))
    print(f'Copied scp_statements.csv to {out_dir}')

  print(f'Saved PTB-XL HF dataset to {out_dir}')


def convert_capture24(data_dir, out_dir, verbose=False):
  """Convert Capture-24 to HF dataset with train/test splits."""
  all_data, labels, split_names = Capture24.load_data(data_dir)

  splits = {}
  for split_name in ('train', 'test'):
    mask = split_names == split_name
    indices = np.where(mask)[0]
    split_data = {
      'id': [str(i) for i in indices],
      'data': [all_data[i].tolist() for i in indices],
      'label': [int(labels[i]) for i in indices],
    }
    splits[split_name] = Dataset.from_dict(split_data)
    print(f'  {split_name}: {len(indices)} samples')

  dataset_dict = DatasetDict(splits)
  dataset_dict.save_to_disk(out_dir)
  print(f'Saved Capture-24 HF dataset to {out_dir}')


def convert_fixed_length(dataset_name, data_dir, out_dir, verbose=False):
  """Convert fixed-length ECG datasets (pretrain only, no labels)."""
  dataset_cls = DATASETS[dataset_name]

  if dataset_name == 'code-15':
    dataset_obj = CODE15(data_dir)
    data = dataset_obj.load_raw_data(skip_variable=True, verbose=verbose)
  else:
    record_names = dataset_cls.find_records(data_dir)
    min_channel_size = None
    if hasattr(dataset_cls, 'record_duration'):
      min_channel_size = dataset_cls.record_duration * dataset_cls.sampling_frequency
    data = load_raw_data(record_names, min_channel_size=min_channel_size, verbose=verbose)

  split_data = {
    'id': [str(i) for i in range(len(data))],
    'data': [data[i].tolist() for i in range(len(data))],
    'label': [-1] * len(data),
  }
  dataset_dict = DatasetDict({'train': Dataset.from_dict(split_data)})
  dataset_dict.save_to_disk(out_dir)
  print(f'Saved {dataset_name} HF dataset ({len(data)} samples) to {out_dir}')


def convert_variable_length(dataset_name, data_dir, out_dir, verbose=False):
  """Convert variable-length ECG datasets (pretrain only, no labels)."""
  dataset_cls = DATASETS[dataset_name]
  record_names = dataset_cls.find_records(data_dir)
  concat_data, sizes = load_raw_variable_data(record_names, verbose=verbose)

  starts = np.concatenate([[0], np.cumsum(sizes[:-1])])
  split_data = {
    'id': [str(i) for i in range(len(sizes))],
    'data': [concat_data[start:start + size].tolist()
             for start, size in zip(starts, sizes)],
    'label': [-1] * len(sizes),
  }
  dataset_dict = DatasetDict({'train': Dataset.from_dict(split_data)})
  dataset_dict.save_to_disk(out_dir)
  print(f'Saved {dataset_name} HF dataset ({len(sizes)} variable-length samples) to {out_dir}')


# Dataset type classification
FIXED_LENGTH_DATASETS = {
  'chapman-shaoxing', 'georgia', 'ningbo', 'st-petersburg',
  'code-15', 'mimic-iv-ecg',
}
VARIABLE_LENGTH_DATASETS = {'cpsc', 'cpsc-extra', 'ptb'}

print(f'Converting {args.dataset} from {args.data_dir}')

if args.dataset == 'ptb-xl':
  convert_ptb_xl(args.data_dir, args.out, verbose=args.verbose)
elif args.dataset == 'capture-24':
  convert_capture24(args.data_dir, args.out, verbose=args.verbose)
elif args.dataset in FIXED_LENGTH_DATASETS:
  convert_fixed_length(args.dataset, args.data_dir, args.out, verbose=args.verbose)
elif args.dataset in VARIABLE_LENGTH_DATASETS:
  convert_variable_length(args.dataset, args.data_dir, args.out, verbose=args.verbose)
else:
  raise ValueError(f'Unknown dataset type: {args.dataset}')
