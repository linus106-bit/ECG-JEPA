"""Convert raw datasets to HuggingFace Dataset format.

Each dataset is saved with columns: ['id', 'data', 'label']
  - id: unique identifier (str or int)
  - data: 2D array (num_channels, channel_size) channels first, float16
  - label: multi-hot list[int] (PTB-XL), ClassLabel int (Capture-24), or -1 (no label)

PTB-XL also includes:
  - label_names: list[str] of class names corresponding to each multi-hot position
  - strat_fold: int fold number for splitting

Capture-24 label column uses HuggingFace ClassLabel feature, so
  dataset.features['label'].names provides class names and num_classes.

Splits:
  - PTB-XL: train (folds 1-8), val (fold 9), test (fold 10)
  - Capture-24: train, test (from original HF dataset)
  - Other ECG datasets: train only (pretrain-only datasets, no labels)

Usage (run from project root):
  python -m scripts.convert_to_hf_dataset --data-dir /path/to/ptb-xl --dataset ptb-xl --out /path/to/output
  python -m scripts.convert_to_hf_dataset --data-dir /path/to/ptb-xl --dataset ptb-xl --task superdiagnostic --out /path/to/output
  python -m scripts.convert_to_hf_dataset --data-dir /path/to/capture24 --dataset capture-24 --out /path/to/output
  python -m scripts.convert_to_hf_dataset --data-dir /path/to/mimic-iv-ecg --dataset mimic-iv-ecg --out /path/to/output
"""

import argparse
from os import path

import numpy as np
from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value

from data.datasets import DATASETS
from data.datasets.capture24 import Capture24
from data.datasets.code_15 import CODE15
from data.datasets.ptb_xl import PTB_XL
from data.utils import load_raw_data, load_raw_variable_data

TASKS = (
  'all',
  'diagnostic',
  'subdiagnostic',
  'superdiagnostic',
  'form',
  'rhythm',
)

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', required=True, help='path to raw data directory')
parser.add_argument('--dataset', choices=list(DATASETS), required=True, help='dataset type')
parser.add_argument('--out', required=True, help='output directory for HF dataset')
parser.add_argument('--task', choices=TASKS, default='all',
                    help='label task for PTB-XL (default: all). Ignored for non-PTB-XL datasets.')
parser.add_argument('--verbose', action='store_true', help='verbose mode')
args = parser.parse_args()


def convert_ptb_xl(data_dir, out_dir, task='all', verbose=False):
  """Convert PTB-XL to HF dataset with train/val/test splits and multi-hot labels."""
  import pandas as pd

  record_names = PTB_XL.find_records(data_dir)
  data = load_raw_data(record_names, verbose=verbose)

  # Load raw labels and compute aggregations for the specified task
  labels_df = PTB_XL.load_raw_labels(data_dir)
  labels_df = PTB_XL.compute_label_aggregations(labels_df, data_dir, task)

  # Get multi-hot labels and class names via select_data
  all_x, labels_df, y_multihot, mlb = PTB_XL.select_data(data, labels_df, task, min_samples=0)

  label_names = list(mlb.classes_)
  strat_folds = labels_df.strat_fold.values.tolist()
  ecg_ids = labels_df.index.tolist()

  print(f'  task={task}, num_classes={len(label_names)}, label_names={label_names}')
  print(f'  total samples after filtering: {len(all_x)}')

  splits = {}
  for split_name, fold_values in [('train', range(1, 9)), ('val', [9]), ('test', [10])]:
    fold_set = set(fold_values)
    indices = [i for i, fold in enumerate(strat_folds) if fold in fold_set]
    split_data = {
      'id': [str(ecg_ids[i]) for i in indices],
      'data': [all_x[i].T.tolist() for i in indices],  # (channel_size, num_channels) -> (num_channels, channel_size)
      'label': [y_multihot[i].tolist() for i in indices],
      'label_names': [label_names for _ in indices],
      'strat_fold': [strat_folds[i] for i in indices],
    }
    splits[split_name] = Dataset.from_dict(split_data)
    print(f'  {split_name}: {len(indices)} samples')

  dataset_dict = DatasetDict(splits)
  dataset_dict.save_to_disk(out_dir)

  # Copy scp_statements.csv needed for label aggregation if task changes later
  import shutil
  scp_file = path.join(data_dir, 'scp_statements.csv')
  if path.isfile(scp_file):
    shutil.copy2(scp_file, path.join(out_dir, 'scp_statements.csv'))
    print(f'Copied scp_statements.csv to {out_dir}')

  print(f'Saved PTB-XL HF dataset to {out_dir}')


def convert_capture24(data_dir, out_dir, verbose=False):
  """Convert Capture-24 to HF dataset with train/test splits and ClassLabel."""
  from datasets import load_dataset

  # Load original HF dataset to get label names
  original_ds = load_dataset(data_dir)
  original_label_feature = original_ds['train'].features.get('label')
  if hasattr(original_label_feature, 'names'):
    label_names = original_label_feature.names
  else:
    # Fallback: infer from data
    all_labels = set(original_ds['train']['label']) | set(original_ds['test']['label'])
    label_names = [str(i) for i in range(max(all_labels) + 1)]

  print(f'  num_classes={len(label_names)}, label_names={label_names}')

  class_label = ClassLabel(names=label_names)
  features = Features({
    'id': Value('string'),
    'data': Sequence(Sequence(Value('float32'))),
    'label': class_label,
  })

  # Load data using Capture24 loader (returns channels-last)
  all_data, labels, split_names = Capture24.load_data(data_dir)

  splits = {}
  for split_name in ('train', 'test'):
    mask = split_names == split_name
    indices = np.where(mask)[0]
    split_data = {
      'id': [str(i) for i in indices],
      'data': [all_data[i].T.tolist() for i in indices],  # (channel_size, num_channels) -> (num_channels, channel_size)
      'label': [int(labels[i]) for i in indices],
    }
    splits[split_name] = Dataset.from_dict(split_data, features=features)
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
    'data': [data[i].T.tolist() for i in range(len(data))],  # transpose to channels-first
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
    'data': [concat_data[start:start + size].T.tolist()  # transpose to channels-first
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
  convert_ptb_xl(args.data_dir, args.out, task=args.task, verbose=args.verbose)
elif args.dataset == 'capture-24':
  convert_capture24(args.data_dir, args.out, verbose=args.verbose)
elif args.dataset in FIXED_LENGTH_DATASETS:
  convert_fixed_length(args.dataset, args.data_dir, args.out, verbose=args.verbose)
elif args.dataset in VARIABLE_LENGTH_DATASETS:
  convert_variable_length(args.dataset, args.data_dir, args.out, verbose=args.verbose)
else:
  raise ValueError(f'Unknown dataset type: {args.dataset}')
