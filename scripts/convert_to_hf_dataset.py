"""Convert raw datasets to HuggingFace Dataset format (Parquet).

Each dataset is saved with columns: ['id', 'data', 'label']
  - id: unique identifier (str or int)
  - data: 2D array (num_channels, channel_size) channels first, float16
  - label: multi-hot list[int] (PTB-XL), ClassLabel int (Capture-24), or -1 (no label)

PTB-XL also includes:
  - label_names: list[str] of class names corresponding to each multi-hot position
  - strat_fold: int fold number for splitting

Capture-24 label column uses HuggingFace ClassLabel feature, so
  dataset.features['label'].names provides class names and num_classes.

Output format: Parquet files saved as {out_dir}/{split}.parquet
  Load with: datasets.load_dataset(out_dir)

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
import os
from os import path

import numpy as np
import wfdb
from datasets import ClassLabel, Dataset, Features, Sequence, Value
from tqdm import tqdm

from data.datasets import DATASETS
from data.datasets.code_15 import CODE15
from data.datasets.ptb_xl import PTB_XL

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


def _save_parquet(dataset_or_splits, out_dir):
  """Save dataset splits as parquet files."""
  os.makedirs(out_dir, exist_ok=True)
  if isinstance(dataset_or_splits, dict):
    for split_name, ds in dataset_or_splits.items():
      out_path = path.join(out_dir, f'{split_name}.parquet')
      ds.to_parquet(out_path)
      print(f'  saved {split_name}.parquet ({len(ds)} samples)')
  else:
    out_path = path.join(out_dir, 'train.parquet')
    dataset_or_splits.to_parquet(out_path)
    print(f'  saved train.parquet ({len(dataset_or_splits)} samples)')


def convert_ptb_xl(data_dir, out_dir, task='all', verbose=False):
  """Convert PTB-XL to parquet with train/val/test splits and multi-hot labels."""
  from data.utils import load_raw_data

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
      'data': [all_x[i].T.tolist() for i in indices],
      'label': [y_multihot[i].tolist() for i in indices],
      'label_names': [label_names for _ in indices],
      'strat_fold': [strat_folds[i] for i in indices],
    }
    splits[split_name] = Dataset.from_dict(split_data)
    print(f'  {split_name}: {len(indices)} samples')

  _save_parquet(splits, out_dir)

  # Copy scp_statements.csv needed for label aggregation if task changes later
  import shutil
  scp_file = path.join(data_dir, 'scp_statements.csv')
  if path.isfile(scp_file):
    shutil.copy2(scp_file, path.join(out_dir, 'scp_statements.csv'))
    print(f'Copied scp_statements.csv to {out_dir}')

  print(f'Saved PTB-XL parquet dataset to {out_dir}')


def convert_capture24(data_dir, out_dir, verbose=False):
  """Convert Capture-24 to parquet with train/test splits and ClassLabel."""
  from datasets import load_dataset
  from data.datasets.capture24 import Capture24

  # Load original HF dataset to get label names
  original_ds = load_dataset(data_dir)
  original_label_feature = original_ds['train'].features.get('label')
  if hasattr(original_label_feature, 'names'):
    label_names = original_label_feature.names
  else:
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
      'data': [all_data[i].T.tolist() for i in indices],
      'label': [int(labels[i]) for i in indices],
    }
    splits[split_name] = Dataset.from_dict(split_data, features=features)
    print(f'  {split_name}: {len(indices)} samples')

  _save_parquet(splits, out_dir)
  print(f'Saved Capture-24 parquet dataset to {out_dir}')


def convert_fixed_length(dataset_name, data_dir, out_dir, verbose=False):
  """Convert fixed-length ECG datasets using streaming to avoid OOM."""
  dataset_cls = DATASETS[dataset_name]

  if dataset_name == 'code-15':
    _convert_code15(data_dir, out_dir, verbose)
    return

  record_names = dataset_cls.find_records(data_dir)
  min_channel_size = None
  if hasattr(dataset_cls, 'record_duration'):
    min_channel_size = dataset_cls.record_duration * dataset_cls.sampling_frequency

  def gen():
    idx = 0
    for record_name in tqdm(record_names, desc=dataset_name, disable=not verbose):
      x = wfdb.rdrecord(record_name).p_signal  # (channel_size, num_channels)
      if min_channel_size is not None and x.shape[0] < min_channel_size:
        continue
      x = x.astype(np.float16)
      yield {'id': str(idx), 'data': x.T.tolist(), 'label': -1}
      idx += 1

  ds = Dataset.from_generator(gen)
  _save_parquet(ds, out_dir)
  print(f'Saved {dataset_name} parquet dataset to {out_dir}')


def _convert_code15(data_dir, out_dir, verbose=False):
  """Convert CODE-15 using streaming to avoid OOM."""
  dataset_obj = CODE15(data_dir)
  num_channels = len(CODE15.channels)
  max_channel_size = int(CODE15.record_duration * CODE15.sampling_frequency)

  def gen():
    idx = 0
    for x in tqdm(dataset_obj.stream_raw_data(), total=len(dataset_obj.record_list),
                  desc='code-15', disable=not verbose):
      # Only keep fixed-length records (skip variable)
      slices = []
      for channel in range(num_channels):
        start_idx = np.nonzero(x[:, channel])[0]
        if len(start_idx) == 0:
          slices.append((0, 0, 0))
        else:
          s, e = start_idx[0], start_idx[-1] + 1
          slices.append((s, e, e - s))
      starts, ends, channel_sizes = zip(*slices)
      start, end = min(starts), max(ends)
      channel_size = end - start
      if len(set(channel_sizes)) == 1 and channel_size == max_channel_size:
        x = x.astype(np.float16)
        yield {'id': str(idx), 'data': x.T.tolist(), 'label': -1}
        idx += 1

  ds = Dataset.from_generator(gen)
  _save_parquet(ds, out_dir)
  print(f'Saved code-15 parquet dataset to {out_dir}')


def convert_variable_length(dataset_name, data_dir, out_dir, verbose=False):
  """Convert variable-length ECG datasets using streaming to avoid OOM."""
  dataset_cls = DATASETS[dataset_name]
  record_names = dataset_cls.find_records(data_dir)

  def gen():
    for idx, record_name in enumerate(tqdm(record_names, desc=dataset_name, disable=not verbose)):
      x = wfdb.rdrecord(record_name).p_signal  # (channel_size, num_channels)
      x = x.astype(np.float16)
      yield {'id': str(idx), 'data': x.T.tolist(), 'label': -1}

  ds = Dataset.from_generator(gen)
  _save_parquet(ds, out_dir)
  print(f'Saved {dataset_name} parquet dataset to {out_dir}')


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
