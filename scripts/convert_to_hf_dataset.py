"""Convert raw datasets to HuggingFace Dataset format (sharded Parquet).

Each dataset is saved with columns: ['id', 'data', 'label']
  - id: unique identifier (str or int)
  - data: 2D array (num_channels, channel_size) channels first, float16
  - label: multi-hot list[int] (PTB-XL), ClassLabel int (Capture-24), or -1 (no label)

PTB-XL also includes:
  - label_names: list[str] of class names corresponding to each multi-hot position
  - strat_fold: int fold number for splitting

Capture-24 label column uses HuggingFace ClassLabel feature, so
  dataset.features['label'].names provides class names and num_classes.

Output format: Sharded parquet files saved as {out_dir}/{split}/XXXX.parquet
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

DEFAULT_SHARD_SIZE = 50000

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', required=True, help='path to raw data directory')
parser.add_argument('--dataset', choices=list(DATASETS), required=True, help='dataset type')
parser.add_argument('--out', required=True, help='output directory for HF dataset')
parser.add_argument('--task', choices=TASKS, default='all',
                    help='label task for PTB-XL (default: all). Ignored for non-PTB-XL datasets.')
parser.add_argument('--shard-size', type=int, default=DEFAULT_SHARD_SIZE,
                    help=f'max records per parquet shard (default: {DEFAULT_SHARD_SIZE})')
parser.add_argument('--verbose', action='store_true', help='verbose mode')
args = parser.parse_args()


def _flush_shard(batch, out_path, features=None):
  """Write a batch of records as a single parquet shard."""
  keys = batch[0].keys()
  dict_of_lists = {k: [r[k] for r in batch] for k in keys}
  ds = Dataset.from_dict(dict_of_lists, features=features)
  ds.to_parquet(out_path)


def _save_sharded(gen, out_dir, split='train', shard_size=DEFAULT_SHARD_SIZE, features=None):
  """Stream records from a generator and write sharded parquet files.

  Output: {out_dir}/{split}/0000.parquet, 0001.parquet, ...
  """
  split_dir = path.join(out_dir, split)
  os.makedirs(split_dir, exist_ok=True)

  batch = []
  shard_idx = 0
  total = 0

  for record in gen:
    batch.append(record)
    if len(batch) >= shard_size:
      _flush_shard(batch, path.join(split_dir, f'{shard_idx:04d}.parquet'), features)
      total += len(batch)
      shard_idx += 1
      batch = []

  if batch:
    _flush_shard(batch, path.join(split_dir, f'{shard_idx:04d}.parquet'), features)
    total += len(batch)
    shard_idx += 1

  print(f'  {split}/: {total} samples in {shard_idx} shard(s)')
  return total


def _save_dataset_sharded(dataset, out_dir, split='train', shard_size=DEFAULT_SHARD_SIZE):
  """Save an in-memory Dataset as sharded parquet files."""
  split_dir = path.join(out_dir, split)
  os.makedirs(split_dir, exist_ok=True)

  num_shards = max(1, (len(dataset) + shard_size - 1) // shard_size)
  for i in range(num_shards):
    shard = dataset.shard(num_shards=num_shards, index=i, contiguous=True)
    shard.to_parquet(path.join(split_dir, f'{i:04d}.parquet'))

  print(f'  {split}/: {len(dataset)} samples in {num_shards} shard(s)')


def convert_ptb_xl(data_dir, out_dir, task='all', shard_size=DEFAULT_SHARD_SIZE, verbose=False):
  """Convert PTB-XL to sharded parquet with train/val/test splits and multi-hot labels."""
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

  os.makedirs(out_dir, exist_ok=True)
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
    ds = Dataset.from_dict(split_data)
    _save_dataset_sharded(ds, out_dir, split=split_name, shard_size=shard_size)

  # Copy scp_statements.csv needed for label aggregation if task changes later
  import shutil
  scp_file = path.join(data_dir, 'scp_statements.csv')
  if path.isfile(scp_file):
    shutil.copy2(scp_file, path.join(out_dir, 'scp_statements.csv'))
    print(f'Copied scp_statements.csv to {out_dir}')

  print(f'Saved PTB-XL sharded parquet to {out_dir}')


def convert_capture24(data_dir, out_dir, shard_size=DEFAULT_SHARD_SIZE, verbose=False):
  """Convert Capture-24 to sharded parquet with train/test splits and ClassLabel."""
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

  os.makedirs(out_dir, exist_ok=True)
  for split_name in ('train', 'test'):
    mask = split_names == split_name
    indices = np.where(mask)[0]
    split_data = {
      'id': [str(i) for i in indices],
      'data': [all_data[i].T.tolist() for i in indices],
      'label': [int(labels[i]) for i in indices],
    }
    ds = Dataset.from_dict(split_data, features=features)
    _save_dataset_sharded(ds, out_dir, split=split_name, shard_size=shard_size)

  print(f'Saved Capture-24 sharded parquet to {out_dir}')


def convert_fixed_length(dataset_name, data_dir, out_dir, shard_size=DEFAULT_SHARD_SIZE, verbose=False):
  """Convert fixed-length ECG datasets with streaming + sharding to avoid OOM."""
  dataset_cls = DATASETS[dataset_name]

  if dataset_name == 'code-15':
    _convert_code15(data_dir, out_dir, shard_size, verbose)
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

  os.makedirs(out_dir, exist_ok=True)
  _save_sharded(gen(), out_dir, split='train', shard_size=shard_size)
  print(f'Saved {dataset_name} sharded parquet to {out_dir}')


def _convert_code15(data_dir, out_dir, shard_size=DEFAULT_SHARD_SIZE, verbose=False):
  """Convert CODE-15 with streaming + sharding to avoid OOM."""
  dataset_obj = CODE15(data_dir)
  num_channels = len(CODE15.channels)
  max_channel_size = int(CODE15.record_duration * CODE15.sampling_frequency)

  def gen():
    idx = 0
    for x in tqdm(dataset_obj.stream_raw_data(), total=len(dataset_obj.record_list),
                  desc='code-15', disable=not verbose):
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

  os.makedirs(out_dir, exist_ok=True)
  _save_sharded(gen(), out_dir, split='train', shard_size=shard_size)
  print(f'Saved code-15 sharded parquet to {out_dir}')


def convert_variable_length(dataset_name, data_dir, out_dir, shard_size=DEFAULT_SHARD_SIZE, verbose=False):
  """Convert variable-length ECG datasets with streaming + sharding to avoid OOM."""
  dataset_cls = DATASETS[dataset_name]
  record_names = dataset_cls.find_records(data_dir)

  def gen():
    for idx, record_name in enumerate(tqdm(record_names, desc=dataset_name, disable=not verbose)):
      x = wfdb.rdrecord(record_name).p_signal  # (channel_size, num_channels)
      x = x.astype(np.float16)
      yield {'id': str(idx), 'data': x.T.tolist(), 'label': -1}

  os.makedirs(out_dir, exist_ok=True)
  _save_sharded(gen(), out_dir, split='train', shard_size=shard_size)
  print(f'Saved {dataset_name} sharded parquet to {out_dir}')


# Dataset type classification
FIXED_LENGTH_DATASETS = {
  'chapman-shaoxing', 'georgia', 'ningbo', 'st-petersburg',
  'code-15', 'mimic-iv-ecg',
}
VARIABLE_LENGTH_DATASETS = {'cpsc', 'cpsc-extra', 'ptb'}

print(f'Converting {args.dataset} from {args.data_dir}')

if args.dataset == 'ptb-xl':
  convert_ptb_xl(args.data_dir, args.out, task=args.task, shard_size=args.shard_size, verbose=args.verbose)
elif args.dataset == 'capture-24':
  convert_capture24(args.data_dir, args.out, shard_size=args.shard_size, verbose=args.verbose)
elif args.dataset in FIXED_LENGTH_DATASETS:
  convert_fixed_length(args.dataset, args.data_dir, args.out, shard_size=args.shard_size, verbose=args.verbose)
elif args.dataset in VARIABLE_LENGTH_DATASETS:
  convert_variable_length(args.dataset, args.data_dir, args.out, shard_size=args.shard_size, verbose=args.verbose)
else:
  raise ValueError(f'Unknown dataset type: {args.dataset}')
