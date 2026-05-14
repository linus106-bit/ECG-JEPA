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
  python -m scripts.convert_to_hf_dataset --data-dir /path/to/ptb-xl --dataset ptb-xl --task all \
      --ptb-xl-labels AFIB,1AVB,2AVB,SVTAC,PVC,PAC --out /path/to/output_6labels
  python -m scripts.convert_to_hf_dataset --data-dir /path/to/ptb-xl --dataset ptb-xl --ptb-xl-sampling-frequency 100 --out /path/to/output_100hz
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

from data import transforms
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
parser.add_argument('--ptb-xl-labels', default=None,
                    help='comma-separated PTB-XL label names to keep after task aggregation, e.g. '
                         'AFIB,1AVB,2AVB,SVTAC,PVC,PAC. Only valid for --dataset ptb-xl.')
parser.add_argument('--drop-unselected-label-samples', action='store_true',
                    help='for --dataset ptb-xl with --ptb-xl-labels, drop samples that have none of '
                         'the selected labels. By default all samples from the chosen task are kept, '
                         'with all-zero labels for samples outside the selected label set.')
parser.add_argument('--shard-size', type=int, default=DEFAULT_SHARD_SIZE,
                    help=f'max records per parquet shard (default: {DEFAULT_SHARD_SIZE})')
parser.add_argument('--normalize', action='store_true',
                    help='apply pretrain-style normalization: NaN interpolation → per-channel '
                         'normalize (using dataset mean/std) → clip to [-5, 5]. '
                         'Output dtype becomes float32 instead of float16.')
parser.add_argument('--ptb-xl-sampling-frequency', type=int, choices=(100, 500), default=500,
                    help='sampling frequency to export for PTB-XL: 500 uses filename_hr, '
                         '100 uses filename_lr (default: 500). Ignored for non-PTB-XL datasets.')
parser.add_argument('--verbose', action='store_true', help='verbose mode')
args = parser.parse_args()


def _normalize_ecg(x, dataset_cls):
  """Apply pretrain-style normalization to a single ECG record.

  Mirrors the PreprocessECG logic in pretrain.py (minus resampling and channel reordering):
    1. NaN interpolation
    2. Per-channel z-score normalization using dataset_cls.mean / .std
    3. Clip to [-5, 5]

  Args:
    x: np.ndarray of shape (num_channels, channel_size), channels-first
    dataset_cls: dataset class with .mean and .std class attributes
  Returns:
    float32 np.ndarray of the same shape
  """
  x = x.astype(np.float32)
  transforms.interpolate_NaNs_(x)
  mean = np.array(dataset_cls.mean, dtype=np.float32).reshape(-1, 1)
  std = np.array(dataset_cls.std, dtype=np.float32).reshape(-1, 1)
  transforms.normalize_(x, mean_std=(mean, std))
  x.clip(-5, 5, out=x)
  return x


def _parse_label_names(label_names):
  """Parse a comma-separated label list while preserving user-provided order."""
  if label_names is None:
    return None

  parsed = [label.strip() for label in label_names.split(',') if label.strip()]
  if not parsed:
    raise ValueError('--ptb-xl-labels was provided but no valid label names were found')

  normalized = [label.upper() for label in parsed]
  duplicates = sorted({label for label in normalized if normalized.count(label) > 1})
  if duplicates:
    raise ValueError(f'--ptb-xl-labels contains duplicate label(s): {duplicates}')

  return normalized


def _select_multihot_labels(y_multihot, label_names, selected_label_names, drop_unselected_samples=False):
  """Keep only selected multi-hot label columns for PTB-XL."""
  if selected_label_names is None:
    return y_multihot, label_names, None

  label_to_idx = {name.upper(): idx for idx, name in enumerate(label_names)}
  missing_labels = [label for label in selected_label_names if label not in label_to_idx]
  if missing_labels:
    raise ValueError(
      f'selected PTB-XL label(s) are not present for this task: {missing_labels}. '
      f'Available labels: {label_names}'
    )

  selected_indices = [label_to_idx[label] for label in selected_label_names]
  y_selected = y_multihot[:, selected_indices]
  selected_sample_indices = None

  if drop_unselected_samples:
    selected_sample_indices = np.flatnonzero(y_selected.sum(axis=1) > 0)
    y_selected = y_selected[selected_sample_indices]

  return y_selected, selected_label_names, selected_sample_indices


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


def convert_ptb_xl(data_dir, out_dir, task='all', shard_size=DEFAULT_SHARD_SIZE, verbose=False,
                   normalize=False, sampling_frequency=500, selected_label_names=None, drop_unselected_samples=False):
  """Convert PTB-XL to sharded parquet with train/val/test splits and multi-hot labels.

  PTB-XL ships with both 500 Hz records (``filename_hr``) and 100 Hz records
  (``filename_lr``). ``sampling_frequency`` selects which set of WFDB records to
  read while keeping the same labels and stratified folds.
  """
  from data.utils import load_raw_data

  record_names = PTB_XL.find_records(data_dir, sampling_frequency=sampling_frequency)
  data = load_raw_data(record_names, verbose=verbose)

  # Load raw labels and compute aggregations for the specified task
  labels_df = PTB_XL.load_raw_labels(data_dir)
  labels_df = PTB_XL.compute_label_aggregations(labels_df, data_dir, task)

  # Get multi-hot labels and class names via select_data
  all_x, labels_df, y_multihot, mlb = PTB_XL.select_data(data, labels_df, task, min_samples=0)

  label_names = list(mlb.classes_)
  y_multihot, label_names, selected_sample_indices = _select_multihot_labels(
    y_multihot, label_names, selected_label_names,
    drop_unselected_samples=drop_unselected_samples)

  if selected_sample_indices is not None:
    all_x = all_x[selected_sample_indices]
    labels_df = labels_df.iloc[selected_sample_indices]

  strat_folds = labels_df.strat_fold.values.tolist()
  ecg_ids = labels_df.index.tolist()

  print(f'  sampling_frequency={sampling_frequency}Hz')
  print(f'  task={task}, num_classes={len(label_names)}, label_names={label_names}')
  if selected_label_names is not None:
    print(f'  selected PTB-XL labels: {label_names}')
    print(f'  drop_unselected_label_samples={drop_unselected_samples}')
  print(f'  total samples after filtering: {len(all_x)}')

  os.makedirs(out_dir, exist_ok=True)
  for split_name, fold_values in [('train', range(1, 9)), ('val', [9]), ('test', [10])]:
    fold_set = set(fold_values)
    indices = [i for i, fold in enumerate(strat_folds) if fold in fold_set]
    if normalize:
      data_list = [_normalize_ecg(all_x[i].T, PTB_XL).tolist() for i in indices]
    else:
      data_list = [all_x[i].T.tolist() for i in indices]
    split_data = {
      'id': [str(ecg_ids[i]) for i in indices],
      'data': data_list,
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

  print(f'Saved PTB-XL {sampling_frequency}Hz sharded parquet to {out_dir}')


def convert_capture24(data_dir, out_dir, shard_size=DEFAULT_SHARD_SIZE, verbose=False,
                      normalize=False):
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
    if normalize:
      data_list = [_normalize_ecg(all_data[i].T, Capture24).tolist() for i in indices]
    else:
      data_list = [all_data[i].T.tolist() for i in indices]
    split_data = {
      'id': [str(i) for i in indices],
      'data': data_list,
      'label': [int(labels[i]) for i in indices],
    }
    ds = Dataset.from_dict(split_data, features=features)
    _save_dataset_sharded(ds, out_dir, split=split_name, shard_size=shard_size)

  print(f'Saved Capture-24 sharded parquet to {out_dir}')


def convert_fixed_length(dataset_name, data_dir, out_dir, shard_size=DEFAULT_SHARD_SIZE,
                         verbose=False, normalize=False):
  """Convert fixed-length ECG datasets with streaming + sharding to avoid OOM."""
  dataset_cls = DATASETS[dataset_name]

  if dataset_name == 'code-15':
    _convert_code15(data_dir, out_dir, shard_size, verbose, normalize=normalize)
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
      if normalize:
        x_cf = _normalize_ecg(x.T, dataset_cls)
      else:
        x_cf = x.T.astype(np.float16)
      yield {'id': str(idx), 'data': x_cf.tolist(), 'label': -1}
      idx += 1

  os.makedirs(out_dir, exist_ok=True)
  _save_sharded(gen(), out_dir, split='train', shard_size=shard_size)
  print(f'Saved {dataset_name} sharded parquet to {out_dir}')


def _convert_code15(data_dir, out_dir, shard_size=DEFAULT_SHARD_SIZE, verbose=False,
                    normalize=False):
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
        if normalize:
          x_cf = _normalize_ecg(x.T, CODE15)
        else:
          x_cf = x.astype(np.float16).T
        yield {'id': str(idx), 'data': x_cf.tolist(), 'label': -1}
        idx += 1

  os.makedirs(out_dir, exist_ok=True)
  _save_sharded(gen(), out_dir, split='train', shard_size=shard_size)
  print(f'Saved code-15 sharded parquet to {out_dir}')


def convert_variable_length(dataset_name, data_dir, out_dir, shard_size=DEFAULT_SHARD_SIZE,
                            verbose=False, normalize=False):
  """Convert variable-length ECG datasets with streaming + sharding to avoid OOM."""
  dataset_cls = DATASETS[dataset_name]
  record_names = dataset_cls.find_records(data_dir)

  def gen():
    for idx, record_name in enumerate(tqdm(record_names, desc=dataset_name, disable=not verbose)):
      x = wfdb.rdrecord(record_name).p_signal  # (channel_size, num_channels)
      if normalize:
        x_cf = _normalize_ecg(x.T, dataset_cls)
      else:
        x_cf = x.T.astype(np.float16)
      yield {'id': str(idx), 'data': x_cf.tolist(), 'label': -1}

  os.makedirs(out_dir, exist_ok=True)
  _save_sharded(gen(), out_dir, split='train', shard_size=shard_size)
  print(f'Saved {dataset_name} sharded parquet to {out_dir}')


# Dataset type classification
FIXED_LENGTH_DATASETS = {
  'chapman-shaoxing', 'georgia', 'ningbo', 'st-petersburg',
  'code-15', 'mimic-iv-ecg',
}
VARIABLE_LENGTH_DATASETS = {'cpsc', 'cpsc-extra', 'ptb'}

selected_label_names = _parse_label_names(args.ptb_xl_labels)
if selected_label_names is not None and args.dataset != 'ptb-xl':
  raise ValueError('--ptb-xl-labels is only valid with --dataset ptb-xl')
if args.drop_unselected_label_samples and selected_label_names is None:
  raise ValueError('--drop-unselected-label-samples requires --ptb-xl-labels')

print(f'Converting {args.dataset} from {args.data_dir}')

if args.dataset == 'ptb-xl':
  convert_ptb_xl(args.data_dir, args.out, task=args.task, shard_size=args.shard_size,
                 verbose=args.verbose, normalize=args.normalize,
                 selected_label_names=selected_label_names,
                 drop_unselected_samples=args.drop_unselected_label_samples,
                 sampling_frequency=args.ptb_xl_sampling_frequency)
elif args.dataset == 'capture-24':
  convert_capture24(args.data_dir, args.out, shard_size=args.shard_size,
                    verbose=args.verbose, normalize=args.normalize)
elif args.dataset in FIXED_LENGTH_DATASETS:
  convert_fixed_length(args.dataset, args.data_dir, args.out, shard_size=args.shard_size,
                       verbose=args.verbose, normalize=args.normalize)
elif args.dataset in VARIABLE_LENGTH_DATASETS:
  convert_variable_length(args.dataset, args.data_dir, args.out, shard_size=args.shard_size,
                          verbose=args.verbose, normalize=args.normalize)
else:
  raise ValueError(f'Unknown dataset type: {args.dataset}')
