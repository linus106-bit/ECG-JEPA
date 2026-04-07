import argparse
import copy
import dataclasses
import json
import logging
import logging.config
import os
import pprint
from contextlib import nullcontext
from datetime import datetime
from os import path, makedirs
from pathlib import Path
from time import time

from tqdm import tqdm

import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import configs
from data import transforms, utils as datautils
from data.datasets import DATASETS, PTB_XL, Capture24, SDB
from data.utils import TensorDataset, get_channel_order
from models import create_encoder, EncoderClassifier
from utils.monitoring import AverageMeter, get_memory_usage, get_cpu_count
from utils.schedules import update_learning_rate_, cosine_schedule

TASKS = (
  'all',
  'diagnostic',
  'subdiagnostic',
  'superdiagnostic',
  'form',
  'rhythm',
  # custom tasks
  'ST-MEM',  # Na et al. (2024)
)
VAL_RATIO = 0.2
VAL_SEED = 42

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default=None, help='path to HF dataset directory (overrides dataset.data_dir in config)')
parser.add_argument('--encoder', default=None, help='path to checkpoint or config file (default: run.encoder in config yaml)')
parser.add_argument('--out', default=None, help='output directory (default: run.out_dir in config yaml)')
parser.add_argument('--config', default=None, help='path to config file or config name (default: linear for ecg, har_linear for har)')
parser.add_argument('--amp', default=None, choices=['bfloat16', 'float32'], help='precision (default: run.amp in config yaml)')
parser.add_argument('--dataset-type', choices=['ecg', 'har', 'ppg'], default=None,
                    help='dataset type: ecg (multi-label, ROC-AUC), har (single-label), or ppg (single-label). '
                         'Auto-detected from data_dir if not specified.')
# ECG-specific args (PTB-XL tasks)
parser.add_argument('--task', choices=TASKS, default=None, help='task type (ECG only, default: run.task in config yaml)')
args = parser.parse_args()


def _find_capture24_legacy_prefix(data_dir):
  """Return prefix for legacy Capture24 dumps if available, else None."""
  candidates = [data_dir]
  if path.isdir(data_dir):
    dirname = path.basename(path.normpath(data_dir))
    candidates.extend([
      path.join(data_dir, 'capture24'),
      path.join(data_dir, dirname),
    ])

  required_suffixes = (
    '_train.npy',
    '_test.npy',
    '_train_labels.npz',
    '_test_labels.npz',
  )
  for prefix in candidates:
    if all(path.isfile(f'{prefix}{suffix}') for suffix in required_suffixes):
      return prefix
  return None


def _load_capture24_legacy(prefix):
  """Load legacy Capture24 dumps from a resolved prefix."""
  x_train_all = np.load(f'{prefix}_train.npy', mmap_mode='r')  # (N, T, C), channels-last
  x_test = np.load(f'{prefix}_test.npy', mmap_mode='r')
  with np.load(f'{prefix}_train_labels.npz') as archive:
    train_labels = archive['labels'].copy()
  with np.load(f'{prefix}_test_labels.npz') as archive:
    test_labels = archive['labels'].copy()
  return x_train_all, train_labels, x_test, test_labels






def _canonicalize_single_label_array(x, num_channels):
  """Convert single-label input arrays to (N, C, T)."""
  x = np.asarray(x, dtype=np.float16)

  # Remove singleton dimensions except batch when possible
  while x.ndim > 3:
    squeeze_axes = tuple(i for i in range(1, x.ndim) if x.shape[i] == 1)
    if not squeeze_axes:
      break
    x = np.squeeze(x, axis=squeeze_axes)

  if x.ndim == 2:
    # (N, T) -> (N, 1, T)
    x = x[:, None, :]
  elif x.ndim != 3:
    raise ValueError(f'Expected single-label array with 2D/3D shape, got {x.shape}')

  # Normalize layout to (N, C, T)
  if x.shape[1] == num_channels:
    return x
  if x.shape[2] == num_channels:
    return np.transpose(x, (0, 2, 1))

  # Fallback for one-channel signals with ambiguous axes
  if num_channels == 1:
    if x.shape[1] == 1:
      return x
    if x.shape[2] == 1:
      return np.transpose(x, (0, 2, 1))

  raise ValueError(f'Could not infer (N, C, T) layout for shape {x.shape} with num_channels={num_channels}')


def main():
  # Load config YAML and extract run: section early (needed before distributed setup)
  if args.config is None:
    args.config = path.join(path.dirname(configs.eval.__file__), 'linear.yaml')
  if not path.isfile(args.config):
    raise ValueError(f'Config file not found: {args.config}')
  _early_dict = configs.load_config_file(args.config)
  _run = _early_dict.pop('run', {})
  if args.encoder is None:
    args.encoder = _run.get('encoder')
  if args.out is None:
    args.out = _run.get('out_dir', path.join('finetune', Path(args.config).stem))
  if args.amp is None:
    args.amp = _run.get('amp', 'float32')
  if args.dataset_type is None:
    args.dataset_type = _run.get('dataset_type')
  if args.task is None:
    args.task = _run.get('task', 'all')
  if not args.encoder:
    raise ValueError('encoder must be specified via --encoder or run.encoder in the eval config yaml')

  # Setup distributed training
  local_rank = int(os.environ.get('LOCAL_RANK', 0))
  rank = int(os.environ.get('RANK', 0))
  world_size = int(os.environ.get('WORLD_SIZE', 1))
  is_distributed = world_size > 1
  is_main_process = rank == 0

  if is_distributed:
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)

  if is_main_process:
    makedirs(args.out, exist_ok=True)
    logging.config.fileConfig('logging.ini')
    _timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    _log_file_path = path.join(args.out, f'train_{_timestamp}.log')
    _file_handler = logging.FileHandler(_log_file_path)
    _file_handler.setLevel(logging.DEBUG)
    _file_handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s %(module)s:%(lineno)s => %(message)s'))
    logging.getLogger('app').addHandler(_file_handler)
  logger = logging.getLogger('app')
  if not is_main_process:
    logger.setLevel(logging.CRITICAL)
  else:
    logger.info(f'logging to {_log_file_path}')

  device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
  using_cuda = device.type == 'cuda'
  num_cpus = get_cpu_count()
  if is_main_process:
    logger.debug(f'using {device} accelerator, {num_cpus} CPUs, world_size={world_size}')

  if using_cuda:
    if is_main_process:
      logger.debug('TF32 tensor cores are enabled')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

  if args.amp == 'float32' or not using_cuda:  # don't use AMP on a CPU
    if is_main_process:
      logger.debug('using float32 precision')
    auto_mixed_precision = nullcontext()
  elif args.amp == 'bfloat16':
    # bfloat16 preserves the range of float32, so it does not require scaling
    if is_main_process:
      logger.debug('using bfloat16 with AMP')
    auto_mixed_precision = torch.cuda.amp.autocast(dtype=torch.bfloat16)
  else:
    raise ValueError('Failed to choose floating-point format.')

  # Auto-detect dataset type from data_dir if not explicitly specified
  dataset_type = args.dataset_type
  if dataset_type is None:
    data_dir_lower = (args.data_dir or '').lower()
    if 'capture24' in data_dir_lower or 'capture-24' in data_dir_lower:
      dataset_type = 'har'
    elif 'sdb' in data_dir_lower:
      dataset_type = 'ppg'
    else:
      dataset_type = 'ecg'

  eval_config_dict = configs.load_config_file(args.config)
  eval_config_dict.pop('run', None)
  if is_main_process:
    logger.debug(f'loading configuration file from {args.config}\n'
                 f'{pprint.pformat(eval_config_dict, compact=True, sort_dicts=False, width=120)}')

  # resolve data_dir from CLI args or config yaml
  dataset_cfg = eval_config_dict.pop('dataset', None) or {}
  data_dir = args.data_dir or dataset_cfg.get('data_dir')
  if not data_dir:
    raise ValueError('data_dir must be specified via --data-dir or dataset.data_dir in the eval config yaml')

  # load encoder checkpoint or config
  _, ext = path.splitext(args.encoder)
  if ext == '.yaml':
    if is_main_process:
      logger.debug(f'loading encoder config from {args.encoder}')
    encoder_config_dict = configs.load_config_file(args.encoder)
    encoder_config_dict.pop('run', None)
    encoder_config = configs.pretrain.Config(**encoder_config_dict)
    model_state_dict = None
  else:
    if is_main_process:
      logger.debug(f'loading encoder checkpoint from {args.encoder}')
    chkpt = torch.load(args.encoder, map_location='cpu')
    encoder_config_dict = chkpt['config']
    encoder_config_dict.pop('run', None)
    encoder_config = configs.pretrain.Config(**encoder_config_dict)
    if 'eval_config' in chkpt:  # continue fine-tuning the weights
      model_state_dict = chkpt['model']
    else:  # extract target encoder's weights from the checkpoint
      model_state_dict = {'encoder.' + k.removeprefix('target_encoder.'): v
                          for k, v in chkpt['model'].items()
                          if k.startswith('target_encoder.')}

  # -------------------------------------------------------------------------
  # Unified dataset loading from HuggingFace datasets
  # Data is stored channels-first: (num_channels, channel_size)
  # -------------------------------------------------------------------------
  if is_main_process:
    logger.debug(f'loading dataset from {data_dir} (type={dataset_type})')

  legacy_prefix = _find_capture24_legacy_prefix(data_dir)
  hf_dataset = None
  available_splits = []
  if legacy_prefix is not None:
    if is_main_process:
      logger.debug(f'found legacy Capture24 dump prefix: {legacy_prefix}')
  elif path.isdir(data_dir):
    from datasets import load_dataset
    hf_dataset = load_dataset(data_dir)
    available_splits = list(hf_dataset.keys())
    if is_main_process:
      logger.debug(f'available splits: {available_splits}')
  elif is_main_process:
    logger.debug(f'data_dir is not a directory: {data_dir}')

  if dataset_type in ('har', 'ppg'):
    # Single-label classification for HAR/PPG datasets
    single_label = True
    task_name = dataset_type
    dataset_cls = Capture24 if dataset_type == 'har' else SDB

    har_transpose_input = legacy_prefix is not None
    if legacy_prefix is not None:
      x_train_all, train_labels, x_test, test_labels = _load_capture24_legacy(legacy_prefix)
      train_labels = np.asarray(train_labels, dtype=np.int64)
      test_labels = np.asarray(test_labels, dtype=np.int64)
      num_classes = int(max(train_labels.max(), test_labels.max()) + 1)
      if is_main_process:
        logger.debug('loaded HAR data from legacy dump files')
    elif hf_dataset is not None:
      # Get num_classes from ClassLabel feature
      label_feature = hf_dataset['train'].features.get('label')
      if hasattr(label_feature, 'names'):
        num_classes = len(label_feature.names)
        if is_main_process:
          logger.debug(f'ClassLabel names: {label_feature.names}')
      else:
        # fallback
        train_labels_tmp = np.array(hf_dataset['train']['label'], dtype=np.int64)
        test_labels_tmp = np.array(hf_dataset['test']['label'], dtype=np.int64)
        num_classes = int(max(train_labels_tmp.max(), test_labels_tmp.max()) + 1)

      # Load train split: data is (N, num_channels, channel_size)
      train_ds = hf_dataset['train']
      x_train_all = np.array(train_ds['data'], dtype=np.float16)
      train_labels = np.array(train_ds['label'], dtype=np.int64)

      # Load test split
      test_ds = hf_dataset['test']
      x_test = np.array(test_ds['data'], dtype=np.float16)
      test_labels = np.array(test_ds['label'], dtype=np.int64)
    else:
      raise ValueError(f'Could not load single-label HAR/PPG dataset from {data_dir}. '
                       f'Expected HF dataset directory or legacy Capture24 dump prefix.')

    # Split train into train/val with a fixed random seed
    rng = np.random.RandomState(VAL_SEED)
    n_train_all = len(x_train_all)
    indices = np.arange(n_train_all)
    rng.shuffle(indices)
    num_val = int(n_train_all * VAL_RATIO)
    val_indices = indices[:num_val]
    train_indices = indices[num_val:]

    if is_main_process:
      logger.debug(f'train={len(train_indices)}, val={len(val_indices)}, '
                   f'test={len(x_test)}, num_classes={num_classes}')

    # Canonicalize to channels-first (N, C, T), then compute per-channel stats.
    num_channels = len(dataset_cls.channels)
    x_train_all = _canonicalize_single_label_array(x_train_all, num_channels=num_channels)
    x_test = _canonicalize_single_label_array(x_test, num_channels=num_channels)
    har_transpose_input = False

    mean = np.mean(x_train_all[train_indices], axis=(0, 2), keepdims=True, dtype=np.float32)
    std = np.std(x_train_all[train_indices], axis=(0, 2), keepdims=True, dtype=np.float32)

    y_train_all = torch.from_numpy(train_labels).long()
    y_test = torch.from_numpy(test_labels).long()
    x_train = x_train_all[train_indices]
    y_train = y_train_all[train_indices]
    x_val = x_train_all[val_indices]
    y_val = y_train_all[val_indices]

  else:
    # ECG: multi-label classification (e.g., PTB-XL)
    # HF dataset stores pre-computed multi-hot labels and label_names
    # Splits (train/val/test) are pre-computed during conversion
    dataset_cls = PTB_XL
    single_label = False
    task_name = args.task

    if args.task == 'ST-MEM':
      single_label = True

    if 'val' not in available_splits:
      raise ValueError('ECG dataset must have train/val/test splits. '
                       'Use scripts/convert_to_hf_dataset.py to create the proper format.')

    if is_main_process:
      logger.debug('loading from HF dataset with train/val/test splits')

    # Load each split directly: data is (N, num_channels, channel_size)
    def _load_ecg_split(split_ds):
      x = np.array(split_ds['data'], dtype=np.float16)
      y = np.array(split_ds['label'], dtype=np.float32)
      return x, y

    x_train, y_train = _load_ecg_split(hf_dataset['train'])
    x_val, y_val = _load_ecg_split(hf_dataset['val'])
    x_test, y_test = _load_ecg_split(hf_dataset['test'])

    # Get label_names and num_classes from the dataset
    label_names = hf_dataset['train'][0]['label_names']
    num_classes = len(label_names)
    if is_main_process:
      logger.debug(f'num_classes={num_classes}, label_names={label_names}')

    if single_label:
      for name, (x_arr, y_arr) in [('train', (x_train, y_train)),
                                     ('val', (x_val, y_val)),
                                     ('test', (x_test, y_test))]:
        mask = y_arr.sum(axis=1) == 1
        if name == 'train':
          x_train, y_train = x_arr[mask], y_arr[mask]
        elif name == 'val':
          x_val, y_val = x_arr[mask], y_arr[mask]
        else:
          x_test, y_test = x_arr[mask], y_arr[mask]

    # Preprocess: resample if needed (channels-first: shape is (N, C, T))
    channel_size = PTB_XL.record_duration * encoder_config.sampling_frequency
    if channel_size != x_train.shape[2]:
      from multiprocessing import Pool
      preprocess = PreprocessECG(channel_size=channel_size, remove_baseline_wander=False)
      with Pool(num_cpus) as pool:
        x_train = np.array(pool.map(preprocess, [x_train[i] for i in range(len(x_train))]))
        x_val = np.array(pool.map(preprocess, [x_val[i] for i in range(len(x_val))]))
        x_test = np.array(pool.map(preprocess, [x_test[i] for i in range(len(x_test))]))

    # normalize using training statistics (channels-first: axis=(0, 2) for per-channel stats)
    mean = np.mean(x_train, axis=(0, 2), keepdims=True, dtype=np.float32)
    std = np.std(x_train, axis=(0, 2), keepdims=True, dtype=np.float32)
    transforms.normalize_(x_train, mean_std=(mean, std))
    x_train.clip(-5, 5, out=x_train)
    transforms.normalize_(x_val, mean_std=(mean, std))
    x_val.clip(-5, 5, out=x_val)
    transforms.normalize_(x_test, mean_std=(mean, std))
    x_test.clip(-5, 5, out=x_test)

    # ensure matching channels (channels-first: index along axis 1)
    channel_order = datautils.get_channel_order(PTB_XL.channels, encoder_config.channels)
    x_train = x_train[:, channel_order]
    x_val = x_val[:, channel_order]
    x_test = x_test[:, channel_order]

    y_train = torch.from_numpy(y_train).float()
    y_val = torch.from_numpy(y_val).float()
    y_test = torch.from_numpy(y_test).float()

  if is_main_process:
    logger.debug(f'{get_memory_usage() / 1024 ** 3:,.2f}GB memory used after loading data')

  # -------------------------------------------------------------------------
  # Config + model setup (shared)
  # -------------------------------------------------------------------------
  eval_config = configs.eval.Config(**eval_config_dict, num_classes=num_classes)
  if eval_config.use_register and encoder_config.num_registers == 0:
    if is_main_process:
      logger.debug('adding a randomly initialized register to the encoder')
    encoder_config = dataclasses.replace(encoder_config, num_registers=1)

  if eval_config.dropout != encoder_config.dropout:
    if is_main_process:
      logger.debug('overriding encoder dropout')
    encoder_config = dataclasses.replace(encoder_config, dropout=eval_config.dropout)

  if encoder_config.layer_scale_eps == 0 and eval_config.layer_scale_eps > 0:
    if is_main_process:
      logger.debug('adding LayerScale to the encoder')
    encoder_config = dataclasses.replace(encoder_config, layer_scale_eps=eval_config.layer_scale_eps)

  if eval_config.crop_duration is not None:
    crop_size = int(eval_config.crop_duration * encoder_config.sampling_frequency)
    if eval_config.crop_stride is not None:
      crop_stride = int(eval_config.crop_stride * encoder_config.sampling_frequency)
    else:
      crop_stride = crop_size
  else:
    crop_size = None
    crop_stride = None

  local_batch_size = eval_config.batch_size // world_size
  # Cap dataloader workers to avoid exhausting file descriptors on hosts
  # with very high CPU counts (e.g. single-GPU jobs on large machines).
  num_workers = min(8, max(1, num_cpus // world_size))
  eval_num_workers = min(2, num_workers)

  if dataset_type in ('har', 'ppg'):
    har_preprocess = PreprocessHAR(
      mean_std=(mean, std),
      channel_order=get_channel_order(dataset_cls.channels, encoder_config.channels),
      transpose_input=har_transpose_input)
    train_transform = [har_preprocess, TrainTransform(crop_size=crop_size)]
    eval_transform = [har_preprocess, EvalTransform(crop_size=crop_size, crop_stride=crop_stride)]
  else:
    train_transform = TrainTransform(crop_size=crop_size)
    eval_transform = EvalTransform(crop_size=crop_size, crop_stride=crop_stride)

  train_dataset = TensorDataset(
    data=x_train,
    labels=y_train,
    transform=train_transform)

  train_sampler = DistributedSampler(
    train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
  ) if is_distributed else None

  train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=local_batch_size,
    sampler=train_sampler,
    shuffle=(train_sampler is None),
    drop_last=(train_sampler is None),
    num_workers=num_workers,
    persistent_workers=(num_workers > 0))

  val_loader = DataLoader(
    dataset=TensorDataset(
      data=x_val,
      labels=y_val,
      transform=eval_transform),
    batch_size=eval_config.batch_size,
    num_workers=eval_num_workers)
  test_loader = DataLoader(
    dataset=TensorDataset(
      data=x_test,
      labels=y_test,
      transform=eval_transform),
    batch_size=eval_config.batch_size,
    num_workers=eval_num_workers)

  steps_per_epoch = len(train_loader) if eval_config.epochs > 0 else None
  total_steps = eval_config.epochs * steps_per_epoch if eval_config.epochs > 0 else eval_config.steps

  lr_schedule = cosine_schedule(
    total_steps=total_steps,
    start_value=eval_config.learning_rate,
    final_value=eval_config.final_learning_rate,
    warmup_steps=int(total_steps * eval_config.learning_rate_warmup_ratio),
    warmup_start_value=1e-6)

  encoder = create_encoder(
    config=encoder_config,
    keep_registers=eval_config.use_register,
    use_sdp_kernel=using_cuda)
  original_model = EncoderClassifier(encoder, eval_config, use_sdp_kernel=using_cuda).to(device)
  optimizer = original_model.get_optimizer(fused=using_cuda)

  if model_state_dict is not None:
    incompatible_keys = original_model.load_state_dict(model_state_dict, strict=False)
    for key in incompatible_keys.missing_keys:
      if is_main_process:
        logger.debug(f'missing {key} in the encoder checkpoint')
    for key in incompatible_keys.unexpected_keys:
      if is_main_process:
        logger.debug(f'unexpected {key} in the encoder checkpoint')

  if is_distributed:
    model = DDP(original_model, device_ids=[local_rank])
  else:
    model = original_model

  # -------------------------------------------------------------------------
  # Training loop (shared structure, branched on single_label for metrics)
  # -------------------------------------------------------------------------
  step_time = AverageMeter()
  train_loss = AverageMeter()
  best_val_metric = float('-inf')
  best_val_predictions, saved_val_targets = None, None
  best_epoch_or_step = None
  best_chkpt = None
  global_step = 0
  train_start_time = time()
  last_loss = None
  last_lr = None
  pbar = tqdm(total=total_steps, initial=0, desc='Training',
              unit='step', disable=not is_main_process)

  def _compute_loss(logits, y):
    if single_label:
      return F.cross_entropy(logits, y)
    return F.binary_cross_entropy_with_logits(logits, y)

  def _binary_sensitivity_specificity(y_true_bin, y_pred_bin):
    y_true_bin = y_true_bin.astype(bool)
    y_pred_bin = y_pred_bin.astype(bool)
    tp = np.logical_and(y_true_bin, y_pred_bin).sum()
    tn = np.logical_and(~y_true_bin, ~y_pred_bin).sum()
    fp = np.logical_and(~y_true_bin, y_pred_bin).sum()
    fn = np.logical_and(y_true_bin, ~y_pred_bin).sum()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    return sensitivity, specificity

  def _compute_macro_sensitivity_specificity(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sensitivities, specificities = [], []
    if y_true.ndim == 1:
      classes = np.unique(np.concatenate([y_true, y_pred]))
      for class_id in classes:
        sens, spec = _binary_sensitivity_specificity(y_true == class_id, y_pred == class_id)
        sensitivities.append(sens)
        specificities.append(spec)
    else:
      for class_idx in range(y_true.shape[1]):
        sens, spec = _binary_sensitivity_specificity(y_true[:, class_idx], y_pred[:, class_idx])
        sensitivities.append(sens)
        specificities.append(spec)
    return float(np.nanmean(sensitivities)), float(np.nanmean(specificities))

  def _compute_single_label_metrics(targets, logits):
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    preds = logits.argmax(dim=1).cpu().numpy()
    f1 = f1_score(y_true=targets, y_pred=preds, average='macro')
    acc = accuracy_score(y_true=targets, y_pred=preds)
    sensitivity, specificity = _compute_macro_sensitivity_specificity(targets, preds)
    try:
      auroc = roc_auc_score(y_true=targets, y_score=probs, average='macro', multi_class='ovr')
    except ValueError:
      auroc = float('nan')
    return preds, probs, f1, acc, auroc, sensitivity, specificity

  def _eval_val():
    val_logits_or_preds, val_targets = [], []
    model.eval()
    with torch.inference_mode():
      for batch in val_loader:
        bx, by = (tensor.to(device) for tensor in batch)
        if eval_config.crop_duration is not None:
          batch_size, num_crops, num_channels, channel_size = bx.size()
          bx = bx.reshape(-1, num_channels, channel_size)
        logits = model(bx)
        if eval_config.crop_duration is not None:
          logits = logits.reshape(batch_size, num_crops, eval_config.num_classes)
          logits = logits.mean(dim=1)
        if single_label:
          val_logits_or_preds.append(logits.clone())
        else:
          val_logits_or_preds.append(logits.clone())
        val_targets.append(by.clone())
    model.train()
    targets = torch.cat(val_targets).cpu().numpy()
    if single_label:
      logits = torch.cat(val_logits_or_preds)
      preds, probs, metric, acc, auroc, sensitivity, specificity = _compute_single_label_metrics(targets, logits)
      return {'preds': preds, 'probs': probs}, targets, metric, acc, auroc, sensitivity, specificity
    else:
      preds = torch.cat(val_logits_or_preds).sigmoid().cpu().numpy()
      metric = roc_auc_score(y_true=targets, y_score=preds, average='macro')
      binary_preds = (preds >= 0.5).astype(np.int32)
      sensitivity, specificity = _compute_macro_sensitivity_specificity(targets, binary_preds)
      return preds, targets, metric, None, None, sensitivity, specificity

  def _log_val(epoch_or_step, label, preds, targets, metric, acc, auroc, sensitivity, specificity, new_best):
    if single_label:
      logger.info(f'{label}: {epoch_or_step} '
                  f'{"(*)" if new_best else "   "} '
                  f'val_f1: {metric:.4f} '
                  f'val_acc: {acc:.4f} '
                  f'val_auroc: {auroc:.4f} '
                  f'val_sensitivity: {sensitivity:.4f} '
                  f'val_specificity: {specificity:.4f}')
    else:
      logger.info(f'{label}: {epoch_or_step} '
                  f'{"(*)" if new_best else "   "} '
                  f'val_auc: {metric:.4f} '
                  f'val_sensitivity: {sensitivity:.4f} '
                  f'val_specificity: {specificity:.4f}')

  if eval_config.epochs > 0:
    for epoch in range(eval_config.epochs):
      if is_distributed:
        train_sampler.set_epoch(epoch)
      for x, y in train_loader:
        step_start = time()
        update_learning_rate_(optimizer, next(lr_schedule))
        x, y = x.to(device), y.to(device)
        with auto_mixed_precision:
          logits = model(x)
          loss = _compute_loss(logits, y)
        loss.backward()
        if eval_config.gradient_clip > 0:
          torch.nn.utils.clip_grad_norm_(model.parameters(), eval_config.gradient_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1
        step_time.update(time() - step_start)
        train_loss.update(loss.item())
        if is_main_process:
          current_epoch = global_step / steps_per_epoch
          current_lr = optimizer.param_groups[0]['lr']
          last_loss = train_loss.value
          last_lr = current_lr
          logger.info(f'step: {global_step} '
                      f'epoch: {current_epoch:.4f} '
                      f'train_loss: {train_loss.value:.4f} '
                      f'lr: {current_lr:.6e} '
                      f'step_time: {step_time.value:.4f}')
          pbar.set_postfix(loss=f'{train_loss.value:.4f}', lr=f'{current_lr:.2e}', epoch=f'{current_epoch:.2f}')
          pbar.update(1)
          step_time = AverageMeter()
          train_loss = AverageMeter()
        if is_main_process and global_step % eval_config.checkpoint_interval == 0:
          new_chkpt_path = path.join(args.out, f'chkpt_{global_step}.pt')
          torch.save({
            'model': original_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': dataclasses.asdict(encoder_config),
            'eval_config': dataclasses.asdict(eval_config),
            'step': global_step,
          }, new_chkpt_path)
      val_predictions, val_targets, val_metric, val_acc, val_auroc, val_sensitivity, val_specificity = _eval_val()
      new_best = val_metric > best_val_metric
      if new_best:
        best_val_metric = val_metric
        best_val_predictions = val_predictions
        saved_val_targets = val_targets
        best_epoch_or_step = epoch
        best_chkpt = copy.deepcopy(original_model.state_dict())
      if is_main_process:
        _log_val(epoch + 1, 'epoch', val_predictions, val_targets, val_metric, val_acc, val_auroc,
                 val_sensitivity, val_specificity, new_best)
      if epoch - best_epoch_or_step >= eval_config.early_stopping_patience:
        if is_main_process:
          logging.info(f'stopping training early because validation metric does not improve')
        break
  else:
    def _cycle(dataloader):
      epoch = 0
      while True:
        if is_distributed:
          train_sampler.set_epoch(epoch)
        yield from dataloader
        epoch += 1
    train_dataset_size = len(train_dataset)
    train_iterator = _cycle(train_loader)
    for step in range(eval_config.steps):
      step_start = time()
      update_learning_rate_(optimizer, next(lr_schedule))
      x, y = (tensor.to(device) for tensor in next(train_iterator))
      with auto_mixed_precision:
        logits = model(x)
        loss = _compute_loss(logits, y)
      loss.backward()
      if eval_config.gradient_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), eval_config.gradient_clip)
      optimizer.step()
      optimizer.zero_grad(set_to_none=True)
      step_time.update(time() - step_start)
      train_loss.update(loss.item())
      if is_main_process:
        current_epoch = (step + 1) * eval_config.batch_size / train_dataset_size
        current_lr = optimizer.param_groups[0]['lr']
        last_loss = train_loss.value
        last_lr = current_lr
        logger.info(f'step: {step + 1} '
                    f'epoch: {current_epoch:.4f} '
                    f'train_loss: {train_loss.value:.4f} '
                    f'lr: {current_lr:.6e} '
                    f'step_time: {step_time.value:.4f}')
        pbar.set_postfix(loss=f'{train_loss.value:.4f}', lr=f'{current_lr:.2e}', epoch=f'{current_epoch:.2f}')
        pbar.update(1)
        step_time = AverageMeter()
        train_loss = AverageMeter()
      if (step + 1) % eval_config.checkpoint_interval == 0:
        if is_main_process:
          new_chkpt_path = path.join(args.out, f'chkpt_{step + 1}.pt')
          torch.save({
            'model': original_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': dataclasses.asdict(encoder_config),
            'eval_config': dataclasses.asdict(eval_config),
            'step': step + 1,
          }, new_chkpt_path)
        val_predictions, val_targets, val_metric, val_acc, val_auroc, val_sensitivity, val_specificity = _eval_val()
        new_best = val_metric > best_val_metric
        if new_best:
          best_val_metric = val_metric
          best_val_predictions = val_predictions
          saved_val_targets = val_targets
          best_epoch_or_step = step
          best_chkpt = copy.deepcopy(original_model.state_dict())
        if is_main_process:
          _log_val(step + 1, 'step', val_predictions, val_targets, val_metric, val_acc, val_auroc,
                   val_sensitivity, val_specificity, new_best)
        if step - best_epoch_or_step >= eval_config.early_stopping_patience:
          if is_main_process:
            logging.info('stopping training early because validation metric does not improve')
          break

  # -------------------------------------------------------------------------
  # Save best checkpoint and run test evaluation (main process only)
  # -------------------------------------------------------------------------
  if is_main_process:
    torch.save({
      'model': best_chkpt,
      'config': dataclasses.asdict(encoder_config),
      'eval_config': dataclasses.asdict(eval_config),
      'preprocess': {'mean': torch.from_numpy(mean.squeeze()),
                     'std': torch.from_numpy(std.squeeze())},
      'task': task_name
    }, path.join(args.out, f'{task_name}_best_chkpt.pt'))

  if is_main_process:
    logger.info('loading best model checkpoint')
    original_model.load_state_dict(best_chkpt)

    test_logits_or_preds, test_targets = [], []
    original_model.eval()
    with torch.inference_mode():
      for batch in test_loader:
        bx, by = (tensor.to(device) for tensor in batch)
        if eval_config.crop_duration is not None:
          batch_size, num_crops, num_channels, channel_size = bx.size()
          bx = bx.reshape(-1, num_channels, channel_size)
        logits = original_model(bx)
        if eval_config.crop_duration is not None:
          logits = logits.reshape(batch_size, num_crops, eval_config.num_classes)
          logits = logits.mean(dim=1)
        test_logits_or_preds.append(logits.clone())
        test_targets.append(by.clone())

    test_targets = torch.cat(test_targets).cpu().numpy()
    if single_label:
      test_logits = torch.cat(test_logits_or_preds)
      test_predictions, test_probabilities, test_f1, test_acc, test_auroc, test_sensitivity, test_specificity = _compute_single_label_metrics(
        targets=test_targets, logits=test_logits)
      val_probabilities = best_val_predictions['probs']
      val_sensitivity, val_specificity = _compute_macro_sensitivity_specificity(
        saved_val_targets, best_val_predictions['preds'])
      try:
        val_auroc = roc_auc_score(y_true=saved_val_targets, y_score=val_probabilities, average='macro',
                                  multi_class='ovr')
      except ValueError:
        val_auroc = float('nan')
      logger.info(f'test_f1 {test_f1:.4f}  test_acc {test_acc:.4f}  test_auroc {test_auroc:.4f}  '
                  f'test_sensitivity {test_sensitivity:.4f}  test_specificity {test_specificity:.4f}')
      eval_results = {
        'task': task_name,
        'dataset_type': args.dataset_type,
        'single_label': bool(single_label),
        'best_val_metric': float(best_val_metric),
        'best_epoch_or_step': int(best_epoch_or_step),
        'val_f1': float(best_val_metric),
        'val_auroc': float(val_auroc),
        'val_sensitivity': float(val_sensitivity),
        'val_specificity': float(val_specificity),
        'test_f1': float(test_f1),
        'test_acc': float(test_acc),
        'test_auroc': float(test_auroc),
        'test_sensitivity': float(test_sensitivity),
        'test_specificity': float(test_specificity),
        'timestamp': datetime.now().isoformat(),
        'out_dir': args.out,
        'config_path': args.config,
        'encoder_path': args.encoder,
      }
    else:
      test_predictions = torch.cat(test_logits_or_preds).sigmoid().cpu().numpy()
      test_auc = roc_auc_score(y_true=test_targets, y_score=test_predictions, average='macro')
      test_binary_predictions = (test_predictions >= 0.5).astype(np.int32)
      test_sensitivity, test_specificity = _compute_macro_sensitivity_specificity(
        test_targets, test_binary_predictions)
      logger.info(f'test_auc {test_auc:.4f}  test_sensitivity {test_sensitivity:.4f}  '
                  f'test_specificity {test_specificity:.4f}')
      eval_results = {
        'task': task_name,
        'dataset_type': args.dataset_type,
        'single_label': bool(single_label),
        'best_val_metric': float(best_val_metric),
        'best_epoch_or_step': int(best_epoch_or_step),
        'val_auc': float(best_val_metric),
        'val_sensitivity': float(val_sensitivity),
        'val_specificity': float(val_specificity),
        'test_auc': float(test_auc),
        'test_sensitivity': float(test_sensitivity),
        'test_specificity': float(test_specificity),
        'timestamp': datetime.now().isoformat(),
        'out_dir': args.out,
        'config_path': args.config,
        'encoder_path': args.encoder,
      }

    json_result_path = path.join(args.out, f'{task_name}_eval_results.json')
    with open(json_result_path, 'w', encoding='utf-8') as f:
      json.dump(eval_results, f, indent=2, ensure_ascii=False)
    logger.info(f'saved eval results json to {json_result_path}')

    prediction_dump_path = path.join(args.out, f'{task_name}_predictions.npz')
    if single_label:
      np.savez(prediction_dump_path,
               val_targets=saved_val_targets,
               val_predictions=best_val_predictions['preds'],
               val_probabilities=best_val_predictions['probs'],
               test_targets=test_targets,
               test_predictions=test_predictions,
               test_probabilities=test_probabilities)
    else:
      np.savez(prediction_dump_path,
               val_targets=saved_val_targets,
               val_predictions=best_val_predictions,
               test_targets=test_targets,
               test_predictions=test_predictions)

  pbar.close()
  if is_main_process:
    total_time = time() - train_start_time
    h, rem = divmod(int(total_time), 3600)
    m, s = divmod(rem, 60)
    lines = [
      '=' * 50,
      'Training Complete',
      f'  Total steps   : {global_step}',
      f'  Total time    : {h:02d}h {m:02d}m {s:02d}s ({total_time:.1f}s)',
      f'  Best val step/epoch: {best_epoch_or_step}',
      f'  Best val metric : {best_val_metric:.4f}',
    ]
    if single_label:
      lines.append(f'  Test F1       : {test_f1:.4f}  Test Acc: {test_acc:.4f}  Test AUROC: {test_auroc:.4f}')
      lines.append(f'  Test Sens/Spec: {test_sensitivity:.4f} / {test_specificity:.4f}')
    else:
      lines.append(f'  Test AUC      : {test_auc:.4f}')
      lines.append(f'  Test Sens/Spec: {test_sensitivity:.4f} / {test_specificity:.4f}')
    lines.append('=' * 50)
    logger.info('\n' + '\n'.join(lines))

  if is_distributed:
    dist.destroy_process_group()


class PreprocessECG:
  def __init__(self, channel_size=None, remove_baseline_wander=False):
    self.channel_size = channel_size
    self.remove_baseline_wander = remove_baseline_wander

  def __call__(self, x):  # x: (num_channels, channel_size)
    num_channels, channel_size = x.shape
    if self.remove_baseline_wander:
      x = transforms.highpass_filter(x, fs=PTB_XL.sampling_frequency)
    if self.channel_size is not None and self.channel_size != channel_size:
      x = transforms.resample(x, self.channel_size)
    return x


class PreprocessHAR:
  def __init__(self, *, mean_std, channel_order, transpose_input=False):
    self.mean, self.std = mean_std
    self.channel_order = channel_order
    self.transpose_input = transpose_input

  def __call__(self, x):
    x = np.array(x, dtype=np.float32, copy=True)
    x = np.squeeze(x)
    if x.ndim == 1:
      x = x[None, :]
    elif x.ndim != 2:
      raise ValueError(f'Expected (C, T) or (T, C) sample, got {x.shape}')

    expected_channels = len(self.channel_order)
    if self.transpose_input or (x.shape[0] != expected_channels and x.shape[1] == expected_channels):
      x = x.T

    mean = np.asarray(self.mean, dtype=np.float32).reshape(-1, 1)
    std = np.asarray(self.std, dtype=np.float32).reshape(-1, 1)
    x = (x - mean) / (std + 1e-8)
    x.clip(-5, 5, out=x)
    x = x[self.channel_order]
    return x


class TrainTransform:
  def __init__(self, crop_size=None):
    self.crop_size = crop_size

  def __call__(self, x):  # x: (num_channels, channel_size)
    if self.crop_size is not None:
      x = transforms.random_crop(x, self.crop_size)
    x = torch.from_numpy(x).float()
    return x


class EvalTransform:
  def __init__(self, crop_size=None, crop_stride=None):
    self.crop_size = crop_size
    self.crop_stride = crop_stride or crop_size

  def __call__(self, x):  # x: (num_channels, channel_size)
    if self.crop_size is not None:
      x = strided_crops(x, self.crop_size, self.crop_stride)
    x = torch.from_numpy(x).float()
    return x


def strided_crops(x, size, stride):  # x: (num_channels, channel_size)
  num_channels, channel_size = x.shape
  crop_starts = range(0, channel_size - size + 1, stride)
  num_crops = len(crop_starts)
  x_ = np.empty((num_crops, num_channels, size), dtype=x.dtype)
  for i, start in enumerate(crop_starts):
    x_[i] = x[:, start:start + size]
  return x_


if __name__ == '__main__':
  main()
