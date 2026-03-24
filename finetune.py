import argparse
import copy
import dataclasses
import logging
import logging.config
import os
import pprint
from contextlib import nullcontext
from os import path, makedirs
from time import time

import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import configs
from data import transforms, utils as datautils
from data.datasets import PTB_XL, Capture24
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
FOLDS = tuple(range(1, 11))

VAL_RATIO = 0.2
VAL_SEED = 42

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default=None, help='path to data directory (overrides dataset.data_dir in config)')
parser.add_argument('--encoder', required=True, help='path to checkpoint or config file')
parser.add_argument('--out', default='eval', help='output directory')
parser.add_argument('--config', default=None, help='path to config file or config name (default: linear for ptbxl, har_linear for capture24)')
parser.add_argument('--dump', help='path to dump file (.npy) with raw signals (overrides dataset.dump in config)')
parser.add_argument('--amp', default='float32', choices=['bfloat16', 'float32'], help='automated mixed precision')
parser.add_argument('--dataset', choices=['ptbxl', 'capture24'], default=None,
                    help='dataset type (auto-detected from data_dir if not specified)')
# PTB-XL specific args
parser.add_argument('--task', choices=TASKS, default='all', help='task type (PTB-XL only)')
parser.add_argument('--val-fold', choices=FOLDS, type=int, default=9, help='validation fold (PTB-XL only)')
parser.add_argument('--test-fold', choices=FOLDS, type=int, default=10, help='test fold (PTB-XL only)')
args = parser.parse_args()


def main():
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
  logger = logging.getLogger('app')
  if not is_main_process:
    logger.setLevel(logging.CRITICAL)

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

  # Auto-detect dataset from data_dir if not explicitly specified
  dataset = args.dataset
  if dataset is None:
    dataset = 'capture24' if 'capture24' in (args.data_dir or '').lower() else 'ptbxl'
  is_capture24 = (dataset == 'capture24')

  default_config = 'har_linear' if is_capture24 else 'linear'
  config_name = args.config or default_config
  if not path.isfile(config_name):
    config_file = path.join(path.dirname(configs.eval.__file__), f'{config_name}.yaml')
    if not path.isfile(config_file):
      raise ValueError(f'Failed to read configuration file {config_name}')
    config_name = config_file

  eval_config_dict = configs.load_config_file(config_name)
  if is_main_process:
    logger.debug(f'loading configuration file from {config_name}\n'
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
    encoder_config = configs.pretrain.Config(**encoder_config_dict)
    model_state_dict = None
  else:
    if is_main_process:
      logger.debug(f'loading encoder checkpoint from {args.encoder}')
    chkpt = torch.load(args.encoder, map_location='cpu')
    encoder_config_dict = chkpt['config']
    encoder_config = configs.pretrain.Config(**encoder_config_dict)
    if 'eval_config' in chkpt:  # continue fine-tuning the weights
      model_state_dict = chkpt['model']
    else:  # extract target encoder's weights from the checkpoint
      model_state_dict = {'encoder.' + k.removeprefix('target_encoder.'): v
                          for k, v in chkpt['model'].items()
                          if k.startswith('target_encoder.')}

  # -------------------------------------------------------------------------
  # Dataset loading (branched by dataset type)
  # -------------------------------------------------------------------------
  if is_capture24:
    train_dump = args.dump or dataset_cfg.get('dump') or f'{data_dir}_train.npy'
    test_dump = f'{data_dir}_test.npy'
    for f in (train_dump, test_dump):
      if not path.isfile(f):
        raise ValueError(f'Failed to find .npy data file: {f}. '
                         f'Run scripts/dump_data.py to generate split files.')

    if is_main_process:
      logger.debug(f'loading labels from {data_dir}')
    train_labels, test_labels = Capture24.load_labels(data_dir)

    if is_main_process:
      logger.debug(f'loading train data from {train_dump}')
    x_train_all = np.load(train_dump)
    if is_main_process:
      logger.debug(f'loading test data from {test_dump}')
    x_test = np.load(test_dump)

    # split train into train/val with a fixed random seed
    rng = np.random.RandomState(VAL_SEED)
    n_train_all = len(x_train_all)
    indices = np.arange(n_train_all)
    rng.shuffle(indices)
    num_val = int(n_train_all * VAL_RATIO)
    val_indices = indices[:num_val]
    train_indices = indices[num_val:]

    num_classes = int(max(train_labels.max(), test_labels.max()) + 1)
    if is_main_process:
      logger.debug(f'train={len(train_indices)}, val={len(val_indices)}, '
                   f'test={len(x_test)}, num_classes={num_classes}')

    # normalize using training statistics (apply to both splits)
    mean = np.mean(x_train_all[train_indices], axis=(0, 1), keepdims=True, dtype=np.float32)
    std = np.std(x_train_all[train_indices], axis=(0, 1), keepdims=True, dtype=np.float32)
    transforms.normalize_(x_train_all, mean_std=(mean, std))
    x_train_all.clip(-5, 5, out=x_train_all)
    transforms.normalize_(x_test, mean_std=(mean, std))
    x_test.clip(-5, 5, out=x_test)

    # ensure matching channels
    channel_order = get_channel_order(Capture24.channels, encoder_config.channels)
    x_train_all = x_train_all[:, :, channel_order]
    x_test = x_test[:, :, channel_order]

    y_train_all = torch.from_numpy(train_labels).long()
    y_test = torch.from_numpy(test_labels).long()
    x_train = x_train_all[train_indices]
    y_train = y_train_all[train_indices]
    x_val = x_train_all[val_indices]
    y_val = y_train_all[val_indices]

    task_name = 'har'
    single_label = True

  else:  # PTB-XL
    dump_file = args.dump or dataset_cfg.get('dump') or f'{data_dir}.npy'
    if not path.isfile(dump_file):
      raise ValueError(f'Failed to find .npy data file. Attempted location: {dump_file}. '
                       f'Use --dump or dataset.dump in the eval config yaml to specify location.')

    ptb_xl_task = args.task
    single_label = False
    if args.task == 'ST-MEM':
      ptb_xl_task = 'superdiagnostic'
      single_label = True

    if is_main_process:
      logger.debug(f'setting up labels for task `{args.task}`')
    labels_df = PTB_XL.load_raw_labels(data_dir)
    labels_df = PTB_XL.compute_label_aggregations(labels_df, data_dir, ptb_xl_task)

    if is_main_process:
      logger.debug(f'loading data from {dump_file}')
    channel_size = PTB_XL.record_duration * encoder_config.sampling_frequency

    x = datautils.load_data_dump(
      dump_file=dump_file,
      transform=PreprocessECG(
        channel_size=channel_size,
        remove_baseline_wander=False),
      processes=num_cpus)

    x, labels_df, y, _ = PTB_XL.select_data(x, labels_df, ptb_xl_task, min_samples=0)
    if single_label:
      single_label_mask = y.sum(axis=1) == 1
      x, labels_df, y = x[single_label_mask], labels_df[single_label_mask], y[single_label_mask]
    y = torch.from_numpy(y).float()
    num_classes = y.shape[1]

    val_mask = (labels_df.strat_fold == args.val_fold).to_numpy()
    test_mask = (labels_df.strat_fold == args.test_fold).to_numpy()
    train_mask = ~(val_mask | test_mask)

    # normalize data (train stats only; test not normalized separately here)
    mean = np.mean(x[train_mask], axis=(0, 1), keepdims=True, dtype=np.float32)
    std = np.std(x[train_mask], axis=(0, 1), keepdims=True, dtype=np.float32)
    transforms.normalize_(x, mean_std=(mean, std))
    x.clip(-5, 5, out=x)

    # ensure matching channels
    channel_order = datautils.get_channel_order(PTB_XL.channels, encoder_config.channels)
    x = x[:, :, channel_order]

    x_train, y_train = x[train_mask], y[train_mask]
    x_val, y_val = x[val_mask], y[val_mask]
    x_test, y_test = x[test_mask], y[test_mask]

    task_name = args.task

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
  num_workers = max(1, num_cpus // world_size)

  train_dataset = TensorDataset(
    data=x_train,
    labels=y_train,
    transform=TrainTransform(crop_size=crop_size))

  train_sampler = DistributedSampler(
    train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
  ) if is_distributed else None

  train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=local_batch_size,
    sampler=train_sampler,
    shuffle=(train_sampler is None),
    drop_last=(train_sampler is None),
    num_workers=num_workers)

  val_loader = DataLoader(
    dataset=TensorDataset(
      data=x_val,
      labels=y_val,
      transform=EvalTransform(
        crop_size=crop_size,
        crop_stride=crop_stride)),
    batch_size=eval_config.batch_size,
    num_workers=num_workers)
  test_loader = DataLoader(
    dataset=TensorDataset(
      data=x_test,
      labels=y_test,
      transform=EvalTransform(
        crop_size=crop_size,
        crop_stride=crop_stride)),
    batch_size=eval_config.batch_size,
    num_workers=num_workers)

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
  # Training loop (shared structure, branched on metric)
  # -------------------------------------------------------------------------
  step_time = AverageMeter()
  train_loss = AverageMeter()
  best_val_metric = float('-inf')
  best_val_predictions, saved_val_targets = None, None
  best_epoch_or_step = None
  best_chkpt = None
  prev_chkpt_path = None
  global_step = 0

  def _compute_loss(logits, y):
    if single_label or is_capture24:
      return F.cross_entropy(logits, y)
    return F.binary_cross_entropy_with_logits(logits, y)

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
        if is_capture24:
          val_logits_or_preds.append(logits.argmax(dim=1).clone())
        else:
          val_logits_or_preds.append(logits.clone())
        val_targets.append(by.clone())
    model.train()
    targets = torch.cat(val_targets).cpu().numpy()
    if is_capture24:
      preds = torch.cat(val_logits_or_preds).cpu().numpy()
      metric = f1_score(y_true=targets, y_pred=preds, average='macro')
      acc = accuracy_score(y_true=targets, y_pred=preds)
      return preds, targets, metric, acc
    else:
      if single_label:
        preds = torch.cat(val_logits_or_preds).softmax(dim=1).cpu().numpy()
      else:
        preds = torch.cat(val_logits_or_preds).sigmoid().cpu().numpy()
      metric = roc_auc_score(y_true=targets, y_score=preds, average='macro')
      return preds, targets, metric, None

  def _log_val(epoch_or_step, label, preds, targets, metric, acc, new_best):
    if is_capture24:
      logger.info(f'{label}: {epoch_or_step} '
                  f'{"(*)" if new_best else "   "} '
                  f'val_f1: {metric:.4f} '
                  f'val_acc: {acc:.4f}')
    else:
      logger.info(f'{label}: {epoch_or_step} '
                  f'{"(*)" if new_best else "   "} '
                  f'val_auc: {metric:.4f}')

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
          logger.info(f'step: {global_step} '
                      f'epoch: {current_epoch:.4f} '
                      f'train_loss: {train_loss.value:.4f} '
                      f'step_time: {step_time.value:.4f}')
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
          if prev_chkpt_path is not None and path.exists(prev_chkpt_path):
            os.remove(prev_chkpt_path)
          prev_chkpt_path = new_chkpt_path
      val_predictions, val_targets, val_metric, val_acc = _eval_val()
      new_best = val_metric > best_val_metric
      if new_best:
        best_val_metric = val_metric
        best_val_predictions = val_predictions
        saved_val_targets = val_targets
        best_epoch_or_step = epoch
        best_chkpt = copy.deepcopy(original_model.state_dict())
      if is_main_process:
        _log_val(epoch + 1, 'epoch', val_predictions, val_targets, val_metric, val_acc, new_best)
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
        logger.info(f'step: {step + 1} '
                    f'epoch: {current_epoch:.4f} '
                    f'train_loss: {train_loss.value:.4f} '
                    f'step_time: {step_time.value:.4f}')
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
          if prev_chkpt_path is not None and path.exists(prev_chkpt_path):
            os.remove(prev_chkpt_path)
          prev_chkpt_path = new_chkpt_path
        val_predictions, val_targets, val_metric, val_acc = _eval_val()
        new_best = val_metric > best_val_metric
        if new_best:
          best_val_metric = val_metric
          best_val_predictions = val_predictions
          saved_val_targets = val_targets
          best_epoch_or_step = step
          best_chkpt = copy.deepcopy(original_model.state_dict())
        if is_main_process:
          _log_val(step + 1, 'step', val_predictions, val_targets, val_metric, val_acc, new_best)
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
        if is_capture24:
          test_logits_or_preds.append(logits.argmax(dim=1).clone())
        else:
          test_logits_or_preds.append(logits.clone())
        test_targets.append(by.clone())

    test_targets = torch.cat(test_targets).cpu().numpy()
    if is_capture24:
      test_predictions = torch.cat(test_logits_or_preds).cpu().numpy()
      test_f1 = f1_score(y_true=test_targets, y_pred=test_predictions, average='macro')
      test_acc = accuracy_score(y_true=test_targets, y_pred=test_predictions)
      logger.info(f'test_f1 {test_f1:.4f}  test_acc {test_acc:.4f}')
    else:
      if single_label:
        test_predictions = torch.cat(test_logits_or_preds).softmax(dim=1).cpu().numpy()
      else:
        test_predictions = torch.cat(test_logits_or_preds).sigmoid().cpu().numpy()
      test_auc = roc_auc_score(y_true=test_targets, y_score=test_predictions, average='macro')
      logger.info(f'test_auc {test_auc:.4f}')

    np.savez(path.join(args.out, f'{task_name}_predictions.npz'),
             val_targets=saved_val_targets, val_predictions=best_val_predictions,
             test_targets=test_targets, test_predictions=test_predictions)

  if is_distributed:
    dist.destroy_process_group()


class PreprocessECG:
  def __init__(self, channel_size=None, remove_baseline_wander=False):
    self.channel_size = channel_size
    self.remove_baseline_wander = remove_baseline_wander

  def __call__(self, x):
    channel_size, num_channels = x.shape
    if self.remove_baseline_wander:
      x = transforms.highpass_filter(x, fs=PTB_XL.sampling_frequency)
    if self.channel_size is not None and self.channel_size != channel_size:
      x = transforms.resample(x, self.channel_size)
    return x


class TrainTransform:
  def __init__(self, crop_size=None):
    self.crop_size = crop_size

  def __call__(self, x):
    if self.crop_size is not None:
      x = transforms.random_crop(x, self.crop_size)
    x = x.transpose()  # channels first
    x = torch.from_numpy(x).float()
    return x


class EvalTransform:
  def __init__(self, crop_size=None, crop_stride=None):
    self.crop_size = crop_size
    self.crop_stride = crop_stride or crop_size

  def __call__(self, x):
    if self.crop_size is not None:
      x = strided_crops(x, self.crop_size, self.crop_stride)
      x = np.swapaxes(x, 1, 2)  # channels first
    else:
      x = x.transpose()  # channels first
    x = torch.from_numpy(x).float()
    return x


def strided_crops(x, size, stride):  # x: (channel_size, num_channels)
  channel_size, num_channels = x.shape
  crop_starts = range(0, channel_size - size + 1, stride)
  num_crops = len(crop_starts)
  x_ = np.empty((num_crops, size, num_channels), dtype=x.dtype)
  for i, start in enumerate(crop_starts):
    x_[i] = x[start:start + size]
  return x_


if __name__ == '__main__':
  main()
