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
from sklearn.metrics import f1_score, accuracy_score
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import configs
from data import transforms, utils as datautils
from data.datasets import Capture24
from data.utils import TensorDataset, get_channel_order
from models import create_encoder, EncoderClassifier
from utils.monitoring import AverageMeter, get_memory_usage, get_cpu_count
from utils.schedules import update_learning_rate_, cosine_schedule

VAL_RATIO = 0.2
VAL_SEED = 42

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default=None, help='path to Capture-24 data directory (overrides dataset.data_dir in config)')
parser.add_argument('--encoder', required=True, help='path to checkpoint or config file')
parser.add_argument('--out', default='eval_har', help='output directory')
parser.add_argument('--config', default='har_linear', help='path to config file or config name')
parser.add_argument('--dump', help='path to dump file (.npy) with signals (overrides dataset.dump in config)')
parser.add_argument('--amp', default='float32', choices=['bfloat16', 'float32'], help='automated mixed precision')
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

  if args.amp == 'float32' or not using_cuda:
    if is_main_process:
      logger.debug('using float32 precision')
    auto_mixed_precision = nullcontext()
  elif args.amp == 'bfloat16':
    if is_main_process:
      logger.debug('using bfloat16 with AMP')
    auto_mixed_precision = torch.cuda.amp.autocast(dtype=torch.bfloat16)
  else:
    raise ValueError('Failed to choose floating-point format.')

  if not path.isfile(args.config):
    config_file = path.join(path.dirname(configs.eval.__file__), f'{args.config}.yaml')
    if not path.isfile(config_file):
      raise ValueError(f'Failed to read configuration file {args.config}')
    args.config = config_file

  eval_config_dict = configs.load_config_file(args.config)
  dataset_cfg = eval_config_dict.pop('dataset', None) or {}
  if is_main_process:
    logger.debug(f'loading configuration file from {args.config}\n'
                 f'{pprint.pformat(eval_config_dict, compact=True, sort_dicts=False, width=120)}')

  # resolve data_dir and dump_file from CLI args or config yaml
  data_dir = args.data_dir or dataset_cfg.get('data_dir')
  if not data_dir:
    raise ValueError('data_dir must be specified via --data-dir or dataset.data_dir in the eval config yaml')
  dump_file = args.dump or dataset_cfg.get('dump') or f'{data_dir}.npy'
  if not path.isfile(dump_file):
    raise ValueError(f'Failed to find .npy data file. Attempted location: {dump_file}. '
                     f'Use --dump or dataset.dump in the eval config yaml to specify location.')

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
    if 'eval_config' in chkpt:
      model_state_dict = chkpt['model']
    else:
      model_state_dict = {'encoder.' + k.removeprefix('target_encoder.'): v
                          for k, v in chkpt['model'].items()
                          if k.startswith('target_encoder.')}

  # load labels and split info (saved by dump_data.py)
  if is_main_process:
    logger.debug(f'loading labels from {data_dir}')
  labels, splits = Capture24.load_labels(data_dir)

  # load data  (shape: N, channel_size, num_channels) -- channels last
  if is_main_process:
    logger.debug(f'loading data from {dump_file}')
  x = np.load(dump_file)

  train_mask = splits == 'train'
  test_mask = splits == 'test'

  # split train into train/val with a fixed random seed
  rng = np.random.RandomState(VAL_SEED)
  train_indices = np.where(train_mask)[0]
  rng.shuffle(train_indices)
  num_val = int(len(train_indices) * VAL_RATIO)
  val_indices = train_indices[:num_val]
  train_indices = train_indices[num_val:]

  num_classes = int(labels.max() + 1)
  if is_main_process:
    logger.debug(f'train={len(train_indices)}, val={len(val_indices)}, '
                 f'test={test_mask.sum()}, num_classes={num_classes}')

  # normalize using training statistics
  mean = np.mean(x[train_indices], axis=(0, 1), keepdims=True, dtype=np.float32)
  std = np.std(x[train_indices], axis=(0, 1), keepdims=True, dtype=np.float32)
  transforms.normalize_(x, mean_std=(mean, std))
  x.clip(-5, 5, out=x)

  # ensure matching channels
  channel_order = get_channel_order(Capture24.channels, encoder_config.channels)
  x = x[:, :, channel_order]

  y = torch.from_numpy(labels).long()
  x_train, y_train = x[train_indices], y[train_indices]
  x_val, y_val = x[val_indices], y[val_indices]
  x_test, y_test = x[test_mask], y[test_mask]

  if is_main_process:
    logger.debug(f'{get_memory_usage() / 1024 ** 3:,.2f}GB memory used after loading data')

  # initialize configs
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
    transform=TransformSignal(crop_size=crop_size))

  train_sampler = DistributedSampler(
    train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
  ) if is_distributed else None

  train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=local_batch_size,
    sampler=train_sampler,
    shuffle=(train_sampler is None),
    drop_last=True,
    num_workers=num_workers)

  val_loader = DataLoader(
    dataset=TensorDataset(
      data=x_val,
      labels=y_val,
      transform=EvalTransformSignal(
        crop_size=crop_size,
        crop_stride=crop_stride)),
    batch_size=eval_config.batch_size,
    num_workers=num_workers)
  test_loader = DataLoader(
    dataset=TensorDataset(
      data=x_test,
      labels=y_test,
      transform=EvalTransformSignal(
        crop_size=crop_size,
        crop_stride=crop_stride)),
    batch_size=eval_config.batch_size,
    num_workers=num_workers)

  steps_per_epoch = len(train_loader)
  total_steps = eval_config.epochs * steps_per_epoch

  # setup hyperparameter schedules
  lr_schedule = cosine_schedule(
    total_steps=total_steps,
    start_value=eval_config.learning_rate,
    final_value=eval_config.final_learning_rate,
    warmup_steps=eval_config.learning_rate_warmup_steps,
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

  step_time = AverageMeter()
  train_loss = AverageMeter()
  best_val_f1 = float('-inf')
  best_val_predictions, saved_val_targets = None, None
  best_epoch, best_chkpt = None, None

  for epoch in range(eval_config.epochs):
    if is_distributed:
      train_sampler.set_epoch(epoch)
    # train
    for x, y in train_loader:
      step_start = time()
      update_learning_rate_(optimizer, next(lr_schedule))
      x, y = x.to(device), y.to(device)
      with auto_mixed_precision:
        logits = model(x)
        loss = F.cross_entropy(logits, y)
      loss.backward()
      if eval_config.gradient_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), eval_config.gradient_clip)
      optimizer.step()
      optimizer.zero_grad(set_to_none=True)
      step_time.update(time() - step_start)
      train_loss.update(loss.item())
    # evaluate after each epoch
    val_preds, val_targets = [], []
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
        val_preds.append(logits.argmax(dim=1).clone())
        val_targets.append(by.clone())
    model.train()
    val_preds = torch.cat(val_preds).cpu().numpy()
    val_targets = torch.cat(val_targets).cpu().numpy()
    val_f1 = f1_score(y_true=val_targets, y_pred=val_preds, average='macro')
    val_acc = accuracy_score(y_true=val_targets, y_pred=val_preds)
    new_best = val_f1 > best_val_f1
    if new_best:
      best_val_f1 = val_f1
      best_val_predictions = val_preds
      saved_val_targets = val_targets
      best_epoch = epoch
      best_chkpt = copy.deepcopy(original_model.state_dict())
    if is_main_process:
      logger.info(f'[epoch {epoch + 1:04d}] '
                  f'{"(*)" if new_best else "   "} '
                  f'step_time {step_time.value:.4f} '
                  f'train_loss {train_loss.value:.4f} '
                  f'val_f1 {val_f1:.4f} '
                  f'val_acc {val_acc:.4f}')
    step_time = AverageMeter()
    train_loss = AverageMeter()
    if epoch - best_epoch >= eval_config.early_stopping_patience:
      if is_main_process:
        logging.info('stopping training early because validation F1 does not improve')
      break

  if is_main_process:
    torch.save({
      'model': best_chkpt,
      'config': dataclasses.asdict(encoder_config),
      'eval_config': dataclasses.asdict(eval_config),
      'preprocess': {'mean': torch.from_numpy(mean.squeeze()),
                     'std': torch.from_numpy(std.squeeze())},
      'task': 'har'
    }, path.join(args.out, 'har_best_chkpt.pt'))

  # test model
  if is_main_process:
    logger.info('loading best model checkpoint')
    original_model.load_state_dict(best_chkpt)

    test_preds, test_targets = [], []
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
        test_preds.append(logits.argmax(dim=1).clone())
        test_targets.append(by.clone())
    test_preds = torch.cat(test_preds).cpu().numpy()
    test_targets = torch.cat(test_targets).cpu().numpy()
    test_f1 = f1_score(y_true=test_targets, y_pred=test_preds, average='macro')
    test_acc = accuracy_score(y_true=test_targets, y_pred=test_preds)
    logger.info(f'test_f1 {test_f1:.4f}  test_acc {test_acc:.4f}')
    np.savez(path.join(args.out, 'har_predictions.npz'),
             val_targets=saved_val_targets, val_predictions=best_val_predictions,
             test_targets=test_targets, test_predictions=test_preds)

  if is_distributed:
    dist.destroy_process_group()


class TransformSignal:
  def __init__(self, crop_size=None):
    self.crop_size = crop_size

  def __call__(self, x):
    if self.crop_size is not None:
      x = transforms.random_crop(x, self.crop_size)
    x = x.transpose()  # channels first
    x = torch.from_numpy(x).float()
    return x


class EvalTransformSignal:
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
