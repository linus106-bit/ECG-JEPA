import argparse
import dataclasses
import logging.config
import os
import pprint
import queue
import threading
from contextlib import nullcontext
from datetime import datetime
from os import path, makedirs
from pathlib import Path
from time import time

from tqdm import tqdm

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import configs
from data import transforms, utils as datautils
from data.datasets import (
  DATASETS,
  CODE15,
  StPetersburg,
  PTB_XL
)
from data.masks import MaskCollator
from data.utils import (
  TensorDataset,
  VariableTensorDataset,
  DatasetRouter,
  load_hf_dataset,
  load_hf_variable_dataset,
)
from models import JEPA
from models.LeJEPA import LeJEPA
from utils.monitoring import (
  AverageMeter,
  get_cpu_count,
  get_memory_usage
)
from utils.schedules import (
  linear_schedule,
  cosine_schedule,
  update_weight_decay_,
  update_learning_rate_
)

parser = argparse.ArgumentParser()
parser.add_argument('--out', default=None, help='output directory (default: run.out_dir in config yaml)')
parser.add_argument('--config', default='ViTS_mimic', help='path to config file or config name')
parser.add_argument('--chkpt', default=None, help='resume training from model checkpoint (default: run.checkpoint in config yaml)')
parser.add_argument('--amp', default=None, choices=['bfloat16', 'float32'], help='precision (default: run.amp in config yaml)')
parser.add_argument('--compile', action='store_true', help='compile model (or set run.compile: true in config yaml)')
parser.add_argument('--jepa_mode', default=None, choices=['ijepa', 'lejepa'], help='JEPA mode (default: from config)')
parser.add_argument('--sigreg_lambda', default=None, type=float, help='SIGReg lambda for LeJEPA (default: from config)')
args = parser.parse_args()

# NOTE: we update means and standard deviations of some datasets
#  because we use their preprocessed version instead of the original.
#  The preprocessed versions have had their baseline wander removed.
#  This was essential to maintain training stability.
CODE15.mean = [0.000] * len(CODE15.channels)
CODE15.std = [0.488, 0.450, 0.437, 0.416, 0.405, 0.370,
              0.548, 0.639, 0.719, 0.695, 0.676, 0.639]
StPetersburg.mean = [0.000] * len(StPetersburg.channels)
StPetersburg.std = [0.132, 0.370, 0.353, 0.215, 0.191, 0.356,
                    0.234, 0.320, 0.328, 0.290, 0.317, 0.337]
# NOTE: we compute mean and standard deviation of ptb-xl over the train folds (1-8).
#  We only use these folds during pre-training.
PTB_XL.mean = [-0.002, -0.002, 0.000, 0.002, -0.001, -0.001,
               0.000, -0.001, -0.002, -0.001, -0.001, -0.001]
PTB_XL.std = [0.191, 0.166, 0.173, 0.142, 0.149, 0.147,
              0.235, 0.338, 0.335, 0.299, 0.294, 0.242]


def main():
  # Load config YAML and extract run: section early (needed before distributed setup)
  if not path.isfile(args.config):
    raise ValueError(f'Config file not found: {args.config}')
  yaml_dict = configs.load_config_file(args.config)
  run_config = yaml_dict.pop('run', {})
  if args.out is None:
    args.out = run_config.get('out_dir', path.join('pretrain', Path(args.config).stem))
  if args.amp is None:
    args.amp = run_config.get('amp', 'float32')
  args.compile = args.compile or run_config.get('compile', False)
  if args.chkpt is None and run_config.get('checkpoint'):
    args.chkpt = run_config['checkpoint']

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

  if args.chkpt:
    if is_main_process:
      logger.debug(f'resuming from checkpoint {args.chkpt}')
    chkpt = torch.load(args.chkpt, map_location=device)
    config = configs.pretrain.Config(**chkpt['config'])
  else:
    config = configs.pretrain.Config(**yaml_dict)
    if is_main_process:
      logger.debug(f'loading configuration file from {args.config}:\n'
                   f'{pprint.pformat(yaml_dict, compact=True, sort_dicts=False, width=120)}')
    chkpt = None

  # command-line overrides for LeJEPA
  if args.jepa_mode is not None:
    config.jepa_mode = args.jepa_mode
  if args.sigreg_lambda is not None:
    config.sigreg_lambda = args.sigreg_lambda
  if is_main_process:
    logger.debug(f'JEPA mode: {config.jepa_mode}')

  for dataset_name, dataset_info in config.datasets.items():
    if dataset_name not in DATASETS:
      raise ValueError(f'Unknown dataset {dataset_name}. '
                       f'Available datasets are {list(DATASETS)}')
    dataset_path = dataset_info['path']
    if not path.isdir(dataset_path) and not path.isfile(dataset_path):
      raise ValueError(f'Dataset does not exist: {dataset_path}')

  if config.preprocess_mode not in {'online', 'offline_cached'}:
    raise ValueError(f'preprocess_mode must be "online" or "offline_cached", got {config.preprocess_mode}')
  online_mode = config.preprocess_mode == 'online'

  datasets = {}
  for dataset_name, dataset_info in config.datasets.items():
    dataset_path = dataset_info['path']
    split = dataset_info.get('split', 'train')
    weight = dataset_info['weight']
    if is_main_process:
      logger.debug(f'loading {dataset_name} from {dataset_path} (split={split})')
    dataset_cls = DATASETS[dataset_name]
    resample_ratio = config.sampling_frequency / dataset_cls.sampling_frequency
    channel_order = datautils.get_channel_order(dataset_cls.channels, config.channels)
    mean = np.array(dataset_cls.mean, dtype=np.float16).reshape(-1, 1)
    std = np.array(dataset_cls.std, dtype=np.float16).reshape(-1, 1)
    preprocess = ECGPreprocessor(
      mean_std=(mean, std),
      resample_ratio=resample_ratio,
      channel_order=channel_order)
    legacy_preprocess = ECGPreprocessor(
      mean_std=(mean, std),
      resample_ratio=resample_ratio,
      channel_order=channel_order,
      transpose_input=True)
    _, ext = path.splitext(dataset_path)
    if path.isdir(dataset_path):
      # HuggingFace dataset directory
      is_variable = dataset_info.get('variable_length', False)
      if is_variable:
        data, starts, sizes = load_hf_variable_dataset(
          dataset_path, split=split, min_channel_size=config.channel_size)
        processed_records = []
        for i in range(len(sizes)):
          x = data[..., starts[i]:starts[i] + sizes[i]]  # (num_channels, channel_size)
          processed_records.append(preprocess(x) if online_mode else x)
        processed_records = [x for x in processed_records if x.shape[-1] >= config.channel_size]
        new_sizes = np.array([x.shape[-1] for x in processed_records])
        new_starts = np.concatenate([np.array([0]), np.cumsum(new_sizes[:-1])])
        new_data = np.concatenate(processed_records, axis=-1)  # (num_channels, total_time)
        dataset = VariableTensorDataset(
          new_data, new_starts, new_sizes,
          transform=TransformECG(crop_size=config.channel_size))
      else:
        data = load_hf_dataset(dataset_path, split=split)
        transform = [TransformECG(crop_size=config.channel_size)]
        if online_mode:
          transform.insert(0, preprocess)
        dataset = TensorDataset(
          data=data,
          transform=transform)
    elif ext == '.npy':
      transform = [TransformECG(crop_size=config.channel_size)]
      if online_mode:
        transform.insert(0, legacy_preprocess)
      dataset = TensorDataset(
        data=datautils.load_data_dump(dump_file=dataset_path),
        transform=transform)
    elif ext == '.npz':
      var_data, var_starts, var_sizes = load_variable_data_dump(
        dump_file=dataset_path,
        min_channel_size=config.channel_size,
        transform=legacy_preprocess if online_mode else None,
        processes=num_cpus)
      dataset = VariableTensorDataset(
        var_data, var_starts, var_sizes,
        transform=TransformECG(crop_size=config.channel_size))
    else:
      raise ValueError(f'Unsupported dataset format: {dataset_path}')
    datasets[dataset_name] = (dataset, weight)

  if is_main_process:
    logger.debug(f'{get_memory_usage() / 1024 ** 3:,.2f}GB memory used after loading data')

  # Each rank seeds numpy differently so DatasetRouter samples different data per rank
  np.random.seed(42 + rank)

  # With DDP, divide global batch size across all ranks
  local_batch_size = config.batch_size // world_size
  num_workers = config.dataloader_num_workers
  if num_workers is None:
    # Cap auto-selected workers so single-GPU jobs do not overrun file descriptors.
    num_workers = min(8, max(1, num_cpus // world_size))
  if num_workers < 1:
    raise ValueError(f'dataloader_num_workers must be >= 1, got {num_workers}')

  def worker_init_fn(worker_id):
    np.random.seed(rank * num_workers + worker_id)

  train_loader = DataLoader(
    dataset=DatasetRouter(datasets.values()),
    batch_size=local_batch_size,
    pin_memory=using_cuda,
    collate_fn=MaskCollator(
      patch_size=config.patch_size,
      min_block_size=config.min_block_size,
      min_keep_ratio=config.min_keep_ratio,
      max_keep_ratio=config.max_keep_ratio,
      strategy=config.masking_strategy,
      channel_independent=config.per_channel_patching),
    num_workers=num_workers,
    persistent_workers=bool(config.dataloader_persistent_workers and num_workers > 0),
    prefetch_factor=config.dataloader_prefetch_factor if num_workers > 0 else None,
    worker_init_fn=worker_init_fn)

  if is_main_process:
    logger.debug(f'dataloader settings: num_workers={num_workers} '
                 f'persistent_workers={bool(config.dataloader_persistent_workers and num_workers > 0)} '
                 f'prefetch_factor={config.dataloader_prefetch_factor} '
                 f'prefetch_queue_size={config.prefetch_queue_size}')

  # if device is CUDA, batch data will be asynchronously transferred to the GPU,
  #  so we should perform as many CPU operations as possible between loading and using a batch
  # compute total training steps
  total_dataset_size = sum(len(d) for d, _ in datasets.values())
  if config.epochs > 0:
    steps_per_epoch = max(1, total_dataset_size // (config.batch_size * config.gradient_accumulation_steps))
    total_steps = config.epochs * steps_per_epoch
    start_step = chkpt.get('step', chkpt.get('epoch', 0) * steps_per_epoch) if chkpt is not None else 0
    start_epoch = start_step // steps_per_epoch
  else:
    steps_per_epoch = None
    total_steps = config.steps
    start_step = chkpt['step'] if chkpt is not None else 0

  # setup hyperparameter schedules
  use_lejepa = config.jepa_mode == 'lejepa'

  if not use_lejepa:
    momentum_schedule = linear_schedule(
      total_steps=total_steps,
      start_value=config.encoder_momentum,
      final_value=config.final_encoder_momentum,
      step=start_step)

  lr_schedule = cosine_schedule(
    total_steps=total_steps,
    start_value=config.learning_rate,
    final_value=config.final_learning_rate,
    warmup_steps=int(total_steps * config.learning_rate_warmup_ratio),
    warmup_start_value=1e-6,
    step=start_step)
  wd_schedule = cosine_schedule(
    total_steps=total_steps,
    start_value=config.weight_decay,
    final_value=config.final_weight_decay,
    step=start_step)

  # setup model
  if use_lejepa:
    original_model = LeJEPA(
      config=config,
      use_sdp_kernel=using_cuda
    ).to(device)
  else:
    original_model = JEPA(
      config=config,
      momentum_schedule=momentum_schedule,
      use_sdp_kernel=using_cuda
    ).to(device)
  optimizer = original_model.get_optimizer(fused=using_cuda)

  if chkpt is not None:  # resume training from checkpoint
    original_model.load_state_dict(chkpt['model'])
    optimizer.load_state_dict(chkpt['optimizer'])

  # Wrap with DDP after loading checkpoint
  if is_distributed:
    model = DDP(original_model, device_ids=[local_rank])
  else:
    model = original_model

  if args.compile:
    model = torch.compile(model)

  train_loss = AverageMeter()
  train_start_time = time()
  last_loss = None
  last_lr = None
  pbar = tqdm(total=total_steps, initial=start_step, desc='Training',
              unit='step', disable=not is_main_process)

  def _train_step(train_iterator):
    update_learning_rate_(optimizer, next(lr_schedule))
    update_weight_decay_(optimizer, next(wd_schedule))
    batch_loss = 0.
    for i in range(config.gradient_accumulation_steps):
      x, mask_encoder, mask_predictor = train_iterator.next_batch()
      sync_ctx = (nullcontext() if not is_distributed or i == config.gradient_accumulation_steps - 1
                  else model.no_sync())
      with sync_ctx, auto_mixed_precision:
        loss = model(x, mask_encoder, mask_predictor)
        loss = loss / config.gradient_accumulation_steps
      loss.backward()
      batch_loss += loss.item()
    if config.gradient_clip > 0:
      torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
    optimizer.step()
    train_loss.update(batch_loss)
    optimizer.zero_grad(set_to_none=True)

  def log_training_stats(global_step, total_dataset_size):
    current_epoch = global_step * config.batch_size * config.gradient_accumulation_steps / total_dataset_size
    current_lr = optimizer.param_groups[0]['lr']
    pbar.set_postfix(loss=f'{train_loss.value:.4f}', lr=f'{current_lr:.2e}', epoch=f'{current_epoch:.2f}')
    pbar.update(1)
    return current_lr

  train_prefetcher = AsyncBatchPrefetcher(
    loader=train_loader,
    device=device,
    using_cuda=using_cuda,
    queue_size=config.prefetch_queue_size)
  train_prefetcher.reset()

  global_step = start_step
  last_completed_epoch = start_epoch

  def _save_checkpoint(step, epoch=None):
    chkpt = {
      'model': original_model.state_dict(),
      'optimizer': optimizer.state_dict(),
      'config': dataclasses.asdict(config),
      'step': step,
    }
    if epoch is not None:
      chkpt['epoch'] = epoch
    new_chkpt_path = path.join(args.out, f'chkpt_{step}.pt')
    torch.save(chkpt, new_chkpt_path)
    latest_chkpt_path = path.join(args.out, 'last_chkpt.pt')
    torch.save(chkpt, latest_chkpt_path)

  try:
    if config.epochs > 0:
      for epoch in range(start_epoch, config.epochs):
        last_completed_epoch = epoch + 1
        for _ in range(steps_per_epoch):
          _train_step(train_prefetcher)
          global_step += 1
          if is_main_process:
            last_loss = train_loss.value
            last_lr = log_training_stats(
              global_step=global_step,
              total_dataset_size=total_dataset_size)
            train_loss = AverageMeter()
            train_prefetcher.reset_metrics()
          if is_main_process and global_step % config.checkpoint_interval == 0:
            _save_checkpoint(step=global_step, epoch=epoch + 1)
    else:
      for step in range(start_step, config.steps):
        _train_step(train_prefetcher)
        global_step = step + 1
        if is_main_process:
          last_loss = train_loss.value
          last_lr = log_training_stats(
            global_step=step + 1,
            total_dataset_size=total_dataset_size)
          train_loss = AverageMeter()
          train_prefetcher.reset_metrics()
        if is_main_process and (step + 1) % config.checkpoint_interval == 0:
          _save_checkpoint(step=step + 1)

    if is_main_process and global_step > start_step and global_step % config.checkpoint_interval != 0:
      final_epoch = last_completed_epoch if config.epochs > 0 else None
      _save_checkpoint(step=global_step, epoch=final_epoch)
  finally:
    train_prefetcher.close()
    pbar.close()

  if is_main_process:
    total_time = time() - train_start_time
    h, rem = divmod(int(total_time), 3600)
    m, s = divmod(rem, 60)
    logger.info(
      f'\n{"=" * 50}\n'
      f'Training Complete\n'
      f'  Total steps   : {global_step}\n'
      f'  Total time    : {h:02d}h {m:02d}m {s:02d}s ({total_time:.1f}s)\n'
      f'  Final loss    : {last_loss:.4f}\n'
      f'  Final LR      : {last_lr:.6e}\n'
      f'{"=" * 50}'
    )

  if is_distributed:
    dist.destroy_process_group()


def load_variable_data_dump(dump_file, min_channel_size, transform=None, processes=None):
  data = datautils.load_variable_data_dump(dump_file, transform=transform, processes=processes)
  normalized_data = []
  for x in data:
    # Legacy variable-length dumps are channels-last: (channel_size, num_channels).
    if x.ndim == 2 and x.shape[-1] < min_channel_size <= x.shape[0]:
      x = x.T
    normalized_data.append(x)
  data = normalized_data
  data = [x for x in data if x.shape[-1] >= min_channel_size]
  sizes = np.array([x.shape[-1] for x in data])
  starts = np.concatenate([np.array([0]), np.cumsum(sizes[:-1])])
  data = np.concatenate(data, axis=-1)
  return data, starts, sizes

class AsyncBatchPrefetcher:
  _SENTINEL = object()

  def __init__(self, loader, device, using_cuda, queue_size=16):
    if queue_size < 1:
      raise ValueError(f'prefetch_queue_size must be >= 1, got {queue_size}')
    self.loader = loader
    self.device = device
    self.using_cuda = using_cuda
    self.queue_size = queue_size
    self.stream = torch.cuda.Stream(device=device) if using_cuda else None
    self.queue = None
    self.thread = None
    self.stop_event = threading.Event()
    self.exception = None
    self.next_device_batch = None
    self.queue_empty_count = 0
    self.queue_wait_time = 0.
    self.step_times = []

  def reset_metrics(self):
    self.queue_empty_count = 0
    self.queue_wait_time = 0.
    self.step_times = []

  def reset(self):
    self.close()
    self.queue = queue.Queue(maxsize=self.queue_size)
    self.stop_event.clear()
    self.exception = None
    self.next_device_batch = None
    self.thread = threading.Thread(target=self._producer_loop, daemon=True)
    self.thread.start()
    self._preload_next()

  def _producer_loop(self):
    try:
      for batch in self.loader:
        if self.stop_event.is_set():
          break
        if not self._put_queue_item(batch):
          return
      self._put_queue_item(self._SENTINEL)
    except Exception as exc:
      self.exception = exc
      self._put_queue_item(self._SENTINEL)

  def _put_queue_item(self, item):
    while True:
      try:
        self.queue.put(item, timeout=0.1)
        return True
      except queue.Full:
        if self.stop_event.is_set():
          return False

  def _preload_next(self):
    wait_start = time()
    if self.queue.empty():
      self.queue_empty_count += 1
    item = self.queue.get()
    self.queue_wait_time += time() - wait_start
    if item is self._SENTINEL:
      self.next_device_batch = None
    else:
      if self.using_cuda:
        with torch.cuda.stream(self.stream):
          item = tuple(x.to(self.device, non_blocking=True) for x in item)
      self.next_device_batch = item

  def next_batch(self):
    if self.exception is not None:
      raise self.exception
    if self.next_device_batch is None:
      raise StopIteration('Prefetcher reached end of loader')
    if self.using_cuda:
      torch.cuda.current_stream(device=self.device).wait_stream(self.stream)
    batch = self.next_device_batch
    step_start = time()
    self._preload_next()
    self.step_times.append(time() - step_start)
    return batch

  def close(self):
    self.stop_event.set()
    if self.thread is not None and self.thread.is_alive():
      self.thread.join(timeout=1.0)
    self.thread = None


class ECGPreprocessor:  # called per sample in dataloader workers
  def __init__(self, *, mean_std, resample_ratio, channel_order, transpose_input=False):
    self.mean, self.std = mean_std
    self.resample_ratio = resample_ratio
    self.channel_order = channel_order
    self.transpose_input = transpose_input  # for legacy channels-last .npy/.npz data

  def __call__(self, x):  # x: (num_channels, channel_size)
    if self.transpose_input:
      x = x.T  # (channel_size, num_channels) -> (num_channels, channel_size)
    x = x.copy()  # mmap slice is read-only; make a writable copy
    transforms.interpolate_NaNs_(x)
    if self.resample_ratio != 1.0:
      num_channels, channel_size = x.shape
      channel_size = int(self.resample_ratio * channel_size)
      x = transforms.resample(x, channel_size)
    transforms.normalize_(x, mean_std=(self.mean, self.std))
    x.clip(-5, 5, out=x)
    x = x[self.channel_order]
    return x


class TransformECG:  # called whenever dataloader accesses the data
  def __init__(self, crop_size):
    self.crop_size = crop_size

  def __call__(self, x):  # x: (num_channels, channel_size)
    x = transforms.random_crop(x, self.crop_size)
    x = torch.from_numpy(x).float()
    return x


if __name__ == '__main__':
  main()
