import argparse
import dataclasses
import logging.config
import os
import pprint
from contextlib import nullcontext
from os import path, makedirs
from time import time

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
  # Resolve config file and extract run: section early (needed before distributed setup)
  if not path.isfile(args.config):
    config_file = path.join(path.dirname(configs.pretrain.__file__), f'{args.config}.yaml')
    if not path.isfile(config_file):
      raise ValueError(f'Failed to read configuration file {args.config}')
    args.config = config_file
  yaml_dict = configs.load_config_file(args.config)
  run_config = yaml_dict.pop('run', {})
  if args.out is None:
    args.out = run_config.get('out_dir', 'pretrain')
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

  for dataset_name, dataset_info in config.datasets.items():
    if dataset_name not in DATASETS:
      raise ValueError(f'Unknown dataset {dataset_name}. '
                       f'Available datasets are {list(DATASETS)}')
    dataset_path = dataset_info['path']
    if not path.isdir(dataset_path) and not path.isfile(dataset_path):
      raise ValueError(f'Dataset does not exist: {dataset_path}')

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
    _, ext = path.splitext(dataset_path)
    if path.isdir(dataset_path):
      # HuggingFace dataset directory
      is_variable = dataset_info.get('variable_length', False)
      if is_variable:
        data, starts, sizes = load_hf_variable_dataset(
          dataset_path, split=split, min_channel_size=config.channel_size)
        # apply preprocessing to variable-length data
        preprocessed = []
        preprocess = PreprocessECG(
          mean_std=(mean, std),
          resample_ratio=resample_ratio,
          channel_order=channel_order)
        for i in range(len(sizes)):
          x = data[..., starts[i]:starts[i] + sizes[i]]  # (num_channels, channel_size)
          preprocessed.append(preprocess(x))
        preprocessed = [x for x in preprocessed if x.shape[-1] >= config.channel_size]
        new_sizes = np.array([x.shape[-1] for x in preprocessed])
        new_starts = np.concatenate([np.array([0]), np.cumsum(new_sizes[:-1])])
        new_data = np.concatenate(preprocessed, axis=-1)  # (num_channels, total_time)
        dataset = VariableTensorDataset(
          new_data, new_starts, new_sizes,
          transform=TransformECG(crop_size=config.channel_size))
      else:
        data = load_hf_dataset(dataset_path, split=split)
        dataset = TensorDataset(
          data=data,
          transform=[
            PreprocessECG(
              mean_std=(mean, std),
              resample_ratio=resample_ratio,
              channel_order=channel_order),
            TransformECG(crop_size=config.channel_size),
          ])
    elif ext == '.npy':
      dataset = TensorDataset(
        data=datautils.load_data_dump(dump_file=dataset_path),
        transform=[
          PreprocessECG(
            mean_std=(mean, std),
            resample_ratio=resample_ratio,
            channel_order=channel_order,
            transpose_input=True),  # legacy .npy is channels-last
          TransformECG(crop_size=config.channel_size),
        ])
    elif ext == '.npz':
      records = load_variable_data_dump(
        dump_file=dataset_path,
        transform=PreprocessECG(
          mean_std=(mean, std),
          resample_ratio=resample_ratio,
          channel_order=channel_order,
          transpose_input=True),  # legacy .npz is channels-last
        processes=num_cpus)
      # filter short records and build concatenated variable-length dataset
      records = [x for x in records if x.shape[-1] >= config.channel_size]
      var_sizes = np.array([x.shape[-1] for x in records])
      var_starts = np.concatenate([np.array([0]), np.cumsum(var_sizes[:-1])])
      var_data = np.concatenate(records, axis=-1)  # (num_channels, total_time)
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
  # Cap dataloader workers so single-GPU runs on high-core-count hosts
  # do not exhaust file descriptors while keeping good throughput.
  num_workers = min(8, max(1, num_cpus // world_size))

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
      strategy=config.masking_strategy),
    num_workers=num_workers,
    worker_init_fn=worker_init_fn,
    persistent_workers=(num_workers > 0))

  def map_to_device(data_iterator, device=None):
    for batch in data_iterator:
      yield tuple(x.to(device, non_blocking=using_cuda) for x in batch)

  def prefetch_batch(data_iterator):
    prefetched_batch = next(data_iterator)
    for next_batch in data_iterator:
      yield prefetched_batch
      prefetched_batch = next_batch
    yield prefetched_batch

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

  step_time = AverageMeter()
  train_loss = AverageMeter()

  def _train_step(train_iterator):
    step_start = time()
    update_learning_rate_(optimizer, next(lr_schedule))
    update_weight_decay_(optimizer, next(wd_schedule))
    batch_loss = 0.
    for i in range(config.gradient_accumulation_steps):
      x, mask_encoder, mask_predictor = next(train_iterator)
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
    step_time.update(time() - step_start)

  global_step = start_step
  if config.epochs > 0:
    for epoch in range(start_epoch, config.epochs):
      train_iterator = iter(train_loader)
      train_iterator = map_to_device(train_iterator, device=device)
      train_iterator = prefetch_batch(train_iterator)
      for _ in range(steps_per_epoch):
        _train_step(train_iterator)
        global_step += 1
        if is_main_process:
          current_epoch = global_step * config.batch_size * config.gradient_accumulation_steps / total_dataset_size
          logger.info(f'step: {global_step} '
                      f'epoch: {current_epoch:.4f} '
                      f'train_loss: {train_loss.value:.4f} '
                      f'step_time: {step_time.value:.4f}')
          step_time = AverageMeter()
          train_loss = AverageMeter()
        if is_main_process and global_step % config.checkpoint_interval == 0:
          new_chkpt_path = path.join(args.out, f'chkpt_{global_step}.pt')
          torch.save({
            'model': original_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': dataclasses.asdict(config),
            'epoch': epoch + 1,
            'step': global_step,
          }, new_chkpt_path)
  else:
    train_iterator = iter(train_loader)
    train_iterator = map_to_device(train_iterator, device=device)
    train_iterator = prefetch_batch(train_iterator)
    for step in range(start_step, config.steps):
      _train_step(train_iterator)
      if is_main_process:
        current_epoch = (step + 1) * config.batch_size * config.gradient_accumulation_steps / total_dataset_size
        logger.info(f'step: {step + 1} '
                    f'epoch: {current_epoch:.4f} '
                    f'train_loss: {train_loss.value:.4f} '
                    f'step_time: {step_time.value:.4f}')
        step_time = AverageMeter()
        train_loss = AverageMeter()
      if is_main_process and (step + 1) % config.checkpoint_interval == 0:
        new_chkpt_path = path.join(args.out, f'chkpt_{step + 1}.pt')
        torch.save({
          'model': original_model.state_dict(),
          'optimizer': optimizer.state_dict(),
          'config': dataclasses.asdict(config),
          'step': step + 1,
        }, new_chkpt_path)

  if is_distributed:
    dist.destroy_process_group()


def load_variable_data_dump(dump_file, min_channel_size, transform=None, processes=None):
  data = datautils.load_variable_data_dump(dump_file, transform=transform, processes=processes)
  data = [x for x in data if len(x) >= min_channel_size]
  sizes = np.array([len(x) for x in data])
  starts = np.concatenate([np.array([0]), np.cumsum(sizes[:-1])])
  data = np.concatenate(data)
  return data, starts, sizes


class PreprocessECG:  # called per sample in dataloader workers
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
