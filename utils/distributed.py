import os
from contextlib import nullcontext

import torch
import torch.distributed as dist


def setup_distributed():
  local_rank = int(os.environ.get('LOCAL_RANK', 0))
  rank = int(os.environ.get('RANK', 0))
  world_size = int(os.environ.get('WORLD_SIZE', 1))
  is_distributed = world_size > 1
  is_main_process = rank == 0

  if is_distributed:
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)

  return local_rank, rank, world_size, is_distributed, is_main_process


def setup_cuda(logger, is_main_process):
  if is_main_process:
    logger.debug('TF32 tensor cores are enabled')
  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.allow_tf32 = True
  torch.backends.cudnn.benchmark = True


def setup_amp(amp_arg, using_cuda, logger, is_main_process):
  if amp_arg == 'float32' or not using_cuda:  # don't use AMP on a CPU
    if is_main_process:
      logger.debug('using float32 precision')
    return nullcontext()
  elif amp_arg == 'bfloat16':
    # bfloat16 preserves the range of float32, so it does not require scaling
    if is_main_process:
      logger.debug('using bfloat16 with AMP')
    return torch.cuda.amp.autocast(dtype=torch.bfloat16)
  else:
    raise ValueError('Failed to choose floating-point format.')
