import argparse
import logging
import logging.config
from contextlib import nullcontext
from os import path

import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader

import configs
from data.datasets import Capture24
from data.utils import TensorDataset, get_channel_order
from models import create_encoder, EncoderClassifier
from utils.monitoring import get_cpu_count

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True, help='path to finetuned HAR checkpoint (.pt)')
parser.add_argument('--data-dir', default=None, help='path to Capture-24 data directory')
parser.add_argument('--dump', help='path to test dump file (.npy); defaults to {data_dir}_test.npy')
parser.add_argument('--amp', default='float32', choices=['bfloat16', 'float32'])
args = parser.parse_args()


def main():
  logging.config.fileConfig('logging.ini')
  logger = logging.getLogger('app')

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  using_cuda = device.type == 'cuda'
  num_cpus = get_cpu_count()
  logger.debug(f'using {device}, {num_cpus} CPUs')

  if using_cuda:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

  if args.amp == 'bfloat16' and using_cuda:
    auto_mixed_precision = torch.cuda.amp.autocast(dtype=torch.bfloat16)
  else:
    auto_mixed_precision = nullcontext()

  # load finetuned checkpoint
  logger.debug(f'loading checkpoint from {args.checkpoint}')
  chkpt = torch.load(args.checkpoint, map_location='cpu')
  encoder_config = configs.pretrain.Config(**chkpt['config'])
  eval_config = configs.eval.Config(**chkpt['eval_config'])

  # resolve test data path
  data_dir = args.data_dir
  test_dump = args.dump or (f'{data_dir}_test.npy' if data_dir else None)
  if test_dump is None:
    raise ValueError('Specify --data-dir or --dump to locate the test .npy file')
  if not path.isfile(test_dump):
    raise ValueError(f'Test dump not found: {test_dump}')

  # load labels
  if data_dir is None:
    # infer data_dir from dump path (strip trailing _test.npy)
    data_dir = test_dump.removesuffix('_test.npy')
  logger.debug(f'loading test labels from {data_dir}')
  _, test_labels = Capture24.load_labels(data_dir)

  logger.debug(f'loading test data from {test_dump}')
  x_test = np.load(test_dump)

  num_classes = int(test_labels.max() + 1)
  logger.debug(f'test={len(x_test)}, num_classes={num_classes}')

  # channel ordering
  channel_order = get_channel_order(Capture24.channels, encoder_config.channels)
  x_test = x_test[:, :, channel_order]
  y_test = torch.from_numpy(test_labels).long()

  # crop config
  if eval_config.crop_duration is not None:
    crop_size = int(eval_config.crop_duration * encoder_config.sampling_frequency)
    crop_stride = int((eval_config.crop_stride or eval_config.crop_duration) * encoder_config.sampling_frequency)
  else:
    crop_size = None
    crop_stride = None

  test_loader = DataLoader(
    dataset=TensorDataset(
      data=x_test,
      labels=y_test,
      transform=EvalTransformSignal(crop_size=crop_size, crop_stride=crop_stride)),
    batch_size=eval_config.batch_size,
    num_workers=max(1, num_cpus))

  # build model
  encoder = create_encoder(config=encoder_config, keep_registers=eval_config.use_register, use_sdp_kernel=using_cuda)
  model = EncoderClassifier(encoder, eval_config, use_sdp_kernel=using_cuda).to(device)
  model.load_state_dict(chkpt['model'])
  logger.debug('model loaded')

  # run test inference
  test_preds, test_targets = [], []
  model.eval()
  with torch.inference_mode():
    for batch in test_loader:
      bx, by = (t.to(device) for t in batch)
      with auto_mixed_precision:
        if eval_config.crop_duration is not None:
          batch_size, num_crops, num_channels, channel_size = bx.size()
          logits = model(bx.reshape(-1, num_channels, channel_size))
          logits = logits.reshape(batch_size, num_crops, eval_config.num_classes).mean(dim=1)
        else:
          logits = model(bx)
      test_preds.append(logits.argmax(dim=1).clone())
      test_targets.append(by.clone())

  test_preds = torch.cat(test_preds).cpu().numpy()
  test_targets = torch.cat(test_targets).cpu().numpy()
  test_f1 = f1_score(y_true=test_targets, y_pred=test_preds, average='macro')
  test_acc = accuracy_score(y_true=test_targets, y_pred=test_preds)
  logger.info(f'test_f1={test_f1:.4f}  test_acc={test_acc:.4f}')


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
    return torch.from_numpy(x).float()


def strided_crops(x, size, stride):  # x: (T, C)
  T, num_channels = x.shape
  starts = range(0, T - size + 1, stride)
  out = np.empty((len(starts), size, num_channels), dtype=x.dtype)
  for i, s in enumerate(starts):
    out[i] = x[s:s + size]
  return out


if __name__ == '__main__':
  main()
