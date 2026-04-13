import argparse
from os import makedirs, path

import numpy as np
from tqdm import tqdm

from data import transforms, utils as datautils
from data.datasets import DATASETS
from data.utils import load_hf_dataset


class ECGPreprocessor:
  def __init__(self, *, mean_std, resample_ratio, channel_order):
    self.mean, self.std = mean_std
    self.resample_ratio = resample_ratio
    self.channel_order = channel_order

  def __call__(self, x):
    x = x.copy()
    transforms.interpolate_NaNs_(x)
    if self.resample_ratio != 1.0:
      _, channel_size = x.shape
      x = transforms.resample(x, int(self.resample_ratio * channel_size))
    transforms.normalize_(x, mean_std=(self.mean, self.std))
    x.clip(-5, 5, out=x)
    return x[self.channel_order]


def main():
  parser = argparse.ArgumentParser(description='Offline preprocess MIMIC (or other) ECG HF dataset into .npy/.npz cache.')
  parser.add_argument('--dataset-path', required=True, help='HF dataset directory')
  parser.add_argument('--dataset-name', default='mimic-iv-ecg', choices=sorted(DATASETS.keys()))
  parser.add_argument('--split', default='train')
  parser.add_argument('--out', required=True, help='output .npy or .npz path')
  parser.add_argument('--sampling-frequency', type=int, default=500)
  parser.add_argument('--channels', nargs='+', default=['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'])
  parser.add_argument('--dtype', default='float16', choices=['float16', 'float32'])
  parser.add_argument('--variable-length', action='store_true', help='save variable-length cache as .npz(data, sizes)')
  args = parser.parse_args()

  dataset_cls = DATASETS[args.dataset_name]
  resample_ratio = args.sampling_frequency / dataset_cls.sampling_frequency
  channel_order = datautils.get_channel_order(dataset_cls.channels, args.channels)
  mean = np.array(dataset_cls.mean, dtype=np.float16).reshape(-1, 1)
  std = np.array(dataset_cls.std, dtype=np.float16).reshape(-1, 1)
  preprocess = ECGPreprocessor(mean_std=(mean, std), resample_ratio=resample_ratio, channel_order=channel_order)

  ds = load_hf_dataset(args.dataset_path, split=args.split, dtype=np.float16)

  dtype = np.float16 if args.dtype == 'float16' else np.float32
  if args.variable_length:
    records = []
    sizes = []
    for i in tqdm(range(len(ds)), desc='preprocess'):
      x = preprocess(ds[i]).astype(dtype, copy=False)
      records.append(x)
      sizes.append(x.shape[-1])
    flat = np.concatenate(records, axis=-1)
    sizes = np.asarray(sizes, dtype=np.int64)
    makedirs(path.dirname(args.out) or '.', exist_ok=True)
    np.savez_compressed(args.out, data=flat, sizes=sizes)
  else:
    first = preprocess(ds[0]).astype(dtype, copy=False)
    cache = np.empty((len(ds), *first.shape), dtype=dtype)
    cache[0] = first
    for i in tqdm(range(1, len(ds)), desc='preprocess'):
      cache[i] = preprocess(ds[i]).astype(dtype, copy=False)
    makedirs(path.dirname(args.out) or '.', exist_ok=True)
    np.save(args.out, cache)


if __name__ == '__main__':
  main()
