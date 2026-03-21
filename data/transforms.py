import numpy as np
from scipy import signal


def interpolate_NaNs_(x):  # x: (channel_size, num_channels)
  # this transformation is in-place
  nan_mask = np.isnan(x)
  for index, contains_nans in enumerate(nan_mask.any(axis=0)):
    if contains_nans:
      mask = nan_mask[:, index]
      x[mask, index] = np.interp(
        np.flatnonzero(mask),
        np.flatnonzero(~mask),
        x[~mask, index])
  return x


def normalize_(x, mean_std=None, eps=0):  # x: (channel_size, num_channels)
  # this transformation is in-place
  if mean_std is None:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
  else:
    mean, std = mean_std
  x -= mean
  x /= std + eps
  return x


def highpass_filter(x, fs):  # x: (channel_size, num_channels)
  dtype = x.dtype
  [b, a] = signal.butter(4, 0.5, btype='highpass', fs=fs)
  x = signal.filtfilt(b, a, x, axis=0)
  x = x.astype(dtype)
  return x


def resample(x, channel_size):  # x: (channel_size, num_channels)
  dtype = x.dtype
  x = signal.resample(x, channel_size, axis=0)
  x = x.astype(dtype)
  return x


def random_crop(x, size):  # x: (channel_size, num_channels)
  start = np.random.randint(len(x) - size + 1)
  x = x[start:start + size]
  return x


import torch  # noqa: E402


class PreprocessECG:  # called once when loading the data
  def __init__(self, *, mean_std, resample_ratio, channel_order):
    self.mean, self.std = mean_std
    self.resample_ratio = resample_ratio
    self.channel_order = channel_order

  def __call__(self, x):
    interpolate_NaNs_(x)
    if self.resample_ratio != 1.0:
      channel_size, num_channels = x.shape
      channel_size = int(self.resample_ratio * channel_size)
      x = resample(x, channel_size)
    normalize_(x, mean_std=(self.mean, self.std))
    x.clip(-5, 5, out=x)
    x = x[:, self.channel_order]
    return x


class TransformECG:  # called whenever dataloader accesses the data
  def __init__(self, crop_size):
    self.crop_size = crop_size

  def __call__(self, x):
    x = random_crop(x, self.crop_size)
    x = x.transpose()  # channels first
    x = torch.from_numpy(x).float()
    return x


class FinetunePreprocessECG:
  def __init__(self, channel_size=None, remove_baseline_wander=False, fs=500):
    self.channel_size = channel_size
    self.remove_baseline_wander = remove_baseline_wander
    self.fs = fs

  def __call__(self, x):
    channel_size, num_channels = x.shape
    if self.remove_baseline_wander:
      x = highpass_filter(x, fs=self.fs)
    if self.channel_size is not None and self.channel_size != channel_size:
      x = resample(x, self.channel_size)
    return x


class TrainTransformECG:  # called whenever dataloader accesses the data
  def __init__(self, crop_size=None):
    self.crop_size = crop_size

  def __call__(self, x):
    if self.crop_size is not None:
      x = random_crop(x, self.crop_size)
    x = x.transpose()  # channels first
    x = torch.from_numpy(x).float()
    return x


class EvalTransformECG:  # called whenever dataloader accesses the data
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
