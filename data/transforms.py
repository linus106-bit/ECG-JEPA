import numpy as np
from scipy import signal


def interpolate_NaNs_(x):  # x: (num_channels, channel_size)
  # this transformation is in-place
  nan_mask = np.isnan(x)
  for index, contains_nans in enumerate(nan_mask.any(axis=1)):
    if contains_nans:
      mask = nan_mask[index]
      x[index, mask] = np.interp(
        np.flatnonzero(mask),
        np.flatnonzero(~mask),
        x[index, ~mask])
  return x


def normalize_(x, mean_std=None, eps=0):  # x: (num_channels, channel_size)
  # this transformation is in-place
  if mean_std is None:
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
  else:
    mean, std = mean_std
  x -= mean
  x /= std + eps
  return x


def highpass_filter(x, fs):  # x: (num_channels, channel_size)
  dtype = x.dtype
  [b, a] = signal.butter(4, 0.5, btype='highpass', fs=fs)
  x = signal.filtfilt(b, a, x, axis=-1)
  x = x.astype(dtype)
  return x


def resample(x, channel_size):  # x: (num_channels, channel_size)
  dtype = x.dtype
  x = signal.resample(x, channel_size, axis=-1)
  x = x.astype(dtype)
  return x


def random_crop(x, size):  # x: (num_channels, channel_size)
  channel_size = x.shape[-1]
  start = np.random.randint(channel_size - size + 1)
  x = x[..., start:start + size]
  return x
