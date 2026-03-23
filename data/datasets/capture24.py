import numpy as np


class Capture24:
  sampling_frequency = 100
  channels = ('x', 'y', 'z')
  # placeholder: compute from training data with scripts/compute_mean_std.py
  mean = [0.0, 0.0, 0.0]
  std = [1.0, 1.0, 1.0]

  @staticmethod
  def load_data(data_dir):
    """Load Capture-24 from a HuggingFace dataset directory.

    Returns:
      data: np.ndarray of shape (N, channel_size, 3), dtype float16, channels last
      labels: np.ndarray of shape (N,), dtype int64
      splits: np.ndarray of shape (N,), 'train' or 'test'
    """
    from datasets import load_from_disk

    dataset = load_from_disk(data_dir)
    all_data, all_labels, all_splits = [], [], []
    for split_name in ('train', 'test'):
      ds = dataset[split_name]
      x = np.array(ds['input_values'], dtype=np.float16)  # (N, 3, 1000)
      x = x.transpose(0, 2, 1)  # (N, 1000, 3) channels last
      y = np.array(ds['label'], dtype=np.int64)
      all_data.append(x)
      all_labels.append(y)
      all_splits.extend([split_name] * len(x))
    return np.concatenate(all_data), np.concatenate(all_labels), np.array(all_splits)

  @staticmethod
  def load_labels(data_dir):
    """Load labels and split info saved by dump_data.py.

    Returns:
      labels: np.ndarray of shape (N,), dtype int64
      splits: np.ndarray of shape (N,), 'train' or 'test'
    """
    archive = np.load(f'{data_dir}_labels.npz')
    return archive['labels'], archive['splits']
