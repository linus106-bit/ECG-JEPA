import numpy as np


class SDB:
  """Sleep Disorder Breathing dataset metadata.

  Assumes HF dataset records use channels-first layout: (C, T),
  with a single "green" channel and variable channel sizes.
  """

  # Keep defaults aligned with Capture-24 unless overridden in configs.
  sampling_frequency = 100
  channels = ('green',)
  # Placeholder stats; compute dataset-specific values when available.
  mean = [0.0]
  std = [1.0]

  @staticmethod
  def load_data(data_dir):
    """Load SDB from a HuggingFace dataset directory.

    Returns:
      data: np.ndarray(dtype=object), each item shape (channel_size, 1), float16
      labels: np.ndarray of shape (N,), dtype int64 (or -1 if label column is missing)
      splits: np.ndarray of shape (N,), split names (e.g., train/test)
    """
    from datasets import load_dataset
    from tqdm import tqdm

    dataset = load_dataset(data_dir)
    all_data, all_labels, all_splits = [], [], []

    for split_name in dataset.keys():
      ds = dataset[split_name]
      for sample in tqdm(ds, desc=split_name):
        x = np.array(sample['data'], dtype=np.float16)  # (C, T)
        x = x.T  # (T, C), channels last
        all_data.append(x)
        all_labels.append(int(sample['label']) if 'label' in sample else -1)
        all_splits.append(split_name)

    return np.array(all_data, dtype=object), np.array(all_labels, dtype=np.int64), np.array(all_splits)
