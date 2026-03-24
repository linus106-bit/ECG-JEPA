from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
  # data
  crop_duration: Optional[float]  # seconds
  crop_stride: Optional[float]  # seconds
  # model architecture
  num_classes: int
  use_register: bool
  attn_pooling: bool
  layer_scale_eps: float
  bias: bool
  dropout: float
  frozen: bool
  # training
  steps: int = 0
  epochs: int = 0
  batch_size: int = 256
  learning_rate: float = 1e-3
  final_learning_rate: float = 1e-5
  learning_rate_warmup_ratio: float = 0.0
  weight_decay: float = 0.
  opt_betas: tuple[float, float] = (0.9, 0.999)
  gradient_clip: float = 0.
  checkpoint_interval: int = 0
  early_stopping_patience: int = 0
