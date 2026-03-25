from dataclasses import dataclass, field


@dataclass
class Config:
  # data
  sampling_frequency: int = 500
  channels: tuple[str, ...] = ('I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
  channel_size: int = 5000
  patch_size: int = 25
  min_block_size: int = 10
  min_keep_ratio: float = 0.15
  max_keep_ratio: float = 0.25
  datasets: dict = field(default_factory=dict)  # {name: {path: str, weight: float}}
  # model architecture
  dim: int = 384
  depth: int = 8
  num_heads: int = 6
  pred_dim: int = 192
  pred_depth: int = 8
  pred_num_heads: int = 6
  mlp_ratio: float = 4.
  qkv_bias: bool = False
  dropout: float = 0.
  attn_dropout: float = 0.
  num_registers: int = 1
  bias: bool = False
  norm_eps: float = 1e-6
  layer_scale_eps: float = 0.
  # training
  batch_size: int = 2048
  encoder_momentum: float = 0.998
  final_encoder_momentum: float = 0.9995
  learning_rate: float = 1e-3
  final_learning_rate: float = 1e-6
  learning_rate_warmup_ratio: float = 0.05
  weight_decay: float = 1e-2
  final_weight_decay: float = 1e-1
  opt_betas: tuple[float, float] = (0.9, 0.99)
  opt_eps: float = 1e-6
  gradient_clip: float = 0.
  gradient_accumulation_steps: int = 1
  checkpoint_interval: int = 200
  steps: int = 0
  epochs: int = 0
  # model type selection ('vit', 'cnn', or 'mamba')
  model_type: str = 'vit'
  # CNN-specific
  cnn_kernel_size: int = 7
  # Mamba-specific
  mamba_d_state: int = 16
  mamba_d_conv: int = 4
  mamba_expand: int = 2
  # DMT-JEPA: discriminative masked targets (0 = disabled)
  dmt_window_size: int = 0
  dmt_num_neighbors: int = 4

  @property
  def num_channels(self):
    return len(self.channels)
