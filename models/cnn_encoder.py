import torch
from torch import nn

import configs
from models.modules import PatchEmbedding
from models.utils import get_1d_pos_embed, apply_mask


class ConvBlock(nn.Module):
  def __init__(self, dim, kernel_size=7, dropout=0.):
    super().__init__()
    padding = kernel_size // 2
    self.conv1 = nn.Conv1d(dim, dim, kernel_size, padding=padding, bias=False)
    self.norm1 = nn.GroupNorm(1, dim)
    self.act1 = nn.GELU()
    self.conv2 = nn.Conv1d(dim, dim, kernel_size, padding=padding, bias=False)
    self.norm2 = nn.GroupNorm(1, dim)
    self.act2 = nn.GELU()
    self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

  def forward(self, x):  # x: (B, D, N)
    residual = x
    x = self.conv1(x)
    x = self.norm1(x)
    x = self.act1(x)
    x = self.dropout(x)
    x = self.conv2(x)
    x = self.norm2(x)
    x = x + residual
    x = self.act2(x)
    return x


class CNNEncoder(nn.Module):
  def __init__(self, config: configs.pretrain.Config, keep_registers=False, use_sdp_kernel=True):
    super().__init__()
    self.config = config
    self.keep_registers = keep_registers
    assert config.channel_size % config.patch_size == 0
    num_patches = config.num_patches
    self.patch_embed = PatchEmbedding(
      dim=config.dim,
      in_channels=config.num_channels,
      patch_size=config.patch_size,
      channel_independent=config.per_channel_patching,
      bias=config.bias)
    self.register_buffer(
      'pos_embed',
      get_1d_pos_embed(
        dim=config.dim,
        num_patches=num_patches),
      persistent=False)
    if config.num_registers > 0:
      self.registers = nn.Parameter(torch.empty(1, config.num_registers, config.dim))
      nn.init.trunc_normal_(self.registers, mean=0., std=0.02)
    self.blocks = nn.ModuleList([
      ConvBlock(
        dim=config.dim,
        kernel_size=config.cnn_kernel_size,
        dropout=config.dropout)
      for _ in range(config.depth)
    ])
    self.norm = nn.LayerNorm(config.dim, eps=config.norm_eps, bias=config.bias)

    for name, module in self.named_modules():
      if isinstance(module, (nn.Linear, nn.Conv1d)):
        nn.init.trunc_normal_(module.weight, mean=0., std=0.02)
        if module.bias is not None:
          nn.init.zeros_(module.bias)
      elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        if module.bias is not None:
          nn.init.zeros_(module.bias)

  def forward(self, x, mask=None):
    x = self.patch_embed(x)  # (B, N, D)
    B, N, D = x.size()
    x = x + self.pos_embed[:, :N]
    # CNN processes all tokens, then we mask the output
    x = x.transpose(1, 2)  # (B, D, N) for conv1d
    for block in self.blocks:
      x = block(x)
    x = x.transpose(1, 2)  # (B, N, D)
    x = self.norm(x)
    if mask is not None:
      x = apply_mask(x, mask)
    if self.config.num_registers > 0:
      registers = self.registers.repeat(B, 1, 1)
      x = torch.cat([registers, x], dim=1)
    if not self.keep_registers and self.config.num_registers > 0:
      x = x[:, self.config.num_registers:]
    return x
