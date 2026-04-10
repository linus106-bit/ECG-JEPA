import math

import torch
from torch import nn

import configs
from models.modules import PatchEmbedding, Block
from models.utils import get_1d_pos_embed, get_2d_rope_cache, apply_mask


class VisionTransformer(nn.Module):
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
    self.patches_per_channel = config.channel_size // config.patch_size
    self.register_buffer(
      'pos_embed',
      get_1d_pos_embed(
        dim=config.dim,
        num_patches=num_patches),
      persistent=False)
    self.use_2d_rope = config.per_channel_patching
    if self.use_2d_rope:
      head_dim = config.dim // config.num_heads
      rope_cos, rope_sin = get_2d_rope_cache(
        head_dim=head_dim,
        num_channels=config.num_channels,
        patches_per_channel=self.patches_per_channel)
      self.register_buffer('rope_cos', rope_cos, persistent=False)
      self.register_buffer('rope_sin', rope_sin, persistent=False)
    if config.num_registers > 0:
      self.registers = nn.Parameter(torch.empty(1, config.num_registers, config.dim))
      nn.init.trunc_normal_(self.registers, mean=0., std=0.02)
    self.blocks = nn.ModuleList([
      Block(
        dim=config.dim,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio,
        qkv_bias=config.qkv_bias,
        bias=config.bias,
        dropout=config.dropout,
        attn_dropout=config.attn_dropout,
        eps=config.norm_eps,
        layer_scale_eps=config.layer_scale_eps,
        use_sdp_kernel=use_sdp_kernel)
      for _ in range(config.depth)
    ])
    self.norm = nn.LayerNorm(config.dim, eps=config.norm_eps, bias=config.bias)

    for name, module in self.named_modules():
      if isinstance(module, (nn.Linear, nn.Conv1d)):
        if name.endswith('mlp.fc2') or name.endswith('attn.proj'):
          # residual projections are initialized with scaled std
          nn.init.trunc_normal_(module.weight, mean=0., std=0.02 / math.sqrt(2 * config.depth))
        else:
          nn.init.trunc_normal_(module.weight, mean=0., std=0.02)
        if module.bias is not None:
          nn.init.zeros_(module.bias)
      elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        if module.bias is not None:
          nn.init.zeros_(module.bias)

  def forward(self, x, mask=None):
    x = self.patch_embed(x)
    B, N, D = x.size()
    x = x + self.pos_embed[:, :N]
    token_positions = None
    if mask is not None:
      x = apply_mask(x, mask)
      token_positions = mask
    elif self.use_2d_rope:
      token_positions = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
    if self.config.num_registers > 0:
      registers = self.registers.repeat(B, 1, 1)
      x = torch.cat([registers, x], dim=1)
    for block in self.blocks:
      x = block(
        x,
        rope_cos=self.rope_cos if self.use_2d_rope else None,
        rope_sin=self.rope_sin if self.use_2d_rope else None,
        rope_positions=token_positions,
        rope_prefix_tokens=self.config.num_registers if self.config.num_registers > 0 else 0)
    x = self.norm(x)
    if not self.keep_registers and self.config.num_registers > 0:
      x = x[:, self.config.num_registers:]
    return x
