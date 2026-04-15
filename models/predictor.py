import math

import torch
from torch import nn

import configs
from models.modules import Block
from models.utils import get_1d_pos_embed, get_2d_rope_cache, apply_mask


class Predictor(nn.Module):
  def __init__(self, config: configs.pretrain.Config, use_sdp_kernel=True):
    super().__init__()
    self.config = config
    assert config.channel_size % config.patch_size == 0
    num_patches = config.num_patches
    self.embed = nn.Linear(config.dim, config.pred_dim, bias=config.bias)
    self.mask_token = nn.Parameter(torch.zeros(1, 1, config.pred_dim))
    self.patches_per_channel = config.channel_size // config.patch_size
    self.register_buffer(
      'pos_embed',
      get_1d_pos_embed(
        dim=config.pred_dim,
        num_patches=num_patches),
      persistent=False)
    self.use_2d_rope = config.per_channel_patching
    if self.use_2d_rope:
      head_dim = config.pred_dim // config.pred_num_heads
      rope_cos, rope_sin = get_2d_rope_cache(
        head_dim=head_dim,
        num_channels=config.num_channels,
        patches_per_channel=self.patches_per_channel)
      self.register_buffer('rope_cos', rope_cos, persistent=False)
      self.register_buffer('rope_sin', rope_sin, persistent=False)
    self.blocks = nn.ModuleList([
      Block(
        dim=config.pred_dim,
        num_heads=config.pred_num_heads,
        mlp_ratio=config.mlp_ratio,
        qkv_bias=config.qkv_bias,
        bias=config.bias,
        dropout=config.dropout,
        attn_dropout=config.attn_dropout,
        eps=config.norm_eps,
        use_sdp_kernel=use_sdp_kernel)
      for _ in range(config.pred_depth)
    ])
    self.norm = nn.LayerNorm(config.pred_dim, eps=config.norm_eps, bias=config.bias)
    self.proj = nn.Linear(config.pred_dim, config.dim, bias=config.bias)

    for name, module in self.named_modules():
      if isinstance(module, nn.Linear):
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
    nn.init.trunc_normal_(self.mask_token, mean=0., std=0.02)

  def forward(self, x, mask_encoder, mask_predictor):
    B, K = mask_predictor.size()
    pos_embed = self.pos_embed.repeat(B, 1, 1)
    pos_encoder = apply_mask(pos_embed, mask_encoder)
    x = self.embed(x)
    x = x + pos_encoder
    pos_predictor = apply_mask(pos_embed, mask_predictor)
    mask_token = self.mask_token.repeat(B, K, 1)
    mask_token = mask_token + pos_predictor
    x = torch.cat([x, mask_token], dim=1)
    token_positions = torch.cat([mask_encoder, mask_predictor], dim=1) if self.use_2d_rope else None
    for block in self.blocks:
      x = block(
        x,
        rope_cos=self.rope_cos if self.use_2d_rope else None,
        rope_sin=self.rope_sin if self.use_2d_rope else None,
        rope_positions=token_positions)
    x = self.norm(x)
    mask_token = x[:, -K:]
    mask_token = self.proj(mask_token)
    return mask_token
