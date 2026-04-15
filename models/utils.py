import math

import torch


def apply_mask(x, mask):  # x: (B, N, D); mask: (B, K) with values in range [0, N)
  mask = mask.unsqueeze(-1).repeat(1, 1, x.size(-1))
  x = torch.gather(x, dim=1, index=mask)
  return x  # x: (B, K, D)


def get_1d_pos_embed(dim, num_patches):
  assert dim % 2 == 0
  position = torch.arange(num_patches).unsqueeze(1)
  div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
  pos_embed = torch.zeros(1, num_patches, dim)
  pos_embed[0, :, 0::2] = torch.sin(position * div_term)
  pos_embed[0, :, 1::2] = torch.cos(position * div_term)
  return pos_embed


def get_2d_rope_cache(head_dim, num_channels, patches_per_channel):
  if head_dim % 2 != 0:
    raise ValueError(f'2D RoPE requires an even head_dim, got {head_dim}')
  half_dim = head_dim // 2  # number of rotary pairs
  ch_dim = half_dim // 2
  t_dim = half_dim - ch_dim
  if ch_dim == 0 or t_dim == 0:
    raise ValueError(f'2D RoPE requires head_dim >= 4, got {head_dim}')

  channel_ids = torch.arange(num_channels, dtype=torch.float32).repeat_interleave(patches_per_channel)
  patch_ids = torch.arange(patches_per_channel, dtype=torch.float32).repeat(num_channels)

  ch_scale = -math.log(10000.0) / ch_dim
  t_scale = -math.log(10000.0) / t_dim
  ch_div_term = torch.exp(torch.arange(0, ch_dim, dtype=torch.float32) * ch_scale)
  t_div_term = torch.exp(torch.arange(0, t_dim, dtype=torch.float32) * t_scale)

  angles_ch = channel_ids.unsqueeze(-1) * ch_div_term.unsqueeze(0)  # (N, ch_dim)
  angles_t = patch_ids.unsqueeze(-1) * t_div_term.unsqueeze(0)  # (N, t_dim)
  angles = torch.cat([angles_ch, angles_t], dim=-1)  # (N, half_dim)

  cos = torch.repeat_interleave(torch.cos(angles), 2, dim=-1)  # (N, head_dim)
  sin = torch.repeat_interleave(torch.sin(angles), 2, dim=-1)  # (N, head_dim)
  return cos, sin
