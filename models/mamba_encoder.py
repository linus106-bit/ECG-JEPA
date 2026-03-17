import torch
from torch import nn
from torch.nn import functional as F

import configs
from models.modules import PatchEmbedding
from models.utils import get_1d_pos_embed, apply_mask


class SelectiveSSM(nn.Module):
  """Selective State Space Model (S6) with pure PyTorch implementation.
  For optimized CUDA kernels, install mamba-ssm package."""
  def __init__(self, d_inner, d_state=16, d_conv=4, dt_rank=None):
    super().__init__()
    self.d_inner = d_inner
    self.d_state = d_state
    self.dt_rank = dt_rank or (d_inner + 15) // 16

    # S4D real initialization for A
    A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(d_inner, -1)
    self.A_log = nn.Parameter(torch.log(A))

    self.D = nn.Parameter(torch.ones(d_inner))

    # short conv for local context
    self.conv1d = nn.Conv1d(
      d_inner, d_inner,
      kernel_size=d_conv,
      padding=d_conv - 1,
      groups=d_inner)

    # projections for B, C, delta from input
    self.x_proj = nn.Linear(d_inner, self.dt_rank + d_state * 2, bias=False)
    self.dt_proj = nn.Linear(self.dt_rank, d_inner)

  def forward(self, x):  # x: (B, L, d_inner)
    B, L, D = x.shape

    # conv
    x_conv = x.transpose(1, 2)  # (B, D, L)
    x_conv = self.conv1d(x_conv)[:, :, :L]
    x_conv = F.silu(x_conv).transpose(1, 2)  # (B, L, D)

    # project to get delta, B, C
    x_proj = self.x_proj(x_conv)
    dt = x_proj[:, :, :self.dt_rank]
    B_param = x_proj[:, :, self.dt_rank:self.dt_rank + self.d_state]
    C_param = x_proj[:, :, self.dt_rank + self.d_state:]

    # compute delta (discretization step)
    dt = self.dt_proj(dt)  # (B, L, d_inner)
    dt = F.softplus(dt)

    # discretize: A_bar = exp(delta * A), B_bar = delta * B
    A = -torch.exp(self.A_log)  # (d_inner, d_state)

    # selective scan
    y = self._selective_scan(x_conv, dt, A, B_param, C_param)
    y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)
    return y

  def _selective_scan(self, x, dt, A, B, C):
    """Parallel selective scan using Heinsen recurrence (no Python for-loop).
    Converts h[t] = dA[t]*h[t-1] + dB[t] into cumsum operations."""
    # dt: (B, L, D), A: (D, N), B: (B, L, N), C: (B, L, N), x: (B, L, D)
    dtA = torch.einsum('bld,dn->bldn', dt, A)  # (B, L, D, N) — log of dA since A < 0
    dB_x = torch.einsum('bld,bln->bldn', dt * x, B)  # (B, L, D, N)

    # Heinsen parallel recurrence:
    # h[t] = exp(log_dA_cum[t]) * cumsum(dB_x * exp(-log_dA_cum))[t]
    # where log_dA_cum = cumsum(log(dA)) = cumsum(dtA)
    log_dA_cum = torch.cumsum(dtA, dim=1)  # (B, L, D, N)
    h = torch.exp(log_dA_cum) * torch.cumsum(dB_x * torch.exp(-log_dA_cum), dim=1)

    # output: y[t, d] = sum_n C[t, n] * h[t, d, n]
    y = torch.einsum('bldn,bln->bld', h, C)
    return y


class MambaBlock(nn.Module):
  def __init__(self, dim, d_state=16, d_conv=4, expand=2, dropout=0.):
    super().__init__()
    self.dim = dim
    d_inner = int(dim * expand)
    self.norm = nn.LayerNorm(dim)
    self.in_proj = nn.Linear(dim, d_inner * 2, bias=False)
    self.ssm = SelectiveSSM(d_inner, d_state=d_state, d_conv=d_conv)
    self.out_proj = nn.Linear(d_inner, dim, bias=False)
    self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

  def forward(self, x):  # x: (B, L, D)
    residual = x
    x = self.norm(x)
    xz = self.in_proj(x)
    x, z = xz.chunk(2, dim=-1)
    x = self.ssm(x)
    x = x * F.silu(z)  # gated output
    x = self.out_proj(x)
    x = self.dropout(x)
    x = x + residual
    return x


class MambaEncoder(nn.Module):
  def __init__(self, config: configs.pretrain.Config, keep_registers=False, use_sdp_kernel=True):
    super().__init__()
    self.config = config
    self.keep_registers = keep_registers
    assert config.channel_size % config.patch_size == 0
    num_patches = config.channel_size // config.patch_size
    self.patch_embed = PatchEmbedding(
      dim=config.dim,
      in_channels=config.num_channels,
      patch_size=config.patch_size,
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
      MambaBlock(
        dim=config.dim,
        d_state=config.mamba_d_state,
        d_conv=config.mamba_d_conv,
        expand=config.mamba_expand,
        dropout=config.dropout)
      for _ in range(config.depth)
    ])
    self.norm = nn.LayerNorm(config.dim, eps=config.norm_eps, bias=config.bias)

    self.apply(self._init_weights)

  def _init_weights(self, module):
    if isinstance(module, (nn.Linear, nn.Conv1d)):
      nn.init.trunc_normal_(module.weight, mean=0., std=0.02)
      if hasattr(module, 'bias') and module.bias is not None:
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
      nn.init.ones_(module.weight)
      if module.bias is not None:
        nn.init.zeros_(module.bias)

  def forward(self, x, mask=None):
    x = self.patch_embed(x)  # (B, N, D)
    B, N, D = x.size()
    x = x + self.pos_embed[:, :N]
    # Mamba processes full sequence, then mask output
    for block in self.blocks:
      x = block(x)
    x = self.norm(x)
    if mask is not None:
      x = apply_mask(x, mask)
    if self.config.num_registers > 0:
      registers = self.registers.repeat(B, 1, 1)
      x = torch.cat([registers, x], dim=1)
    if not self.keep_registers and self.config.num_registers > 0:
      x = x[:, self.config.num_registers:]
    return x
