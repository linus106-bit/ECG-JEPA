"""
LeJEPA: JEPA without EMA/stop-gradient, using SIGReg for collapse prevention.
- Single shared encoder (2-pass: full → z_target, masked → z_context)
- Predictor predicts masked patch latents from visible patch latents
- Loss = MSE(z_pred, z_target.detach()) + λ * SIGReg(z_all)
"""

from collections import OrderedDict

import torch
from torch import nn

import configs
from models.JEPA import create_encoder
from models.predictor import Predictor
from models.utils import apply_mask


class SIGReg(nn.Module):
  """Sketched Isotropic Gaussian Regularizer.
  Prevents representation collapse by enforcing N(0,I) distribution.
  Input: (B, N, D) — B: batch, N: patches, D: hidden dim
  """
  def __init__(self, knots=17, num_proj=1024):
    super().__init__()
    self.num_proj = num_proj
    t = torch.linspace(0, 3, knots, dtype=torch.float32)
    dt = 3 / (knots - 1)
    weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
    weights[[0, -1]] = dt
    window = torch.exp(-t.square() / 2.0)
    self.register_buffer("t", t)
    self.register_buffer("phi", window)
    self.register_buffer("weights", weights * window)

  def forward(self, x):
    # x: (B, N, D) → reshape to (B*N, D) for projection
    B, N, D = x.shape
    x_flat = x.reshape(-1, D)  # (B*N, D)
    A = torch.randn(D, self.num_proj, device=x.device)
    A = A.div_(A.norm(p=2, dim=0))
    x_t = (x_flat @ A).unsqueeze(-1) * self.t  # (B*N, num_proj, knots)
    err = (x_t.cos().mean(0) - self.phi).square() + x_t.sin().mean(0).square()
    return (err @ self.weights).mean()


class LeJEPA(nn.Module):
  def __init__(self, config: configs.pretrain.Config, use_sdp_kernel=True):
    super().__init__()
    self.config = config
    self.encoder = create_encoder(config, use_sdp_kernel=use_sdp_kernel)
    self.predictor = Predictor(config, use_sdp_kernel=use_sdp_kernel)
    self.sigreg = SIGReg(num_proj=config.sigreg_num_slices)
    self.sigreg_lambda = config.sigreg_lambda

  def forward(self, x, mask_encoder, mask_predictor):
    # Pass 1: full encoding (all patches) → target
    z_all = self.encoder(x)  # (B, N, D) — no masking
    z_target = apply_mask(z_all, mask_predictor).detach()  # (B, M, D), stop gradient

    # Pass 2: masked encoding (visible patches only) → context
    z_context = self.encoder(x, mask_encoder)  # (B, K, D)

    # Predictor: predict masked patches from visible
    z_pred = self.predictor(z_context, mask_encoder, mask_predictor)  # (B, M, D)

    # Loss
    loss_pred = torch.mean((z_pred - z_target).square())  # MSE
    loss_sigreg = self.sigreg(z_all)

    loss = loss_pred + self.sigreg_lambda * loss_sigreg
    return loss

  def get_optimizer(self, fused=False):
    decay_modules = (nn.Linear, nn.Conv1d)
    decay = set()
    for module_name, module in self.named_modules():
      for param_name, param in module.named_parameters():
        if isinstance(module, decay_modules) and param_name.endswith('weight') and param.requires_grad:
          param_name = f'{module_name}.{param_name}' if module_name else param_name
          decay.add(param_name)

    decay_params, non_decay_params = OrderedDict(), OrderedDict()
    for name, param in self.named_parameters():
      if param.requires_grad:
        if name in decay:
          decay_params[name] = param
        else:
          non_decay_params[name] = param

    param_groups = [
      {'params': list(decay_params.values()),
       'weight_decay': self.config.weight_decay,
       'use_weight_decay': True},
      {'params': list(non_decay_params.values()),
       'weight_decay': 0.,
       'use_weight_decay': False}
    ]

    optimizer = torch.optim.AdamW(
      param_groups,
      lr=self.config.learning_rate,
      betas=self.config.opt_betas,
      eps=self.config.opt_eps,
      weight_decay=0.,
      fused=fused)

    return optimizer
