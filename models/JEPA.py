import copy
import math
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

import configs
from models.predictor import Predictor
from models.modules import CrossAttention
from models.utils import apply_mask


def create_encoder(config, **kwargs):
  if config.model_type == 'cnn':
    from models.cnn_encoder import CNNEncoder
    return CNNEncoder(config, **kwargs)
  elif config.model_type == 'mamba':
    from models.mamba_encoder import MambaEncoder
    return MambaEncoder(config, **kwargs)
  else:
    from models.vision_transformer import VisionTransformer
    return VisionTransformer(config, **kwargs)


class DiscriminativeTargetHead(nn.Module):
  """DMT-JEPA: generates discriminative targets by aggregating
  semantically similar neighboring patches via cross-attention.
  See: https://arxiv.org/abs/2405.17995"""

  def __init__(self, dim, num_heads, window_size, num_neighbors,
               bias=True, eps=1e-6, use_sdp_kernel=True):
    super().__init__()
    self.window_size = window_size
    self.num_neighbors = num_neighbors
    self.norm = nn.LayerNorm(dim, eps=eps, bias=bias)
    self.xattn = CrossAttention(
      dim, num_heads, qkv_bias=False, bias=bias,
      use_sdp_kernel=use_sdp_kernel)

    for module in self.modules():
      if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, mean=0., std=0.02)
        if module.bias is not None:
          nn.init.zeros_(module.bias)
      elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        if module.bias is not None:
          nn.init.zeros_(module.bias)

  def forward(self, h_full, mask_predictor):
    """
    h_full: (B, N, D) all patch representations from target encoder
    mask_predictor: (B, K) indices of masked patches
    Returns: (B, K, D) discriminative targets
    """
    B, N, D = h_full.shape
    K = mask_predictor.shape[1]
    W = self.window_size
    num_nb = min(self.num_neighbors, 2 * W)

    # Get masked patch representations as queries
    h_masked = apply_mask(h_full, mask_predictor)  # (B, K, D)

    # Build neighbor positions: offsets [-W, ..., -1, 1, ..., W]
    offsets = torch.arange(-W, W + 1, device=mask_predictor.device)
    offsets = offsets[offsets != 0]  # exclude self, shape (2W,)

    # (B, K, 2W) neighbor position indices
    positions = mask_predictor.unsqueeze(-1) + offsets.unsqueeze(0).unsqueeze(0)
    positions = positions.clamp(0, N - 1)

    # Gather neighbor representations from target encoder output
    pos_flat = positions.reshape(B, -1)  # (B, K*2W)
    nb_flat = apply_mask(h_full, pos_flat)  # (B, K*2W, D)
    neighbors = nb_flat.reshape(B, K, 2 * W, D)  # (B, K, 2W, D)

    # Cosine similarity for semantic neighbor selection
    h_norm = F.normalize(h_masked, dim=-1)
    nb_norm = F.normalize(neighbors, dim=-1)
    sim = torch.einsum('bkd,bkwd->bkw', h_norm, nb_norm)  # (B, K, 2W)

    # Select top-K most semantically similar neighbors
    _, topk_idx = sim.topk(num_nb, dim=-1)  # (B, K, num_nb)
    topk_idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, -1, D)
    selected = torch.gather(neighbors, dim=2, index=topk_idx_exp)  # (B, K, num_nb, D)

    # Aggregate via cross-attention: query=masked patch, kv=selected neighbors
    query = h_masked.reshape(B * K, 1, D)
    context = self.norm(selected.reshape(B * K, num_nb, D))
    agg = self.xattn(query, context)  # (B*K, 1, D)

    return agg.reshape(B, K, D)


class JEPA(nn.Module):
  def __init__(self, config: configs.pretrain.Config, momentum_schedule, use_sdp_kernel=True):
    super().__init__()
    self.config = config
    self.momentum_schedule = momentum_schedule
    self.encoder = create_encoder(config, use_sdp_kernel=use_sdp_kernel)
    self.predictor = Predictor(config, use_sdp_kernel=use_sdp_kernel)
    self.target_encoder = copy.deepcopy(self.encoder)

    for param in self.target_encoder.parameters():
      param.requires_grad = False

    # DMT-JEPA: discriminative target head
    self.dmt_head = None
    if config.dmt_window_size > 0:
      self.dmt_head = DiscriminativeTargetHead(
        dim=config.dim,
        num_heads=config.num_heads,
        window_size=config.dmt_window_size,
        num_neighbors=config.dmt_num_neighbors,
        bias=config.bias,
        eps=config.norm_eps,
        use_sdp_kernel=use_sdp_kernel)

  @torch.compiler.disable()
  def update_momentum_encoder(self):
    m = next(self.momentum_schedule)
    for param_z, param_h in zip(self.encoder.parameters(), self.target_encoder.parameters()):
      param_h.data = m * param_h.data + (1. - m) * param_z.data

  def forward(self, x, mask_encoder, mask_predictor):
    with torch.no_grad():
      self.update_momentum_encoder()
      # compute target encoder representations
      h = self.target_encoder(x)
    if self.dmt_head is not None:
      # DMT: aggregate semantically similar neighbors into discriminative targets
      h = self.dmt_head(h, mask_predictor)
    else:
      h = apply_mask(h, mask_predictor)
    # encode unmasked patches
    z = self.encoder(x, mask_encoder)
    # predict masked patches
    z = self.predictor(z, mask_encoder, mask_predictor)
    # target representation normalization (collapse 방지)
    if self.config.target_norm == 'layer_norm':
      h = F.layer_norm(h, (h.size(-1),))
      z = F.layer_norm(z, (z.size(-1),))
    elif self.config.target_norm == 'instance_norm':
      h = (h - h.mean(dim=-1, keepdim=True)) / (h.std(dim=-1, keepdim=True) + 1e-6)
      z = (z - z.mean(dim=-1, keepdim=True)) / (z.std(dim=-1, keepdim=True) + 1e-6)
    loss = torch.mean(torch.abs(z - h))
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
