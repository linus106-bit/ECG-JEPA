import copy

import torch
from torch import nn

import configs
from models.predictor import Predictor
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

  @torch.compiler.disable()
  def update_momentum_encoder(self):
    m = next(self.momentum_schedule)
    for param_z, param_h in zip(self.encoder.parameters(), self.target_encoder.parameters()):
      param_h.data = m * param_h.data + (1. - m) * param_z.data

  def forward(self, x, mask_encoder, mask_predictor):
    with torch.no_grad():
      self.update_momentum_encoder()
      # compute prediction targets
      h = self.target_encoder(x)
      h = apply_mask(h, mask_predictor)
    # encode unmasked patches
    z = self.encoder(x, mask_encoder)
    # predict masked patches
    z = self.predictor(z, mask_encoder, mask_predictor)
    loss = torch.mean(torch.abs(z - h))
    return loss

  def get_optimizer(self, fused=False):
    from utils.optimizer import build_optimizer
    return build_optimizer(self, self.config, fused=fused)
