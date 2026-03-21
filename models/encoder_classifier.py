import torch
from torch import nn

import configs
from models.modules import AttentivePooler


class EncoderClassifier(nn.Module):
  def __init__(self, encoder: nn.Module, config: configs.eval.Config, use_sdp_kernel=True):
    super().__init__()
    self.config = config
    self.encoder = encoder
    if config.attn_pooling:
      self.attn_pool = AttentivePooler(
        dim=encoder.config.dim,
        num_heads=encoder.config.num_heads,
        num_queries=1,
        mlp_ratio=encoder.config.mlp_ratio,
        qkv_bias=encoder.config.qkv_bias,
        bias=encoder.config.bias,
        proj_dim=config.num_classes,
        eps=encoder.config.norm_eps,
        use_sdp_kernel=use_sdp_kernel)
    else:
      assert not config.use_register or (encoder.config.num_registers > 0 and encoder.keep_registers)
      self.fc = nn.Linear(encoder.config.dim, config.num_classes, bias=config.bias)

    for name, module in self.named_modules(memo=set(encoder.modules())):
      if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, mean=0., std=0.02)
        if module.bias is not None:
          nn.init.zeros_(module.bias)
      elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        if module.bias is not None:
          nn.init.zeros_(module.bias)

    if self.config.frozen:
      for parameter in self.encoder.parameters():
        parameter.requires_grad = False

  def encode(self, x):
    if self.config.frozen:
      with torch.no_grad():
        x = self.encoder(x)
    else:
      x = self.encoder(x)
    return x

  def forward(self, x, encoded=False):
    if not encoded:
      x = self.encode(x)
    if self.config.attn_pooling:
      x = self.attn_pool(x).squeeze(dim=1)
    else:
      if self.config.use_register:
        x = x[:, 0]
      else:
        x = x.mean(dim=1)  # global average pooling
      x = self.fc(x)
    return x

  def get_optimizer(self, fused=False):
    from utils.optimizer import build_optimizer
    return build_optimizer(self, self.config, fused=fused)
