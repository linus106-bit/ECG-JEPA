from collections import OrderedDict

import torch
from torch import nn


def build_optimizer(model, config, fused=False):
  decay_modules = (nn.Linear, nn.Conv1d)
  decay = set()
  for module_name, module in model.named_modules():
    for param_name, param in module.named_parameters():
      if isinstance(module, decay_modules) and param_name.endswith('weight') and param.requires_grad:
        param_name = f'{module_name}.{param_name}' if module_name else param_name
        decay.add(param_name)

  decay_params, non_decay_params = OrderedDict(), OrderedDict()
  for name, param in model.named_parameters():
    if param.requires_grad:
      if name in decay:
        decay_params[name] = param
      else:
        non_decay_params[name] = param

  param_groups = [
    {'params': list(decay_params.values()),
     'weight_decay': config.weight_decay,
     'use_weight_decay': True},
    {'params': list(non_decay_params.values()),
     'weight_decay': 0.,
     'use_weight_decay': False}
  ]

  kwargs = {}
  opt_eps = getattr(config, 'opt_eps', None)
  if opt_eps is not None:
    kwargs['eps'] = opt_eps

  optimizer = torch.optim.AdamW(
    param_groups,
    lr=config.learning_rate,
    betas=config.opt_betas,
    weight_decay=0.,
    fused=fused,
    **kwargs)

  return optimizer
