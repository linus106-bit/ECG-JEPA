#!/usr/bin/env python3
"""Export a trained ECG-JEPA encoder checkpoint to ONNX.

This exporter accepts the same checkpoint families as ``infer_ecg_encoder.py``:
full pre-training checkpoints, minified target-encoder checkpoints, and
fine-tuned checkpoints that contain an ``encoder.`` state dict prefix.

The exported ONNX graph takes one input named ``ecg`` with shape
``(batch, channels, samples)`` and returns one output named
``token_embeddings`` with shape ``(batch, tokens, embedding_dim)``.
Only the batch dimension is dynamic; the channel count and sample length are
fixed by the checkpoint config because ECG-JEPA positional embeddings are
created for the training-time signal length.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import torch

# Make this file runnable both from the repository root and from another CWD.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import configs
from models import create_encoder


def _as_config_dict(raw_config: Any) -> dict[str, Any]:
  if dataclasses.is_dataclass(raw_config):
    config_dict = dataclasses.asdict(raw_config)
  elif isinstance(raw_config, configs.pretrain.Config):
    config_dict = dataclasses.asdict(raw_config)
  elif isinstance(raw_config, dict):
    config_dict = dict(raw_config)
  else:
    raise TypeError(f'Unsupported checkpoint config type: {type(raw_config)!r}')
  config_dict.pop('run', None)
  return config_dict


def _encoder_state_from_checkpoint(chkpt: dict[str, Any]) -> dict[str, torch.Tensor]:
  if 'model' not in chkpt:
    raise KeyError('Checkpoint must contain a "model" state dict')
  model_state = chkpt['model']

  if any(key.startswith('target_encoder.') for key in model_state):
    return {
      key.removeprefix('target_encoder.'): value
      for key, value in model_state.items()
      if key.startswith('target_encoder.')
    }
  if any(key.startswith('encoder.') for key in model_state):
    return {
      key.removeprefix('encoder.'): value
      for key, value in model_state.items()
      if key.startswith('encoder.')
    }
  return dict(model_state)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description='Export a trained 500 Hz ECG-JEPA encoder checkpoint to ONNX.')
  parser.add_argument('--checkpoint', required=True, type=Path,
                      help='Path to a full/minified pre-train checkpoint or fine-tuned checkpoint.')
  parser.add_argument('--output', required=True, type=Path,
                      help='Output ONNX file path.')
  parser.add_argument('--opset', type=int, default=17,
                      help='ONNX opset version. Default: 17.')
  parser.add_argument('--device', default='cpu',
                      help='Device used during export. CPU is usually safest for ONNX export.')
  parser.add_argument('--keep-registers', action='store_true',
                      help='Keep register tokens in the exported encoder output when the encoder has registers.')
  parser.add_argument('--allow-non-500hz', action='store_true',
                      help='Do not fail if checkpoint config.sampling_frequency is not 500.')
  parser.add_argument('--metadata-output', type=Path, default=None,
                      help='Optional metadata JSON path. Defaults to <output>.json.')
  return parser.parse_args()


def _metadata_from_config(
    checkpoint: Path,
    encoder_config: configs.pretrain.Config,
    keep_registers: bool,
    opset: int) -> dict[str, Any]:
  return {
    'checkpoint': str(checkpoint),
    'sampling_frequency': encoder_config.sampling_frequency,
    'channels': list(encoder_config.active_channels),
    'num_channels': encoder_config.num_channels,
    'channel_size': encoder_config.channel_size,
    'patch_size': encoder_config.patch_size,
    'num_patches': encoder_config.num_patches,
    'embedding_dim': encoder_config.dim,
    'model_type': encoder_config.model_type,
    'keep_registers': keep_registers,
    'opset': opset,
    'input_name': 'ecg',
    'output_name': 'token_embeddings',
    'input_shape': ['batch', encoder_config.num_channels, encoder_config.channel_size],
  }


def main() -> None:
  args = parse_args()
  device = torch.device(args.device)

  chkpt = torch.load(args.checkpoint, map_location='cpu')
  if 'config' not in chkpt:
    raise KeyError('Checkpoint must contain a "config" entry')
  config_dict = _as_config_dict(chkpt['config'])
  encoder_config = configs.pretrain.Config(**config_dict)

  if encoder_config.sampling_frequency != 500 and not args.allow_non_500hz:
    raise ValueError(
      f'Checkpoint sampling_frequency is {encoder_config.sampling_frequency}, not 500. '
      'Use --allow-non-500hz only if this is intentional.')

  # Disable scaled_dot_product_attention during export; explicit matmul/softmax
  # attention is more portable across ONNX runtimes and opset versions.
  encoder = create_encoder(
    config=encoder_config,
    keep_registers=args.keep_registers,
    use_sdp_kernel=False).to(device)
  incompatible = encoder.load_state_dict(_encoder_state_from_checkpoint(chkpt), strict=False)
  if incompatible.missing_keys or incompatible.unexpected_keys:
    raise RuntimeError(
      'Checkpoint is incompatible with the encoder. '
      f'Missing keys: {incompatible.missing_keys}; unexpected keys: {incompatible.unexpected_keys}')
  encoder.eval()

  dummy_ecg = torch.zeros(
    1,
    encoder_config.num_channels,
    encoder_config.channel_size,
    dtype=torch.float32,
    device=device)

  args.output.parent.mkdir(parents=True, exist_ok=True)
  torch.onnx.export(
    encoder,
    dummy_ecg,
    args.output,
    export_params=True,
    opset_version=args.opset,
    do_constant_folding=True,
    input_names=['ecg'],
    output_names=['token_embeddings'],
    dynamic_axes={
      'ecg': {0: 'batch'},
      'token_embeddings': {0: 'batch'},
    })

  metadata_output = args.metadata_output or args.output.with_suffix(args.output.suffix + '.json')
  metadata_output.parent.mkdir(parents=True, exist_ok=True)
  metadata_output.write_text(
    json.dumps(_metadata_from_config(args.checkpoint, encoder_config, args.keep_registers, args.opset), indent=2),
    encoding='utf-8')
  print(f'Exported ONNX encoder to {args.output}')
  print(f'Wrote metadata to {metadata_output}')


if __name__ == '__main__':
  main()
