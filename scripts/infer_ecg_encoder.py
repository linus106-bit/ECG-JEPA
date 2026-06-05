#!/usr/bin/env python3
"""Run standalone inference with a trained 500 Hz ECG-JEPA encoder.

The script loads an ECG-JEPA pre-training checkpoint (full or minified) or a
fine-tuned checkpoint, prepares channels-first ECG arrays, and writes encoder
patch/token embeddings plus a simple mean-pooled embedding to an ``.npz`` file.

Input examples:
  * ``record.npy`` with shape ``(channels, samples)`` or ``(batch, channels, samples)``
  * ``records.npz`` containing one array selected by ``--input-key``
  * ``tensor.pt`` / ``tensor.pth`` containing a Tensor, ndarray, or a dict key

Typical 500 Hz ECG-JEPA checkpoints are trained on 10-second records, so the
expected shape is often ``(12, 5000)``. Use ``--length-mode center-crop`` or
``--length-mode pad`` when records are longer/shorter than the checkpoint config.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Make this file runnable both from the repository root and from another CWD.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import configs
from models import create_encoder


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description='Standalone inference for a trained 500 Hz ECG-JEPA encoder.')
  parser.add_argument('--checkpoint', required=True, type=Path,
                      help='Path to a full/minified pre-train checkpoint or fine-tuned checkpoint.')
  parser.add_argument('--input', required=True, type=Path,
                      help='Input ECG file: .npy, .npz, .pt, or .pth.')
  parser.add_argument('--output', required=True, type=Path,
                      help='Output .npz path for embeddings and metadata.')
  parser.add_argument('--input-key', default=None,
                      help='Array/dict key for .npz or dict-style .pt inputs. Defaults to the first array/key.')
  parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Torch device. Defaults to cuda when available, otherwise cpu.')
  parser.add_argument('--batch-size', type=int, default=32,
                      help='Inference batch size.')
  parser.add_argument('--length-mode', choices=('strict', 'center-crop', 'left-crop', 'pad'),
                      default='strict',
                      help='How to handle sample length mismatch with checkpoint config.channel_size.')
  parser.add_argument('--normalize', choices=('none', 'per-record'), default='none',
                      help='Optional per-channel z-score normalization for each record.')
  parser.add_argument('--keep-registers', action='store_true',
                      help='Keep ViT/CNN/Mamba register tokens in the output when the encoder has registers.')
  parser.add_argument('--allow-non-500hz', action='store_true',
                      help='Do not fail if checkpoint config.sampling_frequency is not 500.')
  return parser.parse_args()


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


def _load_array(path: Path, input_key: str | None) -> np.ndarray:
  suffix = path.suffix.lower()
  if suffix == '.npy':
    data = np.load(path)
  elif suffix == '.npz':
    archive = np.load(path)
    key = input_key or archive.files[0]
    if key not in archive.files:
      raise KeyError(f'--input-key {key!r} not found in {path}; available keys: {archive.files}')
    data = archive[key]
  elif suffix in {'.pt', '.pth'}:
    loaded = torch.load(path, map_location='cpu')
    if isinstance(loaded, dict):
      key = input_key or next(iter(loaded.keys()))
      if key not in loaded:
        raise KeyError(f'--input-key {key!r} not found in {path}; available keys: {list(loaded.keys())}')
      loaded = loaded[key]
    if isinstance(loaded, torch.Tensor):
      data = loaded.detach().cpu().numpy()
    else:
      data = np.asarray(loaded)
  else:
    raise ValueError(f'Unsupported input suffix {path.suffix!r}; use .npy, .npz, .pt, or .pth')
  return np.asarray(data, dtype=np.float32)


def _ensure_batched_channels_first(x: np.ndarray) -> np.ndarray:
  if x.ndim == 2:
    x = x[None, ...]
  if x.ndim != 3:
    raise ValueError(f'Input must have shape (channels, samples) or (batch, channels, samples), got {x.shape}')
  return x


def _match_channels(x: np.ndarray, expected_channels: int) -> np.ndarray:
  channels = x.shape[1]
  if channels == expected_channels:
    return x
  if channels > expected_channels:
    return x[:, :expected_channels, :]
  raise ValueError(f'Input has {channels} channels, but the checkpoint expects {expected_channels}')


def _match_length(x: np.ndarray, expected_length: int, mode: str) -> np.ndarray:
  length = x.shape[-1]
  if length == expected_length:
    return x
  if mode == 'strict':
    raise ValueError(f'Input has {length} samples, but the checkpoint expects {expected_length}; choose another --length-mode')
  if mode in {'center-crop', 'left-crop'}:
    if length < expected_length:
      raise ValueError(f'Cannot crop {length} samples up to expected length {expected_length}; use --length-mode pad')
    start = 0 if mode == 'left-crop' else (length - expected_length) // 2
    return x[..., start:start + expected_length]
  if mode == 'pad':
    if length > expected_length:
      return x[..., :expected_length]
    pad_width = expected_length - length
    return np.pad(x, ((0, 0), (0, 0), (0, pad_width)), mode='constant')
  raise ValueError(f'Unknown length mode: {mode}')


def _normalize_per_record(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
  mean = x.mean(axis=-1, keepdims=True)
  std = x.std(axis=-1, keepdims=True)
  return (x - mean) / (std + eps)


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

  encoder = create_encoder(
    config=encoder_config,
    keep_registers=args.keep_registers,
    use_sdp_kernel=device.type == 'cuda').to(device)
  incompatible = encoder.load_state_dict(_encoder_state_from_checkpoint(chkpt), strict=False)
  if incompatible.missing_keys or incompatible.unexpected_keys:
    raise RuntimeError(
      'Checkpoint is incompatible with the encoder. '
      f'Missing keys: {incompatible.missing_keys}; unexpected keys: {incompatible.unexpected_keys}')
  encoder.eval()

  x = _load_array(args.input, args.input_key)
  x = _ensure_batched_channels_first(x)
  x = _match_channels(x, encoder_config.num_channels)
  x = _match_length(x, encoder_config.channel_size, args.length_mode)
  if args.normalize == 'per-record':
    x = _normalize_per_record(x)

  token_batches: list[np.ndarray] = []
  with torch.inference_mode():
    for start in range(0, len(x), args.batch_size):
      batch = torch.from_numpy(x[start:start + args.batch_size]).to(device=device, dtype=torch.float32)
      tokens = encoder(batch).detach().cpu().numpy()
      token_batches.append(tokens)

  token_embeddings = np.concatenate(token_batches, axis=0)
  pooled_embeddings = token_embeddings.mean(axis=1)
  metadata = {
    'checkpoint': str(args.checkpoint),
    'input': str(args.input),
    'sampling_frequency': encoder_config.sampling_frequency,
    'channels': list(encoder_config.active_channels),
    'channel_size': encoder_config.channel_size,
    'patch_size': encoder_config.patch_size,
    'num_patches': encoder_config.num_patches,
    'embedding_dim': encoder_config.dim,
    'keep_registers': args.keep_registers,
    'normalize': args.normalize,
    'length_mode': args.length_mode,
  }

  args.output.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    args.output,
    token_embeddings=token_embeddings,
    pooled_embeddings=pooled_embeddings,
    metadata=json.dumps(metadata, ensure_ascii=False))
  print(f'Saved token_embeddings {token_embeddings.shape} and pooled_embeddings {pooled_embeddings.shape} to {args.output}')


if __name__ == '__main__':
  main()
