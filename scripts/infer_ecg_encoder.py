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
from models import EncoderClassifier, create_encoder


DEFAULT_SIX_LABEL_NAMES = ('AFIB', '1AVB', '2AVB', 'SVTAC', 'PAC', 'PVC')


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
  parser.add_argument('--label-names', default=None,
                      help='Comma-separated label names for printed classifier probabilities. Defaults to AFIB,1AVB,2AVB,SVTAC,PAC,PVC for 6-label checkpoints.')
  parser.add_argument('--num-print-labels', type=int, default=6,
                      help='Number of classifier labels to print per sample. Default: 6.')
  parser.add_argument('--probability-mode', choices=('sigmoid', 'softmax'), default='sigmoid',
                      help='How to convert classifier logits to probabilities. Default: sigmoid for multi-label ECG tasks.')
  parser.add_argument('--no-print-labels', action='store_true',
                      help='Do not print classifier probabilities even when a fine-tuned checkpoint is provided.')
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


def _as_eval_config(raw_config: Any) -> configs.eval.Config:
  eval_config_dict = _as_config_dict(raw_config)
  return configs.eval.Config(**eval_config_dict)


def _label_names(num_classes: int, raw_label_names: str | None) -> list[str]:
  if raw_label_names is None:
    if num_classes == len(DEFAULT_SIX_LABEL_NAMES):
      return list(DEFAULT_SIX_LABEL_NAMES)
    return [f'label_{index}' for index in range(num_classes)]
  names = [name.strip() for name in raw_label_names.split(',') if name.strip()]
  if len(names) > num_classes:
    raise ValueError(f'--label-names cannot contain more than {num_classes} names, got {len(names)}')
  names.extend(f'label_{index}' for index in range(len(names), num_classes))
  return names


def _print_label_probabilities(probabilities: np.ndarray, label_names: list[str], num_print_labels: int) -> None:
  labels_to_print = min(num_print_labels, probabilities.shape[1])
  for sample_index, row in enumerate(probabilities):
    print(f'sample {sample_index} label probabilities:')
    for label_index in range(labels_to_print):
      print(f'  {label_names[label_index]}: {row[label_index]:.6f}')


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

  classifier = None
  actual_keep_registers = args.keep_registers
  if 'eval_config' in chkpt:
    eval_config = _as_eval_config(chkpt['eval_config'])
    actual_keep_registers = args.keep_registers or eval_config.use_register
    encoder = create_encoder(
      config=encoder_config,
      keep_registers=actual_keep_registers,
      use_sdp_kernel=device.type == 'cuda')
    classifier = EncoderClassifier(encoder, eval_config, use_sdp_kernel=device.type == 'cuda').to(device)
    incompatible = classifier.load_state_dict(chkpt['model'], strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
      raise RuntimeError(
        'Checkpoint is incompatible with the encoder classifier. '
        f'Missing keys: {incompatible.missing_keys}; unexpected keys: {incompatible.unexpected_keys}')
    classifier.eval()
    encoder = classifier.encoder
  else:
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
  logit_batches: list[np.ndarray] = []
  with torch.inference_mode():
    for start in range(0, len(x), args.batch_size):
      batch = torch.from_numpy(x[start:start + args.batch_size]).to(device=device, dtype=torch.float32)
      tokens_tensor = encoder(batch)
      token_batches.append(tokens_tensor.detach().cpu().numpy())
      if classifier is not None:
        logits = classifier(tokens_tensor, encoded=True)
        logit_batches.append(logits.detach().cpu().numpy())

  token_embeddings = np.concatenate(token_batches, axis=0)
  pooled_embeddings = token_embeddings.mean(axis=1)
  logits = np.concatenate(logit_batches, axis=0) if logit_batches else None
  probabilities = None
  names = None
  if logits is not None:
    if args.probability_mode == 'softmax':
      logits_max = logits.max(axis=1, keepdims=True)
      exp_logits = np.exp(logits - logits_max)
      probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    else:
      probabilities = 1.0 / (1.0 + np.exp(-logits))
    names = _label_names(probabilities.shape[1], args.label_names)
    if not args.no_print_labels:
      _print_label_probabilities(probabilities, names, args.num_print_labels)
  metadata = {
    'checkpoint': str(args.checkpoint),
    'input': str(args.input),
    'sampling_frequency': encoder_config.sampling_frequency,
    'channels': list(encoder_config.active_channels),
    'channel_size': encoder_config.channel_size,
    'patch_size': encoder_config.patch_size,
    'num_patches': encoder_config.num_patches,
    'embedding_dim': encoder_config.dim,
    'keep_registers': actual_keep_registers,
    'normalize': args.normalize,
    'length_mode': args.length_mode,
    'has_classifier': classifier is not None,
    'probability_mode': args.probability_mode if classifier is not None else None,
    'label_names': names,
  }

  args.output.parent.mkdir(parents=True, exist_ok=True)
  output = {
    'token_embeddings': token_embeddings,
    'pooled_embeddings': pooled_embeddings,
    'metadata': json.dumps(metadata, ensure_ascii=False),
  }
  if logits is not None and probabilities is not None:
    output['logits'] = logits
    output['probabilities'] = probabilities
  np.savez_compressed(args.output, **output)
  print(f'Saved token_embeddings {token_embeddings.shape} and pooled_embeddings {pooled_embeddings.shape} to {args.output}')


if __name__ == '__main__':
  main()
