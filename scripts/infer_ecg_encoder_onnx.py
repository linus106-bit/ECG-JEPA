#!/usr/bin/env python3
"""Run inference with an ONNX-exported ECG-JEPA encoder.

Use ``scripts/export_ecg_encoder_onnx.py`` first. This script loads the ONNX
model with ONNX Runtime, prepares channels-first ECG arrays, runs batched
inference, and writes ``token_embeddings`` plus mean-pooled embeddings to an
``.npz`` file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description='Inference for an ONNX-exported ECG-JEPA encoder.')
  parser.add_argument('--onnx', required=True, type=Path,
                      help='Path to an ONNX model exported by export_ecg_encoder_onnx.py.')
  parser.add_argument('--input', required=True, type=Path,
                      help='Input ECG file: .npy or .npz with shape (channels, samples) or (batch, channels, samples).')
  parser.add_argument('--output', required=True, type=Path,
                      help='Output .npz path for embeddings and metadata.')
  parser.add_argument('--input-key', default=None,
                      help='Array key for .npz inputs. Defaults to the first array in the archive.')
  parser.add_argument('--metadata', type=Path, default=None,
                      help='Metadata JSON path. Defaults to <onnx>.json when present; otherwise shape is read from ONNX input.')
  parser.add_argument('--batch-size', type=int, default=32,
                      help='Inference batch size.')
  parser.add_argument('--length-mode', choices=('strict', 'center-crop', 'left-crop', 'pad'),
                      default='strict',
                      help='How to handle sample length mismatch with the exported model input length.')
  parser.add_argument('--normalize', choices=('none', 'per-record'), default='none',
                      help='Optional per-channel z-score normalization for each record.')
  parser.add_argument('--providers', nargs='+', default=None,
                      help='Optional ONNX Runtime providers, e.g. CUDAExecutionProvider CPUExecutionProvider.')
  return parser.parse_args()


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
  else:
    raise ValueError(f'Unsupported input suffix {path.suffix!r}; use .npy or .npz')
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
  raise ValueError(f'Input has {channels} channels, but the ONNX model expects {expected_channels}')


def _match_length(x: np.ndarray, expected_length: int, mode: str) -> np.ndarray:
  length = x.shape[-1]
  if length == expected_length:
    return x
  if mode == 'strict':
    raise ValueError(f'Input has {length} samples, but the ONNX model expects {expected_length}; choose another --length-mode')
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


def _static_dim(value: Any, name: str) -> int:
  if isinstance(value, int):
    return value
  raise ValueError(f'ONNX input {name} dimension must be static, got {value!r}')


def _load_metadata(args: argparse.Namespace) -> dict[str, Any]:
  metadata_path = args.metadata or args.onnx.with_suffix(args.onnx.suffix + '.json')
  if metadata_path.exists():
    return json.loads(metadata_path.read_text(encoding='utf-8'))
  return {}


def main() -> None:
  args = parse_args()
  session = ort.InferenceSession(str(args.onnx), providers=args.providers)
  model_input = session.get_inputs()[0]
  input_name = model_input.name
  input_shape = model_input.shape
  metadata = _load_metadata(args)

  expected_channels = int(metadata.get('num_channels') or _static_dim(input_shape[1], 'channel'))
  expected_length = int(metadata.get('channel_size') or _static_dim(input_shape[2], 'sample'))

  x = _load_array(args.input, args.input_key)
  x = _ensure_batched_channels_first(x)
  x = _match_channels(x, expected_channels)
  x = _match_length(x, expected_length, args.length_mode)
  if args.normalize == 'per-record':
    x = _normalize_per_record(x)

  token_batches: list[np.ndarray] = []
  for start in range(0, len(x), args.batch_size):
    batch = np.ascontiguousarray(x[start:start + args.batch_size], dtype=np.float32)
    tokens = session.run(None, {input_name: batch})[0]
    token_batches.append(tokens)

  token_embeddings = np.concatenate(token_batches, axis=0)
  pooled_embeddings = token_embeddings.mean(axis=1)
  output_metadata = {
    **metadata,
    'onnx': str(args.onnx),
    'input': str(args.input),
    'normalize': args.normalize,
    'length_mode': args.length_mode,
    'providers': session.get_providers(),
  }

  args.output.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    args.output,
    token_embeddings=token_embeddings,
    pooled_embeddings=pooled_embeddings,
    metadata=json.dumps(output_metadata, ensure_ascii=False))
  print(f'Saved token_embeddings {token_embeddings.shape} and pooled_embeddings {pooled_embeddings.shape} to {args.output}')


if __name__ == '__main__':
  main()
