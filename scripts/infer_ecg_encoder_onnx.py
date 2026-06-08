#!/usr/bin/env python3
"""Run inference with an ONNX-exported ECG-JEPA encoder or classifier.

Use ``scripts/export_ecg_encoder_onnx.py`` first. This script loads the ONNX
model with ONNX Runtime, prepares channels-first ECG arrays, runs batched
inference, and writes the ONNX outputs to an ``.npz`` file. Encoder ONNX models
produce ``token_embeddings`` plus mean-pooled embeddings; classifier ONNX models
produce ``logits`` and ``probabilities``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort


DEFAULT_SIX_LABEL_NAMES = ('AFIB', '1AVB', '2AVB', 'SVTAC', 'PAC', 'PVC')


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description='Inference for an ONNX-exported ECG-JEPA encoder or classifier.')
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
  parser.add_argument('--label-names', default=None,
                      help='Comma-separated classifier label names. Defaults to metadata, then AFIB,1AVB,2AVB,SVTAC,PAC,PVC for 6-label models.')
  parser.add_argument('--num-print-labels', type=int, default=6,
                      help='Number of classifier probabilities to print per sample. Default: 6.')
  parser.add_argument('--no-print-labels', action='store_true',
                      help='Do not print classifier probabilities when the ONNX model returns probabilities.')
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


def _label_names(num_classes: int, raw_label_names: str | None, metadata: dict[str, Any]) -> list[str]:
  if raw_label_names is not None:
    names = [name.strip() for name in raw_label_names.split(',') if name.strip()]
    if len(names) > num_classes:
      raise ValueError(f'--label-names cannot contain more than {num_classes} names, got {len(names)}')
    names.extend(f'label_{index}' for index in range(len(names), num_classes))
    return names
  metadata_names = metadata.get('label_names')
  if metadata_names:
    names = [str(name) for name in metadata_names]
    names.extend(f'label_{index}' for index in range(len(names), num_classes))
    return names[:num_classes]
  if num_classes == len(DEFAULT_SIX_LABEL_NAMES):
    return list(DEFAULT_SIX_LABEL_NAMES)
  return [f'label_{index}' for index in range(num_classes)]


def _print_label_probabilities(probabilities: np.ndarray, label_names: list[str], num_print_labels: int) -> None:
  labels_to_print = min(num_print_labels, probabilities.shape[1])
  for sample_index, row in enumerate(probabilities):
    print(f'sample {sample_index} label probabilities:')
    for label_index in range(labels_to_print):
      print(f'  {label_names[label_index]}: {row[label_index]:.6f}')


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

  output_names = metadata.get('output_names') or ([metadata['output_name']] if metadata.get('output_name') else None)
  output_names = output_names or [output.name for output in session.get_outputs()]
  output_batches: dict[str, list[np.ndarray]] = {name: [] for name in output_names}
  for start in range(0, len(x), args.batch_size):
    batch = np.ascontiguousarray(x[start:start + args.batch_size], dtype=np.float32)
    outputs = session.run(None, {input_name: batch})
    for name, values in zip(output_names, outputs):
      output_batches.setdefault(name, []).append(values)

  outputs_np = {
    name: np.concatenate(batches, axis=0)
    for name, batches in output_batches.items()
    if batches
  }
  token_embeddings = outputs_np.get('token_embeddings')
  logits = outputs_np.get('logits')
  probabilities = outputs_np.get('probabilities')
  pooled_embeddings = token_embeddings.mean(axis=1) if token_embeddings is not None else None

  label_names = None
  if probabilities is not None:
    label_names = _label_names(probabilities.shape[1], args.label_names, metadata)
    if not args.no_print_labels:
      _print_label_probabilities(probabilities, label_names, args.num_print_labels)

  output_metadata = {
    **metadata,
    'onnx': str(args.onnx),
    'input': str(args.input),
    'normalize': args.normalize,
    'length_mode': args.length_mode,
    'providers': session.get_providers(),
    'label_names': label_names or metadata.get('label_names'),
  }

  args.output.parent.mkdir(parents=True, exist_ok=True)
  output_npz = {'metadata': json.dumps(output_metadata, ensure_ascii=False), **outputs_np}
  if pooled_embeddings is not None:
    output_npz['pooled_embeddings'] = pooled_embeddings
  np.savez_compressed(args.output, **output_npz)
  saved_shapes = ', '.join(f'{name} {value.shape}' for name, value in output_npz.items() if name != 'metadata')
  print(f'Saved {saved_shapes} to {args.output}')


if __name__ == '__main__':
  main()
