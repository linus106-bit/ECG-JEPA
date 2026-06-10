#!/usr/bin/env python3
"""Export a trained ECG-JEPA encoder or classifier checkpoint to ONNX.

This exporter accepts full pre-training checkpoints, minified target-encoder
checkpoints, and fine-tuned classifier checkpoints.  Fine-tuned checkpoints can
be exported as the full classifier graph so the ONNX model returns label logits
and probabilities directly.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

# Make this file runnable both from the repository root and from another CWD.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

import configs
from models import EncoderClassifier, create_encoder

DEFAULT_SIX_LABEL_NAMES = ('AFIB', '1AVB', '2AVB', 'SVTAC', 'PAC', 'PVC')
DEFAULT_SIX_LABEL_THRESHOLDS = {
  'AFIB': 0.473,
  '1AVB': 0.131,
  '2AVB': 0.009,
  'SVTAC': 0.012,
  'PAC': 0.251,
  'PVC': 0.028,
}


class ClassifierWithProbabilities(nn.Module):
  def __init__(self, classifier: EncoderClassifier, probability_mode: str):
    super().__init__()
    self.classifier = classifier
    self.probability_mode = probability_mode

  def forward(self, x):
    logits = self.classifier(x)
    if self.probability_mode == 'softmax':
      probabilities = torch.softmax(logits, dim=1)
    else:
      probabilities = torch.sigmoid(logits)
    return logits, probabilities


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


def _as_eval_config(raw_config: Any) -> configs.eval.Config:
  return configs.eval.Config(**_as_config_dict(raw_config))


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


def _preprocess_metadata(chkpt: dict[str, Any]) -> dict[str, list[float]] | None:
  preprocess = chkpt.get('preprocess')
  if not preprocess or 'mean' not in preprocess or 'std' not in preprocess:
    return None
  mean = preprocess['mean']
  std = preprocess['std']
  if isinstance(mean, torch.Tensor):
    mean = mean.detach().cpu().numpy()
  if isinstance(std, torch.Tensor):
    std = std.detach().cpu().numpy()
  return {
    'mean': [float(value) for value in mean.reshape(-1)],
    'std': [float(value) for value in std.reshape(-1)],
    'clip': [-5.0, 5.0],
  }


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


def _default_thresholds() -> dict[str, float]:
  return dict(DEFAULT_SIX_LABEL_THRESHOLDS)


def _eval_crop_config(eval_config: configs.eval.Config | None, sampling_frequency: int) -> tuple[int | None, int | None]:
  if eval_config is None or eval_config.crop_duration is None:
    return None, None
  crop_size = int(eval_config.crop_duration * sampling_frequency)
  if eval_config.crop_stride is not None:
    crop_stride = int(eval_config.crop_stride * sampling_frequency)
  else:
    crop_stride = crop_size
  return crop_size, crop_stride

def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description='Export a trained 500 Hz ECG-JEPA encoder or classifier to ONNX.')
  parser.add_argument('--checkpoint', required=True, type=Path,
                      help='Path to a full/minified pre-train checkpoint or fine-tuned checkpoint.')
  parser.add_argument('--output', required=True, type=Path,
                      help='Output ONNX file path.')
  parser.add_argument('--export-target', choices=('auto', 'encoder', 'classifier'), default='auto',
                      help='What to export. auto exports classifier for fine-tuned checkpoints, otherwise encoder.')
  parser.add_argument('--opset', type=int, default=17,
                      help='ONNX opset version. Default: 17.')
  parser.add_argument('--device', default='cpu',
                      help='Device used during export. CPU is usually safest for ONNX export.')
  parser.add_argument('--keep-registers', action='store_true',
                      help='Keep register tokens in encoder ONNX output when exporting only the encoder.')
  parser.add_argument('--probability-mode', choices=('sigmoid', 'softmax'), default='sigmoid',
                      help='Classifier probability activation to include in ONNX. Default: sigmoid for multi-label ECG tasks.')
  parser.add_argument('--label-names', default=None,
                      help='Comma-separated classifier label names. Defaults to AFIB,1AVB,2AVB,SVTAC,PAC,PVC for 6-label checkpoints.')
  parser.add_argument('--allow-non-500hz', action='store_true',
                      help='Do not fail if checkpoint config.sampling_frequency is not 500.')
  parser.add_argument('--metadata-output', type=Path, default=None,
                      help='Optional metadata JSON path. Defaults to <output>.json.')
  return parser.parse_args()


def _metadata_from_config(
    checkpoint: Path,
    encoder_config: configs.pretrain.Config,
    export_target: str,
    output_names: list[str],
    model_input_size: int,
    crop_size: int | None,
    crop_stride: int | None,
    keep_registers: bool,
    opset: int,
    probability_mode: str | None = None,
    label_names: list[str] | None = None,
    preprocess: dict[str, list[float]] | None = None,
    thresholds: dict[str, float] | None = None) -> dict[str, Any]:
  return {
    'checkpoint': str(checkpoint),
    'export_target': export_target,
    'sampling_frequency': encoder_config.sampling_frequency,
    'channels': list(encoder_config.active_channels),
    'num_channels': encoder_config.num_channels,
    'channel_size': encoder_config.channel_size,
    'model_input_size': model_input_size,
    'crop_size': crop_size,
    'crop_stride': crop_stride,
    'patch_size': encoder_config.patch_size,
    'num_patches': encoder_config.num_patches,
    'embedding_dim': encoder_config.dim,
    'model_type': encoder_config.model_type,
    'keep_registers': keep_registers,
    'probability_mode': probability_mode,
    'label_names': label_names,
    'thresholds': thresholds,
    'preprocess': preprocess,
    'opset': opset,
    'input_name': 'ecg',
    'output_names': output_names,
    'input_shape': ['batch', encoder_config.num_channels, model_input_size],
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

  has_classifier = 'eval_config' in chkpt
  export_target = 'classifier' if args.export_target == 'auto' and has_classifier else args.export_target
  if export_target == 'auto':
    export_target = 'encoder'
  if export_target == 'classifier' and not has_classifier:
    raise ValueError('--export-target classifier requires a fine-tuned checkpoint with eval_config')

  # Disable scaled_dot_product_attention during export; explicit matmul/softmax
  # attention is more portable across ONNX runtimes and opset versions.
  if export_target == 'classifier':
    eval_config = _as_eval_config(chkpt['eval_config'])
    crop_size, crop_stride = _eval_crop_config(eval_config, encoder_config.sampling_frequency)
    model_input_size = crop_size or encoder_config.channel_size
    encoder = create_encoder(
      config=encoder_config,
      keep_registers=eval_config.use_register,
      use_sdp_kernel=False)
    classifier = EncoderClassifier(encoder, eval_config, use_sdp_kernel=False).to(device)
    incompatible = classifier.load_state_dict(chkpt['model'], strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
      raise RuntimeError(
        'Checkpoint is incompatible with the encoder classifier. '
        f'Missing keys: {incompatible.missing_keys}; unexpected keys: {incompatible.unexpected_keys}')
    model = ClassifierWithProbabilities(classifier, args.probability_mode).to(device).eval()
    output_names = ['logits', 'probabilities']
    dynamic_axes = {
      'ecg': {0: 'batch'},
      'logits': {0: 'batch'},
      'probabilities': {0: 'batch'},
    }
    label_names = _label_names(eval_config.num_classes, args.label_names)
    keep_registers = eval_config.use_register
  else:
    crop_size, crop_stride = None, None
    model_input_size = encoder_config.channel_size
    model = create_encoder(
      config=encoder_config,
      keep_registers=args.keep_registers,
      use_sdp_kernel=False).to(device)
    incompatible = model.load_state_dict(_encoder_state_from_checkpoint(chkpt), strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
      raise RuntimeError(
        'Checkpoint is incompatible with the encoder. '
        f'Missing keys: {incompatible.missing_keys}; unexpected keys: {incompatible.unexpected_keys}')
    model.eval()
    output_names = ['token_embeddings']
    dynamic_axes = {
      'ecg': {0: 'batch'},
      'token_embeddings': {0: 'batch'},
    }
    label_names = None
    keep_registers = args.keep_registers

  dummy_ecg = torch.zeros(
    1,
    encoder_config.num_channels,
    model_input_size,
    dtype=torch.float32,
    device=device)

  args.output.parent.mkdir(parents=True, exist_ok=True)
  torch.onnx.export(
    model,
    dummy_ecg,
    args.output,
    export_params=True,
    opset_version=args.opset,
    do_constant_folding=True,
    input_names=['ecg'],
    output_names=output_names,
    dynamic_axes=dynamic_axes)

  metadata_output = args.metadata_output or args.output.with_suffix(args.output.suffix + '.json')
  metadata_output.parent.mkdir(parents=True, exist_ok=True)
  metadata_output.write_text(
    json.dumps(
      _metadata_from_config(
        args.checkpoint,
        encoder_config,
        export_target,
        output_names,
        model_input_size,
        crop_size,
        crop_stride,
        keep_registers,
        args.opset,
        probability_mode=args.probability_mode if export_target == 'classifier' else None,
        label_names=label_names,
        preprocess=_preprocess_metadata(chkpt),
        thresholds=_default_thresholds() if export_target == 'classifier' else None),
      indent=2),
    encoding='utf-8')
  print(f'Exported ONNX {export_target} to {args.output}')
  print(f'Wrote metadata to {metadata_output}')


if __name__ == '__main__':
  main()
