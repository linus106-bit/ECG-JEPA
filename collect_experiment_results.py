#!/usr/bin/env python3
"""
Collect and aggregate all experiment results into a single JSON file.

This script scans all finetune result directories and collects:
- Experiment metadata (config, hyperparameters, dataset)
- Performance metrics (AUC, F1, accuracy)
- Training details (epochs, best step, etc.)

Output: results/ablation_results.json
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import yaml


def load_json(filepath: str) -> Dict:
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_yaml(filepath: str) -> Dict:
    """Load YAML file."""
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)


def extract_config_name(config_path: str) -> str:
    """Extract config name from path."""
    # e.g., configs/pretrain/ViT/ablation/m0.25_b5_p10.yaml -> m0.25_b5_p10
    return Path(config_path).stem


def parse_ablation_params(config_name: str) -> Dict[str, Any]:
    """Parse ablation parameters from config name.

    Expected format: m{momentum}_b{block_size}_p{patch_size}
    e.g., m0.25_b5_p10 -> momentum=0.25, block_size=5, patch_size=10
    """
    parts = config_name.split('_')
    params = {}

    for part in parts:
        if part.startswith('m'):
            params['encoder_momentum'] = float(part[1:])
        elif part.startswith('b'):
            params['min_block_size'] = int(part[1:])
        elif part.startswith('p'):
            params['patch_size'] = int(part[1:])

    return params


def collect_finetune_result(result_dir: str) -> Dict[str, Any]:
    """Collect all information from a finetune result directory."""

    result_path = Path(result_dir)
    if not result_path.exists():
        return None

    # Find all *_eval_results.json files
    eval_files = list(result_path.glob('*_eval_results.json'))
    if not eval_files:
        return None

    results = {}

    for eval_file in eval_files:
        try:
            data = load_json(eval_file)

            # Extract task name from filename
            task_name = eval_file.stem.replace('_eval_results', '')

            # Load pretrain config if available
            pretrain_config = {}
            if 'config_path' in data or 'encoder_path' in data:
                # Try to load pretrain config from checkpoint
                encoder_path = data.get('encoder_path', '')
                if encoder_path and Path(encoder_path).exists():
                    # Check if there's a config file in the same directory
                    pretrain_config_path = Path(encoder_path).parent.parent / 'config.yaml'
                    if pretrain_config_path.exists():
                        pretrain_config = load_yaml(pretrain_config_path)

            # Extract ablation parameters from config name
            config_name = extract_config_name(data.get('config_path', ''))
            ablation_params = parse_ablation_params(config_name)

            results[task_name] = {
                'dataset_type': data.get('dataset_type'),
                'single_label': data.get('single_label', False),
                'best_val_metric': data.get('best_val_metric'),
                'best_epoch_or_step': data.get('best_epoch_or_step'),
                'test_metric': data.get('test_f1') if data.get('single_label') else data.get('test_auc'),
                'test_f1': data.get('test_f1'),
                'test_acc': data.get('test_acc'),
                'test_auc': data.get('test_auc'),
                'val_f1': data.get('val_f1'),
                'val_auc': data.get('val_auc'),
                'timestamp': data.get('timestamp'),
                'out_dir': data.get('out_dir'),
                'config_path': data.get('config_path'),
                'encoder_path': data.get('encoder_path'),
                'ablation_params': ablation_params,
            }

        except Exception as e:
            print(f"Warning: Failed to load {eval_file}: {e}")
            continue

    return results if results else None


def collect_all_results(base_dir: str = 'results/finetune/ablation') -> Dict[str, Any]:
    """Collect all experiment results from the base directory."""

    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Directory not found: {base_dir}")
        return {}

    all_results = {
        'metadata': {
            'collected_at': datetime.now().isoformat(),
            'base_dir': str(base_path.absolute()),
        },
        'experiments': {}
    }

    # Iterate through datasets (ptb-xl, capture24, etc.)
    for dataset_dir in base_path.iterdir():
        if not dataset_dir.is_dir():
            continue

        dataset_name = dataset_dir.name
        all_results['experiments'][dataset_name] = {}

        # Iterate through finetune modes (linear, finetune, 2stage)
        for mode_dir in dataset_dir.iterdir():
            if not mode_dir.is_dir():
                continue

            mode_name = mode_dir.name
            all_results['experiments'][dataset_name][mode_name] = {}

            # Iterate through experiment configs (m0.25_b5_p10, etc.)
            for exp_dir in mode_dir.iterdir():
                if not exp_dir.is_dir():
                    continue

                exp_name = exp_dir.name
                result = collect_finetune_result(exp_dir)

                if result:
                    all_results['experiments'][dataset_name][mode_name][exp_name] = result

    return all_results


def generate_summary_table(results: Dict[str, Any]) -> str:
    """Generate a markdown summary table of all results."""

    lines = []
    lines.append("# Ablation Study Results\n")
    lines.append(f"Collected at: {results['metadata']['collected_at']}\n")

    for dataset_name, modes in results['experiments'].items():
        lines.append(f"\n## Dataset: {dataset_name}\n")

        for mode_name, experiments in modes.items():
            lines.append(f"\n### Finetune Mode: {mode_name}\n")

            # Get all tasks across experiments
            all_tasks = set()
            for exp_results in experiments.values():
                all_tasks.update(exp_results.keys())

            for task in sorted(all_tasks):
                lines.append(f"\n#### Task: {task}\n")

                # Table header
                lines.append("| Config | Momentum | Block Size | Patch Size | ")
                if experiments and task in next(iter(experiments.values())):
                    if 'test_auc' in next(iter(experiments.values()))[task]:
                        lines.append("Val AUC | Test AUC |")
                    else:
                        lines.append("Val F1 | Test F1 | Test Acc |")
                else:
                    lines.append("Val Metric | Test Metric |")
                lines.append(" Best Step |")
                lines.append("\n|--------|----------|------------|------------|")
                if experiments and task in next(iter(experiments.values())):
                    if 'test_auc' in next(iter(experiments.values()))[task]:
                        lines.append("---------|----------|")
                    else:
                        lines.append("--------|----------|----------|")
                else:
                    lines.append("-----------|------------|")
                lines.append("-----------|")

                # Table rows
                for exp_name, exp_results in sorted(experiments.items()):
                    if task not in exp_results:
                        continue

                    task_result = exp_results[task]
                    params = task_result.get('ablation_params', {})

                    row = f"| {exp_name} "
                    row += f"| {params.get('encoder_momentum', 'N/A')} "
                    row += f"| {params.get('min_block_size', 'N/A')} "
                    row += f"| {params.get('patch_size', 'N/A')} "

                    if 'test_auc' in task_result:
                        val_metric = task_result.get('val_auc', 0)
                        test_metric = task_result.get('test_auc', 0)
                        row += f"| {val_metric:.4f} | {test_metric:.4f} |"
                    else:
                        val_metric = task_result.get('val_f1', 0)
                        test_f1 = task_result.get('test_f1', 0)
                        test_acc = task_result.get('test_acc', 0)
                        row += f"| {val_metric:.4f} | {test_f1:.4f} | {test_acc:.4f} |"

                    row += f" {task_result.get('best_epoch_or_step', 'N/A')} |"
                    lines.append(row)

    return '\n'.join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Collect and aggregate experiment results')
    parser.add_argument('--base-dir', default='results/finetune/ablation',
                        help='Base directory containing finetune results')
    parser.add_argument('--output-json', default='results/ablation_results.json',
                        help='Output JSON file path')
    parser.add_argument('--output-markdown', default='results/ablation_results.md',
                        help='Output markdown summary file path')

    args = parser.parse_args()

    print(f"Collecting results from: {args.base_dir}")
    results = collect_all_results(args.base_dir)

    if not results['experiments']:
        print("No results found!")
        return

    # Save JSON
    output_json_path = Path(args.output_json)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved JSON results to: {output_json_path}")

    # Save markdown summary
    markdown = generate_summary_table(results)
    output_md_path = Path(args.output_markdown)
    with open(output_md_path, 'w') as f:
        f.write(markdown)
    print(f"✓ Saved markdown summary to: {output_md_path}")

    # Print summary
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    for dataset_name, modes in results['experiments'].items():
        print(f"\n{dataset_name}:")
        for mode_name, experiments in modes.items():
            print(f"  {mode_name}: {len(experiments)} experiments")
            for exp_name, exp_results in experiments.items():
                tasks = list(exp_results.keys())
                print(f"    - {exp_name}: {len(tasks)} task(s)")


if __name__ == '__main__':
    main()
