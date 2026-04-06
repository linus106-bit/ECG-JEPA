#!/usr/bin/env python3
"""Generate ablation YAML configs from value lists.

Example:
    python scripts/generate_ablation_yamls.py \
        --template configs/pretrain/ViT/ViTS_mimic.yaml \
        --output-dir configs/pretrain/ViT/ablation \
        --min-keep-ratios 0.15 \
        --max-keep-ratios 0.25 0.35 0.45 \
        --min-block-sizes 5 10 12 \
        --patch-sizes 5 10 20 25
"""

from __future__ import annotations

import argparse
import itertools
from copy import deepcopy
from pathlib import Path

import yaml


def _format_value(value: float | int) -> str:
    """Return a filename-safe representation for numeric values."""
    if isinstance(value, int):
        return str(value)

    text = f"{value:.10g}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create YAML files for every combination of max_keep_ratio, "
            "min_block_size, and patch_size."
        )
    )
    parser.add_argument(
        "--template",
        type=Path,
        required=True,
        help="Path to a base YAML file used as template.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where generated YAML files will be written.",
    )
    parser.add_argument(
        "--min-keep-ratios",
        type=float,
        nargs="+",
        required=True,
        help="List of max_keep_ratio values.",
    )
    parser.add_argument(
        "--max-keep-ratios",
        type=float,
        nargs="+",
        required=True,
        help="List of max_keep_ratio values.",
    )
    parser.add_argument(
        "--min-block-sizes",
        type=int,
        nargs="+",
        required=True,
        help="List of min_block_size values.",
    )
    parser.add_argument(
        "--patch-sizes",
        type=int,
        nargs="+",
        required=True,
        help="List of patch_size values.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with args.template.open("r", encoding="utf-8") as f:
        template = yaml.safe_load(f)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for min_keep_ratio, max_keep_ratio, min_block_size, patch_size in itertools.product(
        args.min_keep_ratios,
        args.max_keep_ratios,
        args.min_block_sizes,
        args.patch_sizes,
    ):
        config = deepcopy(template)
        config["min_keep_ratio"] = min_keep_ratio
        config["max_keep_ratio"] = max_keep_ratio
        config["min_block_size"] = min_block_size
        config["patch_size"] = patch_size

        filename = (
            f"m{_format_value(min_keep_ratio)}_{_format_value(max_keep_ratio)}"
            f"_b{_format_value(min_block_size)}"
            f"_p{_format_value(patch_size)}.yaml"
        )
        yaml_name = Path(filename).stem
        run_config = config.get("run")
        if not isinstance(run_config, dict):
            run_config = {}
        run_config["out_dir"] = f"results/pretrain/{yaml_name}"
        config["run"] = run_config

        output_path = args.output_dir / filename

        if output_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"{output_path} already exists. "
                "Use --overwrite to replace existing files."
            )

        with output_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
        count += 1

    print(f"Generated {count} YAML files in {args.output_dir}")


if __name__ == "__main__":
    main()
