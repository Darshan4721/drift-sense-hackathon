#!/usr/bin/env python3
"""Generate reproducible synthetic Drift-Sense image pairs."""

from __future__ import annotations

import argparse
from dataclasses import fields

from drift_sense.config import AugmentationConfig
from drift_sense.synthetic import write_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=["DRAM", "FinFET"], default="DRAM")
    parser.add_argument("--num-pairs", type=int, default=1000)
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-visualizations", action="store_true")
    defaults = AugmentationConfig()
    for field in fields(AugmentationConfig):
        argument = "--" + field.name.replace("_", "-")
        parser.add_argument(argument, type=float, default=getattr(defaults, field.name), metavar="VALUE")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.num_pairs < 1:
        raise SystemExit("--num-pairs must be positive")
    config = AugmentationConfig(**{field.name: getattr(args, field.name) for field in fields(AugmentationConfig)})
    write_dataset(
        output_dir=args.output_dir,
        num_pairs=args.num_pairs,
        architecture=args.architecture,
        seed=args.seed,
        config=config,
        save_visualizations=args.save_visualizations,
    )


if __name__ == "__main__":
    main()
