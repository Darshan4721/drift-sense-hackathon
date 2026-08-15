#!/usr/bin/env python3
"""Create a diagnostic image for one localization result."""

from __future__ import annotations

import argparse

from drift_sense.image_utils import load_grayscale
from drift_sense.visualization import render_result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--search", required=True)
    parser.add_argument("--prediction", nargs=2, type=float, required=True, metavar=("X", "Y"))
    parser.add_argument("--truth", nargs=2, type=float, default=None, metavar=("X", "Y"))
    parser.add_argument("--truth-bbox", nargs=4, type=float, default=None, metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    render_result(
        load_grayscale(args.reference),
        load_grayscale(args.search),
        (args.prediction[0], args.prediction[1]),
        (args.truth[0], args.truth[1]) if args.truth else None,
        truth_bbox=tuple(args.truth_bbox) if args.truth_bbox else None,
        confidence=args.confidence,
        output=args.output,
    )


if __name__ == "__main__":
    main()
