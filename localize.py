#!/usr/bin/env python3
"""Locate a 0.1-scale reference pattern inside a 1000x1000 search image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from drift_sense.classical import Candidate, generate_candidates, rerank_candidates
from drift_sense.config import SEARCH_SCALES
from drift_sense.image_utils import load_grayscale
from drift_sense.model import load_verifier, prepare_pair


def _default_checkpoint() -> Path:
    return Path(__file__).resolve().parent / "checkpoints" / "best.pt"


def _verify_candidates(reference: np.ndarray, search: np.ndarray, candidates: list[Candidate], checkpoint: str | Path) -> tuple[dict[int, float], dict[str, Any]]:
    """Score candidate crops with the learned verifier when valid weights exist."""

    model, details = load_verifier(checkpoint)
    if model is None:
        return {}, details
    try:
        import torch

        scores: dict[int, float] = {}
        batch_size = 64
        pairs = [prepare_pair(reference, search, candidate.x, candidate.y) for candidate in candidates]
        with torch.no_grad():
            for start in range(0, len(pairs), batch_size):
                batch = torch.from_numpy(np.stack(pairs[start : start + batch_size], axis=0))
                logits = model(batch)
                probabilities = torch.sigmoid(logits).cpu().numpy()
                for offset, probability in enumerate(probabilities):
                    scores[start + offset] = float(probability * 2.0 - 1.0)
        return scores, details
    except (RuntimeError, ValueError, OSError) as exc:
        return {}, {"reason": "verifier_runtime_error", "error": str(exc)}


def compute_calibrated_confidence(best: Candidate, ranked: list[Candidate], dist_threshold: float = 15.0) -> dict[str, Any]:
    """Compute calibrated confidence incorporating spatial margin and peak-to-sidelobe ratio."""
    s1 = float(best.combined_score)
    s2_dist = None
    for c in ranked[1:]:
        if np.hypot(c.x - best.x, c.y - best.y) > dist_threshold:
            s2_dist = float(c.combined_score)
            break
    if s2_dist is None:
        s2_dist = float(ranked[-1].combined_score) if len(ranked) > 1 else s1 - 0.20

    margin_dist = float(s1 - s2_dist)
    sidelobes = [float(c.combined_score) for c in ranked[1:] if np.hypot(c.x - best.x, c.y - best.y) > dist_threshold]
    if sidelobes:
        mu = float(np.mean(sidelobes))
        sigma = float(np.std(sidelobes)) + 1e-6
        psr = float((s1 - mu) / sigma)
    else:
        psr = 10.0

    # Calibrated on separate 100-sample tuning set (seed 500000):
    # Flags ambiguous periodic patterns or weak matches
    is_low_confidence = bool((margin_dist < 0.020) or (s1 < 0.60) or (psr < 1.60))

    if is_low_confidence:
        # Constrain low-confidence scores below the 0.50 acceptance threshold
        conf = 0.10 + 0.38 * min(1.0, max(0.0, margin_dist / 0.020)) * min(1.0, max(0.0, s1 / 0.70))
    else:
        # Confident scores scale gracefully between 0.55 and 0.98
        conf = 0.55 + 0.43 * (0.5 * min(1.0, max(0.0, (margin_dist - 0.020) / 0.080)) + 0.5 * min(1.0, max(0.0, (s1 - 0.60) / 0.30)))

    return {
        "confidence": round(float(np.clip(conf, 0.0, 1.0)), 6),
        "low_confidence": is_low_confidence,
        "margin_distant": round(margin_dist, 6),
        "psr": round(psr, 4),
    }


def localize(
    reference_path: str | Path,
    search_path: str | Path,
    method: str = "hybrid",
    checkpoint: str | Path | None = None,
    verbose: bool = False,
    prefilter: str = "none",
    candidate_pool_size: int = 16,
) -> tuple[Candidate, list[Candidate], dict[str, Any]]:
    """Run localization without reading the training dataset or using the network."""

    reference = load_grayscale(reference_path)
    search = load_grayscale(search_path)
    if reference.shape != (1000, 1000) or search.shape != (1000, 1000):
        raise ValueError("Both reference and search images must be exactly 1000x1000 pixels")
    if method not in {"classical", "deep", "hybrid"}:
        raise ValueError("method must be classical, deep, or hybrid")
    candidates = generate_candidates(
        reference,
        search,
        scales=(0.098, 0.100, 0.102),
        top_k_per_scale=12,
        rotations=(-3.0, 0.0, 3.0),
        prefilter=prefilter,
    )
    if not candidates:
        raise RuntimeError("No candidate locations found")
    checkpoint_path = Path(checkpoint) if checkpoint else _default_checkpoint()
    verifier_details: dict[str, Any] = {"reason": "not_requested"}
    verifier_scores: dict[int, float] = {}
    classical_ranked = rerank_candidates(candidates, search.shape)
    if method in {"deep", "hybrid"}:
        verification_candidates = classical_ranked[:candidate_pool_size]
        local_scores, verifier_details = _verify_candidates(reference, search, verification_candidates, checkpoint_path)
        original_indices = {id(candidate): index for index, candidate in enumerate(candidates)}
        verifier_scores = {
            original_indices[id(verification_candidates[local_index])]: score
            for local_index, score in local_scores.items()
        }
    ranked = rerank_candidates(candidates, search.shape, verifier_scores if verifier_scores else None)
    if verifier_scores and method == "hybrid":
        classical_best = classical_ranked[0]
        learned_best = ranked[0]
        if learned_best.classical_score + 0.02 < classical_best.classical_score:
            ranked = classical_ranked
            verifier_details = {**verifier_details, "arbitration": "classical_guardrail"}
        else:
            verifier_details = {**verifier_details, "arbitration": "hybrid"}
    best = ranked[0]
    calibration = compute_calibrated_confidence(best, ranked)
    details = {
        "method": method,
        "prefilter": prefilter,
        "candidate_pool_size": candidate_pool_size,
        "checkpoint": str(checkpoint_path),
        "verifier": verifier_details,
        "candidate_count": len(ranked),
        "confidence_calibration": calibration,
        "top_candidates": [candidate.to_dict() for candidate in ranked[:5]],
    }
    if verbose:
        print(json.dumps(details, indent=2), file=sys.stderr)
    return best, ranked, details


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--search", required=True)
    parser.add_argument("--method", choices=["hybrid", "classical", "deep"], default="hybrid")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--prefilter", choices=["none", "dog", "log", "lcn"], default="none")
    parser.add_argument("--candidate-pool", type=int, default=16)
    parser.add_argument("--output-format", choices=["coordinate", "json"], default="coordinate")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        best, ranked, details = localize(
            args.reference,
            args.search,
            args.method,
            args.checkpoint,
            args.verbose,
            prefilter=args.prefilter,
            candidate_pool_size=args.candidate_pool,
        )
        x, y = int(round(best.x)), int(round(best.y))
        calibration = details.get("confidence_calibration", compute_calibrated_confidence(best, ranked))
        confidence = calibration["confidence"]
        if args.output_format == "json":
            payload = {
                "x": x,
                "y": y,
                "confidence": confidence,
                "low_confidence": calibration["low_confidence"],
                "method": args.method,
            }
            print(json.dumps(payload))
        else:
            print(f"({x}, {y})")
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        print(f"localize.py: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()

