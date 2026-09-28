#!/usr/bin/env python3
"""Evaluate Drift-Sense localization on an annotated dataset."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from drift_sense.config import save_json
from drift_sense.image_utils import load_grayscale, save_grayscale
from drift_sense.visualization import render_result
from localize import localize


def _failure_category(error: float, candidate_gap: float, row: dict[str, str], prediction: tuple[float, float]) -> str:
    """Assign a transparent heuristic category for triage."""

    if candidate_gap < 0.03:
        return "periodic_ambiguity"
    if abs(float(row.get("rotation", 0.0))) > 1.2:
        return "rotation_mismatch"
    if float(row.get("blur_sigma", 0.0)) > 1.0:
        return "blur"
    if float(row.get("noise_sigma", 0.0)) > 4.0:
        return "excessive_noise"
    if prediction[0] < 60 or prediction[0] > 940 or prediction[1] < 60 or prediction[1] > 940:
        return "boundary_effect"
    if error > 100:
        return "weak_structural_features"
    return "multiple_equally_valid_candidates"


def evaluate(
    data_dir: str | Path,
    checkpoint: str | Path | None,
    tolerance: float,
    output_dir: str | Path,
    method: str = "hybrid",
    prefilter: str = "none",
    candidate_pool_size: int = 16,
) -> dict[str, Any]:
    """Run inference on every annotation and write all requested artifacts."""

    data_path, output_path = Path(data_dir), Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    annotations_path = data_path / "annotations.csv"
    if not annotations_path.is_file():
        raise FileNotFoundError(f"Missing annotations CSV: {annotations_path}")
    rows = list(csv.DictReader(annotations_path.open(encoding="utf-8", newline="")))
    if not rows:
        raise ValueError("annotations.csv contains no samples")
    rows_by_id = {int(row["sample_id"]): row for row in rows}
    predictions: list[dict[str, Any]] = []
    failure_root = output_path / "failure_analysis"
    failure_root.mkdir(exist_ok=True)
    times: list[float] = []
    errors: list[float] = []
    low_confidence = 0
    for row in rows:
        reference_path = data_path / row["reference_path"]
        search_path = data_path / row["search_path"]
        truth = (float(row["x"]), float(row["y"]))
        start = time.perf_counter()
        best, ranked, details = localize(
            reference_path,
            search_path,
            method=method,
            checkpoint=checkpoint,
            prefilter=prefilter,
            candidate_pool_size=candidate_pool_size,
        )
        elapsed = time.perf_counter() - start
        prediction = (float(best.x), float(best.y))
        error = float(np.hypot(prediction[0] - truth[0], prediction[1] - truth[1]))
        calib = details.get("confidence_calibration", {})
        confidence = float(calib.get("confidence", np.clip((best.combined_score + 1.0) / 2.0, 0.0, 1.0)))
        is_low_conf = bool(calib.get("low_confidence", confidence < 0.5))
        if is_low_conf or confidence < 0.5:
            low_confidence += 1
        times.append(elapsed)
        errors.append(error)
        score_order = sorted(ranked, key=lambda candidate: candidate.combined_score, reverse=True)
        score_gap = (score_order[0].combined_score - score_order[1].combined_score) if len(score_order) > 1 else 1.0
        prediction_row = {
            "sample_id": int(row["sample_id"]),
            "pred_x": prediction[0],
            "pred_y": prediction[1],
            "true_x": truth[0],
            "true_y": truth[1],
            "error": error,
            "confidence": confidence,
            "low_confidence": is_low_conf,
            "margin_distant": calib.get("margin_distant", 0.0),
            "psr": calib.get("psr", 0.0),
            "inference_seconds": elapsed,
            "method": method,
            "scale": best.scale,
            "candidate_gap": score_gap,
            "top_candidates": json.dumps([candidate.to_dict() for candidate in ranked[:5]]),
        }
        predictions.append(prediction_row)
        if error > tolerance:
            case_dir = failure_root / f"sample_{int(row['sample_id']):06d}"
            case_dir.mkdir(parents=True, exist_ok=True)
            reference = load_grayscale(reference_path)
            search = load_grayscale(search_path)
            save_grayscale(case_dir / "reference.png", reference)
            save_grayscale(case_dir / "search.png", search)
            category = _failure_category(error, prediction_row["candidate_gap"], row, prediction)
            failure_payload = {
                "sample_id": int(row["sample_id"]),
                "predicted_center": prediction,
                "true_center": truth,
                "error_distance": error,
                "top_candidate_scores": [candidate.to_dict() for candidate in ranked[:5]],
                "first_second_score_difference": prediction_row["candidate_gap"],
                "estimated_scale": best.scale,
                "failure_category": category,
                "details": details,
            }
            save_json(case_dir / "failure.json", failure_payload)
            render_result(
                reference,
                search,
                prediction,
                truth,
                predicted_size=(best.width, best.height),
                truth_bbox=(float(row["bbox_x0"]), float(row["bbox_y0"]), float(row["bbox_x1"]), float(row["bbox_y1"])),
                confidence=confidence,
                output=case_dir / "diagnostic.png",
                title=f"Failure sample {row['sample_id']}",
            )

    with (output_path / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0].keys()))
        writer.writeheader()
        writer.writerows(predictions)
    errors_array = np.asarray(errors, dtype=np.float64)
    times_array = np.asarray(times, dtype=np.float64)
    metrics: dict[str, Any] = {
        "number_of_samples": len(rows),
        "mean_euclidean_error": float(errors_array.mean()),
        "median_euclidean_error": float(np.median(errors_array)),
        "maximum_error": float(errors_array.max()),
        "p95_euclidean_error": float(np.percentile(errors_array, 95)),
        "accuracy_within_1_pixel": float(np.mean(errors_array <= 1.0)),
        "accuracy_within_3_pixels": float(np.mean(errors_array <= 3.0)),
        "accuracy_within_5_pixels": float(np.mean(errors_array <= 5.0)),
        "accuracy_within_10_pixels": float(np.mean(errors_array <= 10.0)),
        "average_inference_seconds": float(times_array.mean()),
        "median_inference_seconds": float(np.median(times_array)),
        "maximum_inference_seconds": float(times_array.max()),
        "failure_count": int(np.sum(errors_array > tolerance)),
        "low_confidence_count": low_confidence,
        "low_confidence_rate": float(low_confidence / len(rows)),
        "tolerance": tolerance,
        "method": method,
    }
    save_json(output_path / "metrics.json", metrics)
    save_json(output_path / "failure_analysis.json", {"failure_count": metrics["failure_count"], "tolerance": tolerance, "cases": [p["sample_id"] for p in predictions if p["error"] > tolerance]})

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(errors_array, bins=min(40, max(10, len(errors) // 4)), color="#2563eb", alpha=0.85)
    ax.axvline(tolerance, color="#dc2626", linestyle="--", label=f"tolerance={tolerance:g}px")
    ax.set_xlabel("Euclidean localization error (pixels)")
    ax.set_ylabel("Samples")
    ax.set_title("Drift-Sense error distribution")
    ax.legend()
    fig.savefig(output_path / "error_histogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    if predictions:
        success = min(predictions, key=lambda item: item["error"])
        failure = max(predictions, key=lambda item: item["error"])
        for item, name in [(success, "success_example.png"), (failure, "failure_example.png")]:
            render_result(
                load_grayscale(data_path / rows_by_id[int(item["sample_id"])] ["reference_path"]),
                load_grayscale(data_path / rows_by_id[int(item["sample_id"])] ["search_path"]),
                (item["pred_x"], item["pred_y"]),
                (item["true_x"], item["true_y"]),
                confidence=item["confidence"],
                output=output_path / name,
                title=f"{name.replace('_', ' ').replace('.png', '')}",
            )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--tolerance", type=float, default=5.0)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--method", choices=["hybrid", "classical", "deep"], default="hybrid")
    parser.add_argument("--prefilter", choices=["none", "dog", "log", "lcn"], default="none")
    parser.add_argument("--candidate-pool", type=int, default=16)
    args = parser.parse_args()
    evaluate(
        args.data_dir,
        args.checkpoint,
        args.tolerance,
        args.output_dir,
        args.method,
        prefilter=args.prefilter,
        candidate_pool_size=args.candidate_pool,
    )


if __name__ == "__main__":
    main()
