#!/usr/bin/env python3
"""Automated Multi-Architecture & Degradation Benchmark for Drift-Sense."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from drift_sense.config import AugmentationConfig, save_json
from drift_sense.synthetic import write_dataset
from evaluate import evaluate


def run_benchmark_suite():
    suites = [
        {
            "id": "dram_default",
            "name": "DRAM (Default Settings)",
            "architecture": "DRAM",
            "seed": 600000,
            "num_pairs": 150,
            "data_dir": Path("data_bench_dram_default"),
            "output_dir": Path("results_bench_dram_default"),
            "config": AugmentationConfig(),
        },
        {
            "id": "finfet_default",
            "name": "FinFET (Default Settings)",
            "architecture": "FinFET",
            "seed": 700000,
            "num_pairs": 150,
            "data_dir": Path("data_bench_finfet_default"),
            "output_dir": Path("results_bench_finfet_default"),
            "config": AugmentationConfig(),
        },
        {
            "id": "dram_degraded",
            "name": "DRAM (Increased Degradation: Blur & Noise)",
            "architecture": "DRAM",
            "seed": 800000,
            "num_pairs": 150,
            "data_dir": Path("data_bench_dram_degraded"),
            "output_dir": Path("results_bench_dram_degraded"),
            "config": AugmentationConfig(
                gaussian_sigma_search=10.0,
                gaussian_sigma_reference=4.0,
                blur_sigma_search=2.0,
                blur_sigma_reference=1.0,
                poisson_strength_search=5.0,
                illumination_sigma=0.12,
            ),
        },
    ]

    benchmark_summary = []

    for suite in suites:
        print(f"\n========================================================")
        print(f"Running Suite: {suite['name']}")
        print(f"========================================================")

        # 1. Generate Dataset if not present
        if not (suite["data_dir"] / "annotations.csv").is_file():
            print(f"Generating {suite['num_pairs']} pairs for {suite['id']} (seed {suite['seed']})...")
            write_dataset(
                output_dir=suite["data_dir"],
                num_pairs=suite["num_pairs"],
                architecture=suite["architecture"],
                seed=suite["seed"],
                config=suite["config"],
                save_visualizations=False,
            )
        else:
            print(f"Found existing dataset at {suite['data_dir']}.")

        # 2. Run Evaluation
        checkpoint_path = Path("checkpoints/best.pt")
        print(f"Evaluating {suite['id']} with method=hybrid...")
        metrics = evaluate(
            data_dir=suite["data_dir"],
            checkpoint=checkpoint_path,
            tolerance=5.0,
            output_dir=suite["output_dir"],
            method="hybrid",
        )

        # 3. Read predictions.csv to compute false alarms and failure detection rates
        predictions_path = suite["output_dir"] / "predictions.csv"
        with predictions_path.open(encoding="utf-8") as f:
            predictions = list(csv.DictReader(f))

        failures = [p for p in predictions if float(p["error"]) > 5.0]
        passes = [p for p in predictions if float(p["error"]) <= 5.0]

        flagged_failures = [
            p for p in failures
            if p["low_confidence"].lower() == "true" or float(p["confidence"]) < 0.5
        ]
        flagged_passes = [
            p for p in passes
            if p["low_confidence"].lower() == "true" or float(p["confidence"]) < 0.5
        ]

        failure_count = len(failures)
        pass_count = len(passes)
        flagged_fail_count = len(flagged_failures)
        flagged_pass_count = len(flagged_passes)

        fail_detection_rate = (flagged_fail_count / failure_count * 100.0) if failure_count > 0 else 100.0
        false_alarm_rate = (flagged_pass_count / pass_count * 100.0) if pass_count > 0 else 0.0

        suite_record = {
            "name": suite["name"],
            "id": suite["id"],
            "architecture": suite["architecture"],
            "seed": suite["seed"],
            "total_samples": len(predictions),
            "accuracy_5px": metrics["accuracy_within_5_pixels"] * 100.0,
            "median_error": metrics["median_euclidean_error"],
            "mean_error": metrics["mean_euclidean_error"],
            "p95_error": metrics["p95_euclidean_error"],
            "max_error": metrics["maximum_error"],
            "failure_count": failure_count,
            "flagged_failures": flagged_fail_count,
            "fail_detection_rate": fail_detection_rate,
            "false_alarm_rate": false_alarm_rate,
            "avg_time": metrics["average_inference_seconds"],
            "raw_metrics_path": str(suite["output_dir"] / "metrics.json"),
        }
        benchmark_summary.append(suite_record)

        # Update metrics.json with calibrated evaluation details
        extended_metrics = dict(metrics)
        extended_metrics["failures_flagged"] = flagged_fail_count
        extended_metrics["failures_total"] = failure_count
        extended_metrics["failure_detection_rate"] = fail_detection_rate
        extended_metrics["false_alarm_count"] = flagged_pass_count
        extended_metrics["passes_total"] = pass_count
        extended_metrics["false_alarm_rate"] = false_alarm_rate
        save_json(suite["output_dir"] / "metrics.json", extended_metrics)

    # 4. Print Markdown Comparison Table
    print("\n" + "=" * 90)
    print("STEP 2: ARCHITECTURE & DEGRADATION BENCHMARK RESULTS")
    print("=" * 90)

    header = (
        f"| {'Benchmark Set':<38} | {'Acc (<=5px)':<11} | {'Median Err':<10} | {'Mean Err':<10} | "
        f"{'P95 Err':<9} | {'Failures Flagged':<16} | {'False Alarms':<12} | {'Avg Time':<8} |"
    )
    separator = (
        f"|:{'-'*38}-|:{'-'*11}:|:{'-'*10}:|:{'-'*10}:|:{'-'*9}:|:{'-'*16}:|:{'-'*12}:|:{'-'*8}:|"
    )
    print(header)
    print(separator)

    for rec in benchmark_summary:
        row_str = (
            f"| {rec['name']:<38} | {rec['accuracy_5px']:6.2f}%     | {rec['median_error']:6.2f} px   | "
            f"{rec['mean_error']:6.2f} px   | {rec['p95_error']:6.2f} px | "
            f"{rec['flagged_failures']}/{rec['failure_count']:<2} ({rec['fail_detection_rate']:5.1f}%)    | "
            f"{rec['false_alarm_rate']:5.1f}%       | {rec['avg_time']:5.3f}s   |"
        )
        print(row_str)

    # Save summary report
    summary_path = Path("benchmark_architectures_summary.json")
    save_json(summary_path, benchmark_summary)
    print(f"\nSaved benchmark summary to {summary_path}")


if __name__ == "__main__":
    run_benchmark_suite()
