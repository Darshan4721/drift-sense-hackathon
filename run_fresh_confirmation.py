import json
from pathlib import Path
from drift_sense.config import AugmentationConfig
from drift_sense.synthetic import write_dataset
from evaluate import evaluate

def run_confirmation():
    fresh_suites = [
        {
            "id": "dram_default_fresh",
            "name": "Fresh DRAM Default",
            "architecture": "DRAM",
            "seed": 650000,
            "data_dir": Path("data_fresh_dram_default"),
            "config": AugmentationConfig(),
        },
        {
            "id": "finfet_default_fresh",
            "name": "Fresh FinFET Default",
            "architecture": "FinFET",
            "seed": 750000,
            "data_dir": Path("data_fresh_finfet_default"),
            "config": AugmentationConfig(),
        },
        {
            "id": "dram_degraded_fresh",
            "name": "Fresh DRAM Degraded",
            "architecture": "DRAM",
            "seed": 850000,
            "data_dir": Path("data_fresh_dram_degraded"),
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

    # 1. Generate datasets if not existing
    for suite in fresh_suites:
        if not (suite["data_dir"] / "annotations.csv").is_file():
            print(f"Generating 150 pairs for {suite['name']} (seed {suite['seed']})...")
            write_dataset(
                output_dir=suite["data_dir"],
                num_pairs=150,
                seed=suite["seed"],
                architecture=suite["architecture"],
                config=suite["config"],
            )

    # 2. Evaluate each suite with prefilter=none (Baseline) and prefilter=dog (Candidate)
    results = {}
    for suite in fresh_suites:
        results[suite["id"]] = {}
        for prefilter in ["none", "dog"]:
            out_dir = Path(f"results_fresh_{suite['id']}_{prefilter}")
            print(f"Evaluating {suite['name']} with prefilter={prefilter}...")
            metrics = evaluate(
                data_dir=suite["data_dir"],
                checkpoint="checkpoints/best.pt",
                tolerance=5.0,
                output_dir=out_dir,
                method="hybrid",
                prefilter=prefilter,
                candidate_pool_size=16,
            )
            
            # Read predictions.csv to compute failure detection & false alarm rates
            import csv
            with open(out_dir / "predictions.csv", "r", encoding="utf-8") as f:
                preds = list(csv.DictReader(f))
            fails = [r for r in preds if float(r["error"]) > 5.0]
            passes = [r for r in preds if float(r["error"]) <= 5.0]
            flagged_fails = [r for r in fails if r["low_confidence"].lower() == "true"]
            flagged_passes = [r for r in passes if r["low_confidence"].lower() == "true"]
            
            results[suite["id"]][prefilter] = {
                "name": suite["name"],
                "prefilter": prefilter,
                "accuracy_5px": metrics["accuracy_within_5_pixels"] * 100.0,
                "median_error": metrics["median_euclidean_error"],
                "mean_error": metrics["mean_euclidean_error"],
                "p95_error": metrics["p95_euclidean_error"],
                "total_fails": len(fails),
                "flagged_fails": len(flagged_fails),
                "fail_catch_rate": (len(flagged_fails) / len(fails) * 100.0) if fails else 100.0,
                "false_alarms": len(flagged_passes),
                "false_alarm_rate": (len(flagged_passes) / len(passes) * 100.0) if passes else 0.0,
                "avg_time": metrics["average_inference_seconds"],
            }

    with open("fresh_confirmation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nSaved confirmation results to fresh_confirmation_results.json")

if __name__ == "__main__":
    run_confirmation()
