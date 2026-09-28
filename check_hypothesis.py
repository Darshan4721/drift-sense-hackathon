import csv
import json
import math
from pathlib import Path
import numpy as np

from drift_sense.image_utils import load_grayscale
from drift_sense.classical import generate_candidates, rerank_candidates

def test_hypothesis():
    data_dir = Path("data_bench_dram_degraded")
    results_dir = Path("results_bench_dram_degraded")
    csv_path = results_dir / "predictions.csv"
    
    with open(csv_path, "r", encoding="utf-8") as f:
        predictions = list(csv.DictReader(f))
        
    failures = [r for r in predictions if float(r["error"]) > 5.0]
    print(f"Total failures to evaluate: {len(failures)}")
    
    hits_in_16_at_5px = 0
    hits_in_16_at_10px = 0
    hits_in_all_candidates_at_5px = 0
    min_dist_in_16_list = []
    
    detailed_results = []
    
    for row in failures:
        sample_id = int(row["sample_id"])
        true_x = float(row["true_x"])
        true_y = float(row["true_y"])
        
        ref_path = data_dir / "reference" / f"ref_{sample_id:06d}.png"
        search_path = data_dir / "search" / f"search_{sample_id:06d}.png"
        
        ref = load_grayscale(ref_path)
        search = load_grayscale(search_path)
        
        candidates = generate_candidates(ref, search, scales=(0.098, 0.100, 0.102), top_k_per_scale=12, rotations=(-3.0, 0.0, 3.0))
        classical_ranked = rerank_candidates(candidates, search.shape)
        
        top16 = classical_ranked[:16]
        
        # Check distances in top 16
        distances_16 = [math.hypot(c.x - true_x, c.y - true_y) for c in top16]
        min_dist_16 = min(distances_16) if distances_16 else float("inf")
        min_dist_in_16_list.append(min_dist_16)
        
        # Check all candidates
        distances_all = [math.hypot(c.x - true_x, c.y - true_y) for c in candidates]
        min_dist_all = min(distances_all) if distances_all else float("inf")
        
        in_16_5px = min_dist_16 <= 5.0
        in_16_10px = min_dist_16 <= 10.0
        in_all_5px = min_dist_all <= 5.0
        
        if in_16_5px:
            hits_in_16_at_5px += 1
        if in_16_10px:
            hits_in_16_at_10px += 1
        if in_all_5px:
            hits_in_all_candidates_at_5px += 1
            
        detailed_results.append({
            "sample_id": sample_id,
            "min_dist_16": min_dist_16,
            "min_dist_all": min_dist_all,
            "in_16_5px": in_16_5px,
            "candidate_count": len(candidates)
        })
        
    print(f"\n--- HYPOTHESIS TEST RESULTS (on {len(failures)} Degraded Failures) ---")
    print(f"Correct location (<= 5px) in classical top-16: {hits_in_16_at_5px} / {len(failures)} ({hits_in_16_at_5px / len(failures) * 100:.1f}%)")
    print(f"Correct location (<= 10px) in classical top-16: {hits_in_16_at_10px} / {len(failures)} ({hits_in_16_at_10px / len(failures) * 100:.1f}%)")
    print(f"Correct location (<= 5px) anywhere in all candidates: {hits_in_all_candidates_at_5px} / {len(failures)} ({hits_in_all_candidates_at_5px / len(failures) * 100:.1f}%)")
    print(f"Median min-distance to true location among top-16: {np.median(min_dist_in_16_list):.2f} px")
    print(f"Mean min-distance to true location among top-16: {np.mean(min_dist_in_16_list):.2f} px")
    
    with open("hypothesis_check.json", "w", encoding="utf-8") as f:
        json.dump({
            "total_failures": len(failures),
            "hits_in_16_at_5px": hits_in_16_at_5px,
            "hits_in_16_at_10px": hits_in_16_at_10px,
            "hits_in_all_candidates_at_5px": hits_in_all_candidates_at_5px,
            "median_min_dist_in_16": float(np.median(min_dist_in_16_list)),
            "details": detailed_results
        }, f, indent=2)

if __name__ == "__main__":
    test_hypothesis()
