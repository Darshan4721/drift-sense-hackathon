import csv
import json
from pathlib import Path

def inspect_suite(name, dir_path):
    csv_path = Path(dir_path) / "predictions.csv"
    if not csv_path.exists():
        print(f"{name}: {csv_path} not found")
        return
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    
    fails = [r for r in reader if float(r["error"]) > 5.0]
    passes = [r for r in reader if float(r["error"]) <= 5.0]
    unflagged_fails = [r for r in fails if r["low_confidence"].lower() != "true"]
    false_alarms = [r for r in passes if r["low_confidence"].lower() == "true"]
    
    print(f"\n=================== {name} ===================")
    print(f"Total: {len(reader)} | Passes: {len(passes)} | Fails: {len(fails)}")
    print(f"Flagged Fails: {len(fails) - len(unflagged_fails)}/{len(fails)} | False Alarms: {len(false_alarms)}/{len(passes)}")
    
    if unflagged_fails:
        print("\n--- UNFLAGGED FAILURES (Missed by Confidence Gate) ---")
        for r in unflagged_fails:
            print(f"Sample {r['sample_id']:>4}: err={float(r['error']):6.1f} px | margin_dist={float(r['margin_distant']):.4f} | psr={float(r['psr']):.2f} | s1={float(r['confidence']):.3f} | scale={float(r['scale']):.2f}")
    
    print("\n--- SAMPLE FAILURES SUMMARY (Top 5) ---")
    for r in fails[:5]:
        print(f"Sample {r['sample_id']:>4}: err={float(r['error']):6.1f} px | low_conf={r['low_confidence']:<5} | margin_dist={float(r['margin_distant']):.4f} | psr={float(r['psr']):.2f} | s1={float(r['confidence']):.3f}")

if __name__ == "__main__":
    inspect_suite("DRAM Default (150)", "results_bench_dram_default")
    inspect_suite("FinFET Default (150)", "results_bench_finfet_default")
    inspect_suite("DRAM Degraded (150)", "results_bench_dram_degraded")
