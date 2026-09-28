import json
import csv
from pathlib import Path

def analyze_degraded():
    csv_path = Path("results_bench_dram_degraded/predictions.csv")
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    
    fails = [r for r in reader if float(r["error"]) > 5.0]
    passes = [r for r in reader if float(r["error"]) <= 5.0]
    
    avg_conf_pass = sum(float(r["confidence"]) for r in passes) / len(passes)
    avg_conf_fail = sum(float(r["confidence"]) for r in fails) / len(fails)
    avg_psr_pass = sum(float(r["psr"]) for r in passes) / len(passes)
    avg_psr_fail = sum(float(r["psr"]) for r in fails) / len(fails)
    
    print(f"Degraded Passes ({len(passes)}): avg conf={avg_conf_pass:.3f}, avg psr={avg_psr_pass:.2f}")
    print(f"Degraded Fails ({len(fails)}): avg conf={avg_conf_fail:.3f}, avg psr={avg_psr_fail:.2f}")
    
    # Check failure categories
    failure_cases = sorted(Path("results_bench_dram_degraded/failure_analysis").glob("sample_*"))
    categories = {}
    for case in failure_cases:
        f_json = case / "failure.json"
        if f_json.exists():
            data = json.load(open(f_json, "r", encoding="utf-8"))
            cat = data.get("failure_category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
    print("Failure Categories:", categories)

if __name__ == "__main__":
    analyze_degraded()
