import csv
import json
import numpy as np
from pathlib import Path
from localize import localize

def analyze_tuning_set():
    data_dir = Path("tuning_data_500000")
    with (data_dir / "annotations.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    results = []
    print(f"Analyzing {len(rows)} samples in {data_dir}...")
    for idx, row in enumerate(rows):
        sample_id = int(row["sample_id"])
        truth_x, truth_y = float(row["x"]), float(row["y"])
        ref = data_dir / row["reference_path"]
        search = data_dir / row["search_path"]

        best, ranked, details = localize(ref, search, method="hybrid")
        error = float(np.hypot(best.x - truth_x, best.y - truth_y))

        # Sort candidates by combined_score
        scores = [c.combined_score for c in ranked]
        s1 = scores[0]

        # Raw margin to next candidate
        margin_raw = s1 - scores[1] if len(scores) > 1 else 1.0

        # Margin to next SPATIALLY DISTINCT candidate (> 15 pixels away)
        dist_threshold = 15.0
        s2_distant = None
        for c in ranked[1:]:
            d = np.hypot(c.x - best.x, c.y - best.y)
            if d > dist_threshold:
                s2_distant = c.combined_score
                break
        if s2_distant is None:
            s2_distant = scores[-1]

        margin_distant = s1 - s2_distant

        # Sidelobes (excluding cluster around winner)
        sidelobes = [c.combined_score for c in ranked[1:] if np.hypot(c.x - best.x, c.y - best.y) > dist_threshold]
        if sidelobes:
            mu = float(np.mean(sidelobes))
            sigma = float(np.std(sidelobes)) + 1e-6
            psr = float((s1 - mu) / sigma)
        else:
            psr = 99.0

        results.append({
            "sample_id": sample_id,
            "error": error,
            "failed": error > 5.0,
            "s1": s1,
            "margin_raw": margin_raw,
            "margin_distant": margin_distant,
            "psr": psr,
        })
        if (idx + 1) % 25 == 0:
            print(f"Processed {idx + 1}/{len(rows)} samples...")

    failed = [r for r in results if r["failed"]]
    passed = [r for r in results if not r["failed"]]

    print(f"\n================ TUNING ANALYSIS SUMMARY ================")
    print(f"Total samples: {len(results)}")
    print(f"Passed: {len(passed)}, Failed: {len(failed)} ({len(failed)/len(results)*100:.1f}%)")

    print("\n--- FAILED CASES (error > 5px) ---")
    for f in failed:
        print(f"Sample {f['sample_id']:03d}: error={f['error']:6.1f}px | s1={f['s1']:.4f} | margin_raw={f['margin_raw']:.4f} | margin_dist={f['margin_distant']:.4f} | PSR={f['psr']:.2f}")

    print("\n--- PASSED CASES (error <= 5px) DISTRIBUTION ---")
    m_raw_pass = [p["margin_raw"] for p in passed]
    m_dist_pass = [p["margin_distant"] for p in passed]
    psr_pass = [p["psr"] for p in passed]
    print(f"Margin Raw:     min={min(m_raw_pass):.4f}, median={np.median(m_raw_pass):.4f}, p5={np.percentile(m_raw_pass, 5):.4f}")
    print(f"Margin Distant: min={min(m_dist_pass):.4f}, median={np.median(m_dist_pass):.4f}, p5={np.percentile(m_dist_pass, 5):.4f}")
    print(f"PSR:            min={min(psr_pass):.2f}, median={np.median(psr_pass):.2f}, p5={np.percentile(psr_pass, 5):.2f}")

    print("\n--- THRESHOLD SWEEP (on 100 tuning samples) ---")
    print(f"{'Condition / Threshold':<35} | {'Detected Failures':<18} | {'False Alarms (Pass flagged)':<28}")
    print("-" * 88)

    for thresh in [0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040]:
        det = sum(1 for f in failed if f["margin_raw"] < thresh)
        fa = sum(1 for p in passed if p["margin_raw"] < thresh)
        print(f"margin_raw < {thresh:.3f}                      | {det}/{len(failed)} ({det/len(failed)*100:5.1f}%)    | {fa}/{len(passed)} ({fa/len(passed)*100:4.1f}%)")

    for thresh in [0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.050]:
        det = sum(1 for f in failed if f["margin_distant"] < thresh)
        fa = sum(1 for p in passed if p["margin_distant"] < thresh)
        print(f"margin_distant < {thresh:.3f}                  | {det}/{len(failed)} ({det/len(failed)*100:5.1f}%)    | {fa}/{len(passed)} ({fa/len(passed)*100:4.1f}%)")

    for psr_thresh in [1.5, 2.0, 2.5, 3.0, 3.5]:
        det = sum(1 for f in failed if f["psr"] < psr_thresh)
        fa = sum(1 for p in passed if p["psr"] < psr_thresh)
        print(f"PSR < {psr_thresh:.1f}                                 | {det}/{len(failed)} ({det/len(failed)*100:5.1f}%)    | {fa}/{len(passed)} ({fa/len(passed)*100:4.1f}%)")

    # Combined conditions
    for m_th, psr_th in [(0.025, 2.0), (0.030, 2.5), (0.025, 2.5), (0.030, 2.0)]:
        det = sum(1 for f in failed if (f["margin_distant"] < m_th or f["psr"] < psr_th))
        fa = sum(1 for p in passed if (p["margin_distant"] < m_th or p["psr"] < psr_th))
        print(f"margin_dist<{m_th:.3f} OR PSR<{psr_th:.1f}           | {det}/{len(failed)} ({det/len(failed)*100:5.1f}%)    | {fa}/{len(passed)} ({fa/len(passed)*100:4.1f}%)")

if __name__ == "__main__":
    analyze_tuning_set()
