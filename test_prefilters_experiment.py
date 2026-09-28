import csv
import json
import math
from pathlib import Path
import cv2
import numpy as np

from drift_sense.image_utils import load_grayscale, normalize_contrast
from drift_sense.classical import (
    Candidate,
    _top_local_maxima,
    _local_feature_score,
    rerank_candidates,
)
from localize import _verify_candidates, _default_checkpoint

def apply_dog(image: np.ndarray, s1: float = 1.0, s2: float = 2.5) -> np.ndarray:
    g1 = cv2.GaussianBlur(image, (0, 0), s1)
    g2 = cv2.GaussianBlur(image, (0, 0), s2)
    dog = g1 - g2
    return normalize_contrast(dog)

def apply_log(image: np.ndarray, s: float = 1.5) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), s)
    log = cv2.Laplacian(blurred, cv2.CV_32F, ksize=3)
    return normalize_contrast(log)

def apply_lcn(image: np.ndarray, s: float = 4.0) -> np.ndarray:
    mu = cv2.GaussianBlur(image, (0, 0), s)
    diff = image - mu
    var = cv2.GaussianBlur(diff * diff, (0, 0), s)
    sigma = np.sqrt(np.maximum(var, 0.0)) + 1e-4
    lcn = diff / sigma
    return normalize_contrast(lcn)

def apply_prefilter(image: np.ndarray, method: str) -> np.ndarray:
    if method == "none":
        return image
    elif method == "dog":
        return apply_dog(image, 1.0, 2.5)
    elif method == "log":
        return apply_log(image, 1.5)
    elif method == "lcn":
        return apply_lcn(image, 4.0)
    else:
        raise ValueError(f"Unknown prefilter {method}")

def generate_candidates_with_prefilter(
    reference: np.ndarray,
    search: np.ndarray,
    prefilter: str = "none",
    scales=(0.098, 0.100, 0.102),
    top_k_per_scale: int = 12,
    rotations=(-3.0, 0.0, 3.0),
) -> list[Candidate]:
    ref_norm = normalize_contrast(reference)
    search_norm = normalize_contrast(search)
    
    if prefilter != "none":
        ref_corr = apply_prefilter(ref_norm, prefilter)
        search_corr = apply_prefilter(search_norm, prefilter)
    else:
        ref_corr = ref_norm
        search_corr = search_norm
        
    all_candidates = []
    for scale in scales:
        width = max(24, int(round(reference.shape[1] * float(scale))))
        height = max(24, int(round(reference.shape[0] * float(scale))))
        if width >= search.shape[1] or height >= search.shape[0]:
            continue
        base_template_corr = cv2.resize(ref_corr, (width, height), interpolation=cv2.INTER_AREA)
        base_template_orig = cv2.resize(ref_norm, (width, height), interpolation=cv2.INTER_AREA)
        
        for rotation in rotations:
            if abs(float(rotation)) > 1e-6:
                matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), float(rotation), 1.0)
                template_corr = cv2.warpAffine(base_template_corr, matrix, (width, height), borderMode=cv2.BORDER_REFLECT_101)
                template_orig = cv2.warpAffine(base_template_orig, matrix, (width, height), borderMode=cv2.BORDER_REFLECT_101)
            else:
                template_corr = base_template_corr
                template_orig = base_template_orig
                
            score_map = cv2.matchTemplate(search_corr.astype(np.float32), template_corr.astype(np.float32), cv2.TM_CCOEFF_NORMED)
            for x, y, _ in _top_local_maxima(score_map, top_k_per_scale, max(8, min(width, height) // 3)):
                intensity, edge, structure, periodicity = _local_feature_score(search_norm, template_orig, x, y)
                all_candidates.append(
                    Candidate(
                        x=x + width / 2.0,
                        y=y + height / 2.0,
                        width=width,
                        height=height,
                        scale=float(scale),
                        rotation_degrees=float(rotation),
                        intensity_score=intensity,
                        edge_score=edge,
                        structure_score=structure,
                        periodicity_score=periodicity,
                    )
                )
    return all_candidates

def evaluate_variant(prefilter: str = "none", pool_size: int = 16):
    data_dir = Path("data_bench_dram_degraded")
    csv_path = Path("results_bench_dram_degraded/predictions.csv")
    with open(csv_path, "r", encoding="utf-8") as f:
        ground_truth = list(csv.DictReader(f))
        
    checkpoint_path = _default_checkpoint()
    total = len(ground_truth)
    correct_5px = 0
    errors = []
    
    # Track the 63 known baseline failure IDs
    baseline_failure_ids = {int(r["sample_id"]) for r in ground_truth if float(r["error"]) > 5.0}
    hits_in_top16_on_failures = 0
    recovered_samples = []
    
    for row in ground_truth:
        sample_id = int(row["sample_id"])
        true_x = float(row["true_x"])
        true_y = float(row["true_y"])
        
        ref = load_grayscale(data_dir / "reference" / f"ref_{sample_id:06d}.png")
        search = load_grayscale(data_dir / "search" / f"search_{sample_id:06d}.png")
        
        candidates = generate_candidates_with_prefilter(ref, search, prefilter=prefilter)
        classical_ranked = rerank_candidates(candidates, search.shape)
        
        # Check top-16 presence for baseline failures
        if sample_id in baseline_failure_ids:
            top16 = classical_ranked[:16]
            if any(math.hypot(c.x - true_x, c.y - true_y) <= 5.0 for c in top16):
                hits_in_top16_on_failures += 1
                
        # Verification stage
        verification_candidates = classical_ranked[:pool_size]
        local_scores, _ = _verify_candidates(ref, search, verification_candidates, checkpoint_path)
        original_indices = {id(c): i for i, c in enumerate(candidates)}
        verifier_scores = {
            original_indices[id(verification_candidates[local_index])]: score
            for local_index, score in local_scores.items()
        }
            
        ranked = rerank_candidates(candidates, search.shape, verifier_scores)
        classical_best = classical_ranked[0]
        learned_best = ranked[0]
        if learned_best.classical_score + 0.02 < classical_best.classical_score:
            ranked = classical_ranked
            best = classical_best
        else:
            best = learned_best
            
        err = math.hypot(best.x - true_x, best.y - true_y)
        errors.append(err)
        if err <= 5.0:
            correct_5px += 1
            if sample_id in baseline_failure_ids:
                recovered_samples.append(sample_id)
                
    acc = correct_5px / total * 100.0
    med_err = np.median(errors)
    mean_err = np.mean(errors)
    
    print(f"\n================ Variant: prefilter={prefilter}, pool={pool_size} ================")
    print(f"Accuracy (<= 5px): {correct_5px} / {total} ({acc:.2f}%)")
    print(f"Median Error: {med_err:.2f} px | Mean Error: {mean_err:.2f} px")
    print(f"Ground truth in top-16 on the 63 baseline failures: {hits_in_top16_on_failures} / 63 ({hits_in_top16_on_failures / 63 * 100:.1f}%)")
    print(f"Recovered from baseline failures: {len(recovered_samples)}: {recovered_samples}")
    return {
        "prefilter": prefilter,
        "pool_size": pool_size,
        "accuracy_5px": acc,
        "correct_count": correct_5px,
        "median_error": float(med_err),
        "mean_error": float(mean_err),
        "hits_in_top16_on_failures": hits_in_top16_on_failures,
        "recovered_samples": recovered_samples,
    }

if __name__ == "__main__":
    results = []
    # Test all variants requested:
    # 0. Baseline (none, pool=16)
    print("Testing 0: Baseline (none, pool=16)...")
    results.append(evaluate_variant("none", 16))
    
    # 1. Pool 32 (none, pool=32)
    print("Testing 1: Enlarged Pool (none, pool=32)...")
    results.append(evaluate_variant("none", 32))
    
    # 2. Difference-of-Gaussians (dog, pool=16)
    print("Testing 2: Difference-of-Gaussians (dog, pool=16)...")
    results.append(evaluate_variant("dog", 16))
    
    # 3. Laplacian of Gaussian (log, pool=16)
    print("Testing 3: Laplacian of Gaussian (log, pool=16)...")
    results.append(evaluate_variant("log", 16))
    
    # 4. Local Contrast Normalization (lcn, pool=16)
    print("Testing 4: Local Contrast Normalization (lcn, pool=16)...")
    results.append(evaluate_variant("lcn", 16))
    
    with open("prefilter_experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nSaved all results to prefilter_experiment_results.json")
