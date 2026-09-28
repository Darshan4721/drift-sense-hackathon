# Drift-Sense: Reference-in-Search Localization

This repository is arranged for independent reviewer execution. It contains a standalone synthetic dataset generator, a standalone inference script, optional learned-verifier weights, a reproducible training script, a complete dependency freeze, and supporting design references. A reviewer can clone or unzip this repository, generate a sample DRAM or FinFET image pair, and run the localization algorithm without editing source files or contacting the authors.

## Deliverable index

| Required deliverable | Reviewer path | Purpose |
|---|---|---|
| Setup instructions | `README.md` | This document, including the exact smoke-test commands below. |
| Reviewer manifest | `SUBMISSION_MANIFEST.md` | Verification checklist and deliverable index. |
| Reviewer validation | `REVIEWER_VALIDATION.md` | Recorded execution evidence of clean-room smoke test. |
| Dataset generator | `generate_dataset.py` | Generates annotated DRAM/FinFET reference and search image pairs. |
| Localization inference | `localize.py` | Returns the predicted `(x, y)` center and calibrated confidence. |
| Model weights | `checkpoints/best.pt` | Default automatic checkpoint path for `--method hybrid` or `--method deep`. |
| Clearly named weight copy | `model_weights/drift_sense_verifier.pt` | Identical downloadable checkpoint supplied for inspection. |
| Training script | `train.py` | Reproduces learned-verifier training from generated or supplied data. |
| Complete package freeze | `requirements.txt` | `pip freeze` output from the full development environment (with PyTorch). |
| Minimal CPU freeze | `requirements-inference.txt` | Compact CPU-only dependency freeze for classical execution. |
| References | `references/citations.md` | Rationale for layout, augmentation, noise, and localization choices. |
| Slide citation map | `references/slide_citation_map.md` | Maps presentation source numbers to the repository citations. |

## 1. Prerequisites

Use Python 3.10–3.12 and pip. The tested development environment used Python 3.12. The complete `requirements.txt` contains the learned-verifier stack. Inference does not download any data or model weights; the default checkpoint is included in this repository.



For a repository clone, clone the submission repository using the reviewer-provided repository URL, then change into the directory that contains this `README.md`. From either starting point, create the environment with:

```bash
python3 -m venv .venv
source .venv/bin/activate              # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If a reviewer wants only deterministic classical inference, `requirements-inference.txt` contains the smaller CPU image-processing dependency set. The expected evaluator command below is still supported with `--method classical` when PyTorch is unavailable.

## 2. Generate a sample image pair

The dataset generator is self-contained. It accepts the required architecture style, number of pairs, and output directory. It also accepts an optional seed and augmentation parameters.

```bash
python generate_dataset.py \
  --architecture DRAM \
  --num-pairs 2 \
  --output-dir sample_data \
  --seed 42
```

To generate the alternative architecture, replace `DRAM` with `FinFET`:

```bash
python generate_dataset.py --architecture FinFET --num-pairs 2 --output-dir sample_finfet --seed 42
```

The generated directory contains `reference/`, `search/`, `annotations.csv`, and `metadata.json`. Each annotation row records the sample ID, reference/search paths, ground-truth center coordinates, transformed target bounding box, seed, and augmentation metadata. Ground-truth centers are expressed as zero-based `(x, y)` pixel coordinates in the 1000×1000 search image.

## 3. Run standalone localization inference

The evaluator-facing entry point is `localize.py`. It accepts only the reference path and search path required for localization, has no manual configuration step, and prints the predicted center as exactly one coordinate tuple to standard output.

```bash
python localize.py \
  --reference sample_data/reference/ref_000001.png \
  --search sample_data/search/search_000001.png
```

Expected output format:

```text
(x, y)
```

`x` is the horizontal column and `y` is the vertical row. Both coordinates are zero-based pixel coordinates in the 1000×1000 search image. The script rejects inputs that are not exactly 1000×1000 and exits nonzero with a message on standard error. It does not need `annotations.csv`, the training data, internet access, or a database at inference time.

For structured output, use `--output-format json`. To force deterministic no-PyTorch behavior, use `--method classical`:

```bash
python localize.py --reference sample_data/reference/ref_000001.png --search sample_data/search/search_000001.png --method classical --output-format json
```

## 4. Model weights and automatic loading

The optional two-channel `TinyVerifier` checkpoint is included in two equivalent paths. `localize.py` automatically searches for `checkpoints/best.pt` next to the script, so no checkpoint argument is required for the default `hybrid` method. The additional `model_weights/drift_sense_verifier.pt` copy is provided as a clearly named artifact for reviewer inspection.

```bash
ls -lh checkpoints/best.pt model_weights/drift_sense_verifier.pt
python localize.py --reference sample_data/reference/ref_000001.png --search sample_data/search/search_000001.png --method hybrid
```

If PyTorch is absent, the checkpoint is malformed, or loading fails, the inference script falls back safely to the classical localization method. This behavior is intentional and keeps the evaluator command executable on a CPU-only environment.

## 5. Training the optional learned verifier

`train.py` trains the verifier on a generated or supplied annotated directory. It uses ID-level data splits so derivative crops from a single base sample are kept within one partition. The script writes checkpoints, JSON metrics, and split metadata to the selected checkpoint directory.

```bash
python generate_dataset.py --architecture DRAM --num-pairs 12 --output-dir training_data --seed 707
python train.py \
  --data-dir training_data \
  --output-dir checkpoints_retrained \
  --epochs 2 \
  --batch-size 16 \
  --seed 42
```

The short command above is a smoke test rather than a claim of final-quality training. The bundled `checkpoints/best.pt` was supplied separately to make learned verification immediately executable. The command-line script documents all training parameters through `python train.py --help`.

## 6. Evaluate and visualize

Evaluation uses ground-truth data only from `annotations.csv`; `localize.py` itself does not read those annotations. The command writes measured metrics, predictions, timing, plots, and per-failure artifacts to its output directory.

```bash
python evaluate.py \
  --data-dir sample_data \
  --checkpoint checkpoints/best.pt \
  --method hybrid \
  --tolerance 5 \
  --output-dir sample_results
```

To render an individual result:

```bash
python visualize_result.py \
  --reference sample_data/reference/ref_000001.png \
  --search sample_data/search/search_000001.png \
  --prediction 500 500 \
  --output example_result.png
```

## 7. Reproducibility and references

Run the included test suite after installing dependencies:

```bash
pytest -q
```

`requirements.txt` is the full `pip freeze` produced from the development environment used for the optional learned verifier. `requirements-inference.txt` is a compact subset for a classical-only CPU environment. The source and parameter rationale for layout, independent noise, blur, illumination, geometric perturbations, template matching, gradients, and resampling are in `references/citations.md`. The presentation-source mapping is in `references/slide_citation_map.md`.

> The synthetic generator is an engineering benchmark. It does not claim to reproduce a proprietary Applied Materials instrument, a specific foundry process, or calibrated SEM physics.

## 8. Measured Benchmark Results

> **Dataset Scope Disclosure:** All performance metrics reported below were measured strictly **on our synthetic benchmark suites** generated by `drift_sense.synthetic`. Performance is **never claimed on real fab images**.

| Benchmark Suite (Synthetic Benchmark) | Test Seed / Draw | Accuracy ($\le 5\text{px}$) | Median Error | Failures Caught by Gate | False Alarm Rate | Avg Time / Pair |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **DRAM Default** | Seed 600000 (Draw 1) | **91.33%** (137/150) | 0.44 px | 12 / 13 (92.31%) | 11.68% (16/137) | 0.966s |
| **DRAM Default** | Seed 650000 (Draw 2) | **90.00%** (135/150) | 0.44 px | 14 / 15 (93.33%) | 9.63% (13/135) | 1.186s |
| **FinFET Default** | Seed 700000 (Draw 1) | **98.67%** (148/150) | 0.41 px | 2 / 2 (100.0%) | 4.05% (6/148) | 0.968s |
| **FinFET Default** | Seed 750000 (Draw 2) | **98.67%** (148/150) | 0.48 px | 2 / 2 (100.0%) | 2.70% (4/148) | 0.939s |
| **DRAM Degraded** | Seed 800000 (Draw 1) | **58.00%** (87/150) | 0.71 px | 63 / 63 (100.0%) | 73.56% (64/87) | 0.995s |
| **DRAM Degraded** | Seed 850000 (Draw 2) | **53.33%** (80/150) | 0.91 px | 68 / 70 (97.14%) | 78.75% (63/80) | 0.955s |

*Summary across default suites:* Across two independent DRAM benchmarks (300 pairs total), our calibrated spatial confidence gate flagged 26 of 28 catastrophic failures (92.86% catch rate), and on our synthetic FinFET layouts it caught 4 of 4 failures (100.0% catch rate). Average inference latency across all sets on modern multi-core CPU hardware is 0.94s–1.19s per image pair.

## 9. Limitations

1. **Synthetic Data Only:** All evaluations and weights are derived solely from our synthetic wafer pattern generator (`drift_sense.synthetic`). Performance has not been validated on real fab SEM images, which exhibit unmodeled charging artifacts, hydrocarbon contamination, and instrument beam drift.
2. **Periodic DRAM Ambiguity:** In periodic DRAM layouts, repeated capacitor contacts produce identical local cross-correlation peaks. The model occasionally locks onto an adjacent memory cell multiple pitches away.
3. **Confidence Gate Catch Rate & Closest Miss:** The confidence gate catches most catastrophic failures, but not all of them. On the seed 600000 benchmark, Sample 67 was an unflagged failure ($133.0\text{px}$ error) whose distant score margin was $0.0218$ against our frozen $0.0200$ gate threshold, slipping through by $0.0018$.
4. **Accuracy Drop Under Severe Blur & Noise:** When Gaussian blur ($\sigma \in [1.5, 3.0]$) and noise variance ($\sigma^2 \in [0.01, 0.04]$) are elevated, accuracy falls drastically to **53.3%–58.0%**.
5. **Elevated False Alarm Rate on Degraded Images:** Because severe blur and noise suppress confidence scores across the entire image distribution, the confidence gate flags **73.6%–78.8%** of even correct predictions on degraded sets as uncertain.
6. **Classical Prefiltering Ineffective:** High-pass and band-pass prefilters (DoG, LoG, LCN) stripped low-frequency spatial context and made localization significantly worse (dropping degraded accuracy by an additional 14–18 percentage points).

## 10. What We Tried That Did Not Work

1. **Difference-of-Gaussians (DoG) Bandpass:** Filtered out low-frequency background and high-frequency noise before cross-correlation; dropped degraded DRAM accuracy from 58.0% to 44.0%.
2. **Laplacian-of-Gaussian (LoG) Edge Sharpening:** Second-derivative operator aimed at enhancing structure; amplified speckle noise and dropped accuracy to 40.0%.
3. **Local Contrast Normalization (LCN):** Divided by local Gaussian standard deviation; reduced global contrast clues and dropped accuracy to 41.3%.
4. **Candidate Pool Enlargement (16 to 32):** Doubled the candidates forwarded to the neural verifier; produced exactly 0.0% accuracy improvement (remained 58.0%).
5. **Root Cause:** In periodic DRAM arrays, low-to-mid frequency global envelopes are essential for disambiguating identical cells; stripping them multiplied periodic aliasing, while pool expansion could not fix the ~65% of failures where the true match was absent from classical candidate proposals.

## 11. Provenance

The base classical-hybrid localization algorithm and synthetic data generator originate from earlier Reference-in-Search baseline development. The calibrated spatial confidence gate, Peak-to-Sidelobe Ratio (PSR) analysis, multi-architecture benchmark suites, systematic degradation studies, and prefilter adaptation experiments were designed, implemented, and empirically audited during this event.

## Quick reviewer smoke test

Run the following without editing any files:

```bash
python generate_dataset.py --architecture DRAM --num-pairs 1 --output-dir reviewer_smoke --seed 2026
python localize.py --reference reviewer_smoke/reference/ref_000001.png --search reviewer_smoke/search/search_000001.png --method classical
python -c "import csv; r=next(csv.DictReader(open('reviewer_smoke/annotations.csv'))); print({'true_center_x': r['x'], 'true_center_y': r['y'], 'seed': r['seed']})"
```

The first command generates the pair and annotation. The second prints the prediction in `(x, y)` format. The third displays the stored true center and metadata so the reviewer can compare prediction and ground truth without modifying the localization code.
