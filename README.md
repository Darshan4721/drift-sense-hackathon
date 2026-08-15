# Drift-Sense: Reference-in-Search Localization

This repository is arranged for independent reviewer execution. It contains a standalone synthetic dataset generator, a standalone inference script, optional learned-verifier weights, a reproducible training script, a complete dependency freeze, and supporting design references. A reviewer can clone or unzip this repository, generate a sample DRAM or FinFET image pair, and run the localization algorithm without editing source files or contacting the authors.

## Deliverable index

| Required deliverable | Reviewer path | Purpose |
|---|---|---|
| Setup instructions | `README.md` | This document, including the exact smoke-test commands below. |
| Dataset generator | `generate_dataset.py` | Generates annotated DRAM/FinFET reference and search image pairs. |
| Localization inference | `localize.py` | Returns the predicted `(x, y)` center for evaluator-supplied images. |
| Model weights | `checkpoints/best.pt` | Default automatic checkpoint path for `--method hybrid` or `--method deep`. |
| Clearly named weight copy | `model_weights/drift_sense_verifier.pt` | Identical downloadable checkpoint supplied for inspection. |
| Training script | `train.py` | Reproduces learned-verifier training from generated or supplied data. |
| Complete package freeze | `requirements.txt` | `pip freeze` output from the Python development environment. |
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

## Quick reviewer smoke test

Run the following without editing any files:

```bash
python generate_dataset.py --architecture DRAM --num-pairs 1 --output-dir reviewer_smoke --seed 2026
python localize.py --reference reviewer_smoke/reference/ref_000001.png --search reviewer_smoke/search/search_000001.png --method classical
python -c "import csv; r=next(csv.DictReader(open('reviewer_smoke/annotations.csv'))); print({'true_center_x': r['x'], 'true_center_y': r['y'], 'seed': r['seed']})"
```

The first command generates the pair and annotation. The second prints the prediction in `(x, y)` format. The third displays the stored true center and metadata so the reviewer can compare prediction and ground truth without modifying the localization code.
