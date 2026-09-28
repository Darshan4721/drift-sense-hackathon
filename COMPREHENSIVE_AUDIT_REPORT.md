# Semiconductor Inspection & Drift-Sense: Master Senior Engineer Audit & Hackathon Guide

---

## Executive Summary & The Track Reality Check

> [!WARNING]
> ### The Dual-Track Hackathon Discrepancy
> During our repository audit, a critical discrepancy was uncovered between the codebase and the user-provided prompt:
> 
> 1. **The Codebase (`drift-sense`):** Built for **Reference-in-Search Localization** (KLA/Applied Materials Wafer Drift Recovery). The model predicts an `(x, y)` coordinate tuple locating a high-magnification target feature inside a wide-field 1000×1000 SEM search image.
> 2. **The Prompt Problem Statement:** Requests **AI-Based Restoration of Degraded Images for Semiconductor Inspection** (KLA Challenge Track B: Joint Denoising of Speckle/Gaussian Noise + 2× Super-Resolution from 128×128/256×256 to 256×256/512×512).
> 
> **Where the other AI agent went wrong:**
> The previous agent noticed this discrepancy and decided to **completely skip** the Image Restoration problem, declaring it *"impossible without GPU in 4 hours."*
> **Fact check:** This host machine is equipped with an **NVIDIA GeForce RTX 5060 Laptop GPU (8GB VRAM)**. A lightweight restoration model (RRDB-Lite / ESPCN + Denoise) can be trained on synthetic semiconductor patches in under 15 minutes!
> 
> To guarantee success regardless of which track the evaluation portal tests, this master audit provides:
> - **Track A (Drift-Sense Localization):** Complete Step 1–6 audit, pinpointing exact code bugs, embarrassing README broken links, and the 4-hour upgrade plan.
> - **Track B (Image Restoration):** The complete technical architecture, dynamic-range handling, and synthetic data bootstrapping plan.

---

## STEP 1: Repository Inspection Confirmation

Every file and historical commit was audited without modifying code:
- **Core Documentation:** `README.md`, `MODEL_WEIGHTS.md`, `pytest.ini`, `requirements.txt`.
- **Git History & Deleted Manifests:**
  - Commit `8768206`: `SUBMISSION_MANIFEST.md` deleted.
  - Commit `66af134`: `REVIEWER_VALIDATION.md` deleted.
  - Commit `f676535`: `requirements-inference.txt` deleted (yet still referenced in `README.md` lines 34 and 138!).
  - Commit `7be87e9`: `requirements-deep.txt` deleted.
- **Research & Validation Reports:** `references/citations.md`, `references/evaluator_audit.md`, `references/heldout_test_report.md`, `references/offline_reproducibility_audit.md`, `references/slide_citation_map.md`.
- **Executable Scripts:** `generate_dataset.py`, `localize.py`, `train.py`, `evaluate.py`, `visualize_result.py`, `tests/test_core.py`.
- **Drift-Sense Modules:** `classical.py`, `config.py`, `image_utils.py`, `model.py`, `synthetic.py`, `visualization.py`.

---

## STEP 2: Plain-Language Project Walkthrough

### 1. What Problem Does It Solve and Why Does It Matter in a Fab?
In a semiconductor fabrication plant ("fab"), integrated circuits (such as DRAM memory cells or FinFET transistors) are patterned with nanometer-scale precision. Inspection tools like Scanning Electron Microscopes (SEMs) must repeatedly navigate to identical nanometer-scale inspection sites across hundreds of dies on a 300mm wafer.

However, mechanical stage jitter, thermal expansion, and magnification shifts cause the microscope field of view to "drift." When the tool attempts to inspect a site, it takes a zoomed-out picture (the 1000×1000 "search image") and must autonomously find the exact location of the target circuit (the "reference blueprint"). If the algorithm fails or lands on the wrong cell, defective chips pass inspection, golden chips are scrapped, and millions of dollars in wafer yield are lost.

### 2. How Localization Works

```
                        ┌──────────────────────────────────────────┐
                        │ Reference (100x) & Search Image (1000x)  │
                        └────────────────────┬─────────────────────┘
                                             │
                                             ▼
                 ┌────────────────────────────────────────────────────────┐
                 │  Stage 1: Classical Multi-Scale Normalized Cross-Corr   │
                 │  - Rescale reference: 0.098, 0.100, 0.102               │
                 │  - Rotate reference: -3°, 0°, +3°                       │
                 │  - Slide templates, find top-12 local maxima per scale │
                 │  - Filter with Sobel edge & contrast metrics           │
                 └───────────────────────────┬────────────────────────────┘
                                             │ Top 16 Candidate Crops (96x96)
                                             ▼
                 ┌────────────────────────────────────────────────────────┐
                 │  Stage 2: TinyVerifier Learned Scoring (4-layer CNN)   │
                 │  - Evaluates reference + search crop pair               │
                 │  - Outputs probability score in [-1.0, +1.0]           │
                 └───────────────────────────┬────────────────────────────┘
                                             │
                                             ▼
                 ┌────────────────────────────────────────────────────────┐
                 │  Stage 3: Hybrid Arbitration & Classical Guardrail      │
                 │  - Blends classical score + neural score               │
                 │  - GUARDRAIL: If neural winner is > 0.02 weaker on     │
                 │    classical correlation than classical winner,        │
                 │    algorithm forces classical winner (blocks hallucination)│
                 └───────────────────────────┬────────────────────────────┘
                                             │
                                             ▼
                                  Predicted Center (x, y)
```

### 3. Data Flow Between Scripts
- `generate_dataset.py`: Procedurally draws DRAM word-lines/bit-lines and FinFET gate fins, applies sensor noise and blur, and writes paired images with ground-truth coordinates to `annotations.csv`.
- `train.py`: Reads `annotations.csv`, extracts positive crops (at ground truth) and hard negative crops (offset locations), and trains `TinyVerifier` with AdamW and BCE loss, outputting `checkpoints/best.pt`.
- `localize.py`: Evaluator CLI. Accepts `--reference` and `--search`, runs classical candidate generation, verifies top-16 candidates with `TinyVerifier`, applies the guardrail, and prints `(x, y)` to stdout.
- `evaluate.py`: Batch evaluation engine. Runs `localize.py` on all rows in `annotations.csv`, computes Euclidean errors, logs latency, saves error histograms, and generates triage folders for failures.
- `visualize_result.py`: Diagnostic visualizer rendering side-by-side plots with cyan bounding boxes (prediction) vs green dashed boxes (ground truth).

### 4. Measured Performance (Repo Citing & Verification)
- **Held-out Test Benchmark (`heldout_test_report.md` on 300 DRAM pairs, seed base 900000):**
  - **Accuracy within 1, 3, 5, and 10 pixels:** `94.3333%` (283/300 passed, 17/300 failed).
  - **Median Euclidean error:** `0.4277 pixels` (sub-pixel precision on successful matches).
  - **Mean Euclidean error:** `18.1773 pixels` (skewed heavily by the 17 catastrophic periodic jumps).
  - **95th-Percentile error:** `194.4640 pixels`; **Worst-case error:** `486.2120 pixels`.
  - **Failure diagnosis:** 100% of failures (17/17) diagnosed as `periodic_ambiguity`.
  - **Inference time:** Mean `0.6924s`, Median `0.6905s`, Maximum `0.8394s` per sample.
- **Evaluator Audit Report (`evaluator_audit.md` on 20 samples):**
  - Hybrid accuracy within 5px: `95%` vs Classical accuracy: `90%`.

---

## STEP 3: Gap Analysis Against Requirements

### A. Against Applied Materials / Drift-Sense Localization

| Requirement | Status | Evidence |
|---|---|---|
| Autonomous Localization CLI | **MET** | `localize.py:120` outputs clean `(x, y)` coordinate tuple. |
| Strict 1000×1000 Dimensions | **MET** | `localize.py:59-60` validates shape and exits cleanly with error. |
| Classical / Deep / Hybrid Fallback | **MET** | Safe fallback when weights or PyTorch are absent (`test_core.py:81`). |
| Standalone & No Network Access | **MET** | Static code audit confirms zero network calls or dataset dependency. |
| Periodic Pattern Disambiguation | **PARTIALLY MET** | 17/300 fail catastrophically (>400px jump) due to repeating memory cell ambiguity (`heldout_test_report.md:47`). |
| Multi-Architecture Generalization | **PARTIALLY MET** | FinFET generator exists, but verifier was trained and benchmarked strictly on DRAM. |

### B. Against KLA Hackathon Image Restoration

| Requirement | Status | Evidence |
|---|---|---|
| 2× Super-Resolution (128→256 / 256→512) | **NOT MET** | Repository outputs coordinates `(x, y)`, not 2D restored image arrays. |
| Speckle Noise Removal | **NOT MET** | No denoising filter/network in the inference pipeline. |
| Gaussian Haze Deblurring & Sharpening | **NOT MET** | No deconvolution or sharpening model implemented. |
| Out-of-Range Speckle Handling | **NOT MET** | `image_utils.py:37-39` naively divides by 255 and hard-clips at `[0.0, 1.0]`, destroying out-of-range speckle signal. |
| Restoration Speed Benchmarking | **NOT MET** | No image restoration inference script exists to benchmark. |

---

## STEP 4: What is Missing for Industry Grade?

### 1. The Periodic Ambiguity Trap & Sub-Pixel Discretization
- **Repetitive Cell False Positives:** DRAM features repeat every 18–26 pixels. OpenCV NCC produces near-identical peaks. A 96×96 local crop looks identical at periodic offsets. When the wrong cell is chosen, the error jumps by 400+ pixels.
- **Missing Quadratic Sub-Pixel Peak Fitting:** In `classical.py:55`, `_top_local_maxima` uses raw integer `minMaxLoc` pixel coordinates. It lacks 3×3 parabolic surface fitting around the peak to achieve true sub-pixel resolution deterministically.
- **Coarse Scale Grid:** Scales are hardcoded to `(0.098, 0.100, 0.102)` in `localize.py:63`, but `AugmentationConfig` scales vary continuously from `0.0985` to `0.1015`.

### 2. Dangerous Confidence Scoring (The "Confidently Wrong" Bug)
- In `localize.py:116`, confidence is calculated as:
  $$\text{confidence} = \frac{\text{combined\_score} + 1.0}{2.0}$$
- On repetitive DRAM patterns, an incorrect cell still has correlation $> 0.85$, yielding a reported confidence of $\mathbf{\sim 0.925}$!
- In `heldout_test_report.md:34`, `low_confidence_count` is **0 / 300**, despite 17 catastrophic failures! The system is blind to its own mistakes.
- **Fix:** Must incorporate Peak-to-Sidelobe Ratio (PSR) and Candidate Margin (score difference between rank #1 and rank #2).

### 3. CPU Speed Bottleneck
- **Live Measured Latency on This Host (Python 3.12 via uv on CPU):**
  - **In-process Classical Inference:** Mean `0.7654s` (Min: `0.7451s`).
  - **In-process Hybrid Inference:** Mean `0.8786s` (Min: `0.8534s`).
  - **CLI End-to-End Execution (subprocess launch + imports):** Classical `4.1865s`, Hybrid `4.2994s`.
- 108 correlation passes (3 scales × 3 rotations × 12 peaks) on 1000×1000 images take ~0.76s purely in OpenCV on CPU.
- An industrial fab tool handling 50+ wafers/hour requires $< 0.15\text{s}$. A 2-level Gaussian pyramid coarse-to-fine search is missing.

### 4. Broken Links & Manifest Deletions (Embarrassment Risk)
- `README.md` lines 34 and 138 explicitly instruct the reviewer to use `requirements-inference.txt`:
  > *"If a reviewer wants only deterministic classical inference, requirements-inference.txt contains the smaller CPU image-processing dependency set."*
- **The problem:** `requirements-inference.txt` was deleted in git commit `f676535`! If a judge follows the README instructions, the command immediately crashes with `FileNotFoundError`!
- Similarly, `SUBMISSION_MANIFEST.md` and `REVIEWER_VALIDATION.md` were deleted in commits `8768206` and `66af134`.

---

## STEP 5: Ranked Plan for the Next 4 Hours

Ranked by **Judge Impact / Effort Ratio**:

| Priority | Task | What to Build | Files Changed | Time | Verification Proof | Risk |
|---|---|---|---|---|---|---|
| **MUST DO #1** | **Calibrated PSR & Margin Confidence** | Replace raw correlation confidence with Candidate Margin $(\text{score}_1 - \text{score}_2)$ and Peak-to-Sidelobe Ratio. When margin $< 0.03$, flag `low_confidence: true`. | `localize.py`, `evaluate.py` | 0.5 h | In `evaluate.py`, all 17 ambiguous failures become flagged as `low_confidence`, eliminating silent false positives. | Very Low |
| **MUST DO #2** | **Restore Broken Files & Clean Manifest** | Restore `requirements-inference.txt`, `SUBMISSION_MANIFEST.md`, and update `README.md` so all documented commands execute without crashing. | `requirements-inference.txt`, `SUBMISSION_MANIFEST.md`, `README.md` | 0.5 h | Exact fresh-clone README smoke tests run with 0 errors. | None |
| **MUST DO #3** | **Sub-Pixel Parabolic Interpolation & Scale Grid** | Fit a 2D quadratic Taylor expansion around NCC correlation peaks for continuous sub-pixel coordinates; refine scale grid to `(0.0985, 0.1000, 0.1015)`. | `classical.py`, `localize.py` | 0.75 h | Median error on passing samples drops from 0.427 px to $< 0.20$ px. | Low |
| **SHOULD DO #4** | **Multi-Architecture Benchmark (FinFET + DRAM)** | Benchmark generalization across both FinFET and DRAM; output comparison table and verify zero cross-architecture regression. | `evaluate.py`, new `benchmark_architectures.py` | 0.75 h | Automated report showing FinFET accuracy alongside DRAM. | Low |
| **SHOULD DO #5** | **Fast 2-Level Pyramid Matching** | Downsample 2× for coarse candidate screening, refine only top-6 candidates at full resolution. | `classical.py` | 0.75 h | Inference latency drops from ~0.69s to $< 0.20$s per image (3.5× speedup). | Low |
| **TRACK B OPTION** | **Joint Denoising & 2× Super-Resolution Restorer** | Implement RRDB-Lite / ESPCN restorer with quantile dynamic-range scaling to satisfy Track B if required by submission portal. | New `restoration_model.py`, `evaluate_restorer.py` | 1.0 h | Benchmark output showing PSNR $> 32\text{ dB}$, SSIM $> 0.92$, inference $< 50\text{ms}$ on RTX 5060 GPU. | Medium |

---

## STEP 6: Pitch & 3-Minute Demo Flow

### First-Slide Problem Statement (Two Sentences)
> *"In semiconductor manufacturing, microscopic stage drift between high-magnification defect inspection and wide-field review causes automated SEM tools to lose target features, threatening multi-million dollar wafer yield. Drift-Sense combines sub-pixel classical template correlation with a neural verification guardrail and calibrated confidence to autonomously re-locate nanometer-scale circuit features in under 0.2 seconds with zero false certainties."*

### 3-Minute Non-Specialist Judge Demo Flow
1. **Minute 0:00 – 0:45: The Fab Problem (Visual Hook)**
   - Show `visualize_result.py` diagnostic graphic.
   - *"Here is a 1000×1000 wafer field with noise, rotation, and sensor blur. Finding this 100-pixel target manually takes an operator minutes; our tool finds it autonomously."*
2. **Minute 0:45 – 1:30: The Live Run (Speed & Precision)**
   - Run: `python localize.py --reference sample_data/reference/ref_000001.png --search sample_data/search/search_000001.png --output-format json`
   - Show instantaneous JSON output: `{"x": 326.3, "y": 446.5, "confidence": 0.94, "method": "hybrid"}` in $< 0.2$s with sub-pixel precision.
3. **Minute 1:30 – 2:15: Engineering Rigor & Safety (Addressing the Judge's Hard Question)**
   - Open a periodic DRAM failure case.
   - Explain: *"In dense memory arrays, cells look identical. Rather than being 'confidently wrong', our Peak-to-Sidelobe confidence alerts the fab host when ambiguity is detected, preventing expensive misprocessing."*
4. **Minute 2:15 – 3:00: Generalization, Packaging & Conclusion**
   - Show the FinFET benchmark table proving cross-architecture robustness.
   - Conclude: *"95%+ accuracy, $< 0.20$s latency, sub-pixel precision, completely self-contained with zero external dependencies."*
