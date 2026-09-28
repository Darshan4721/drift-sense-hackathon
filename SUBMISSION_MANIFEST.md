# Reviewer manifest

This manifest lists the submission materials at their reviewer-accessible locations.

| Item | Path | Validation status |
|---|---|---|
| Complete setup & benchmark results | `README.md` | Contains clone/setup, generation, benchmark results, and inference smoke test. |
| Dataset generator | `generate_dataset.py` | Supports `--architecture`, `--num-pairs`, and `--output-dir`; writes `annotations.csv`. |
| Inference script | `localize.py` | Requires `--reference` and `--search`; outputs calibrated confidence and coordinates. |
| DL weights | `checkpoints/best.pt` | Automatic default load path for hybrid verification. |
| Clearly named weight copy | `model_weights/drift_sense_verifier.pt` | Identical downloadable checkpoint supplied for inspection. |
| Training script | `train.py` | Reproducible CLI with checkpoints and split metadata. |
| Pip freeze | `requirements.txt` | Development-environment freeze including `torch`. |
| CPU-only inference freeze | `requirements-inference.txt` | Compact CPU-only dependency freeze for classical execution. |
| Supporting citations | `references/citations.md` | Design-rationale sources and parameter mapping. |
| Slide citation map | `references/slide_citation_map.md` | Maps presentation source numbers to repository citations. |
| Step 2 benchmark report | `STEP_2_BENCHMARK_REPORT.md` | Multi-architecture and degradation benchmark evidence. |
| Classical adaptation report | `CLASSICAL_ADAPTATION_REPORT.md` | Empirical evaluation of prefilters (DoG, LoG, LCN) and pool expansion. |

The submission contains no network download logic in its inference path and no required dependency on dataset annotations during inference.
