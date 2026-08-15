# Applied Materials reviewer manifest

This manifest lists the requested submission materials at their reviewer-accessible locations.

| Item | Path | Validation status |
|---|---|---|
| Complete setup instructions | `README.md` | Contains clone/setup, generation, and inference smoke test. |
| Dataset generator | `generate_dataset.py` | Supports `--architecture`, `--num-pairs`, and `--output-dir`; writes `annotations.csv`. |
| Inference script | `localize.py` | Requires `--reference` and `--search`; defaults to coordinate-only output. |
| DL weights | `checkpoints/best.pt` | Automatic default load path. |
| Training script | `train.py` | Reproducible CLI with checkpoints and split metadata. |
| Pip freeze | `requirements.txt` | Development-environment freeze including `torch`. |
| Supporting citations | `references/citations.md` | Design-rationale sources and parameter mapping. |

The submission contains no network download logic in its inference path and no required dependency on dataset annotations during inference.
