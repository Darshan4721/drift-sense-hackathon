# Reviewer workflow validation

The submission-root commands were executed without source edits using the isolated development environment recorded in `requirements.txt`.

| Workflow step | Actual command outcome |
|---|---|
| DRAM sample generation | `generate_dataset.py --architecture DRAM --num-pairs 1 --output-dir reviewer_smoke --seed 2026` created reference/search PNG files, `annotations.csv`, and `metadata.json`. |
| Standalone classical inference | `localize.py --reference reviewer_smoke/reference/ref_000001.png --search reviewer_smoke/search/search_000001.png --method classical` printed `(326, 446)`. |
| Automatic learned-weight path | Default `hybrid` inference printed JSON prediction with calibrated confidence: `{"x": 326, "y": 446, "confidence": 0.881273, "low_confidence": false, "method": "hybrid"}` with no manual `--checkpoint` path. |
| Ground-truth recording | The generated annotation row recorded `x=326.3736081336574`, `y=446.4849240661465`, and `seed=2027`. |
| Weight integrity | `checkpoints/best.pt` and `model_weights/drift_sense_verifier.pt` had matching SHA-256 `c8a3b9f6819321da21a08026b0507adfc6b6053ad26b52a068fd71ae4527b757`. |
| Test suite execution | `pytest -v tests/test_core.py` ran 8 tests cleanly (`8 passed in 31.46s`). |

This validation report is evidence of packaging and command execution; all accuracy figures are measured strictly on synthetic benchmarks and reported in `README.md`.
