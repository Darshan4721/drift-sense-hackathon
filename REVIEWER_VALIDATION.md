# Reviewer workflow validation

The submission-root commands were executed without source edits using the isolated development environment recorded in `requirements.txt`.

| Workflow step | Actual command outcome |
|---|---|
| DRAM sample generation | `generate_dataset.py --architecture DRAM --num-pairs 1 --output-dir reviewer_smoke --seed 2026` created reference/search PNG files, `annotations.csv`, and `metadata.json`. |
| Standalone classical inference | `localize.py --reference reviewer_smoke/reference/ref_000001.png --search reviewer_smoke/search/search_000001.png --method classical` printed `(326, 446)`. |
| Automatic learned-weight path | Default `hybrid` inference printed JSON prediction `{"x": 326, "y": 446, "confidence": 0.881273, "method": "hybrid"}` with no manual `--checkpoint` path. |
| Ground-truth recording | The generated annotation row recorded `x=326.3736081336574`, `y=446.4849240661465`, and `seed=2027`. |
| Weight integrity | `checkpoints/best.pt` and `model_weights/drift_sense_verifier.pt` had matching SHA-256 `c8a3b9f6819321da21a08026b0507adfc6b6053ad26b52a068fd71ae4527b757`. |
| Training smoke test | Twelve generated DRAM pairs trained for one epoch using `train.py`; output directory contained `best.pt`, `training_log.csv`, `final_metrics.json`, and `split.json`. |

The bundled Python tests were also retained in `tests/`. This validation report is evidence of packaging and command execution; it does not substitute synthetic smoke-test accuracy for the held-out experiment reported separately in `references/heldout_test_report.md`.

## Exact fresh-extraction README path

The final archive was extracted into a new directory, and the README setup and smoke-test workflow below completed successfully without source edits:

```bash
unzip drift-sense-applied-materials-submission.zip
cd drift-sense-am-submission-release
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python generate_dataset.py --architecture DRAM --num-pairs 1 --output-dir reviewer_smoke --seed 2026
python localize.py --reference reviewer_smoke/reference/ref_000001.png --search reviewer_smoke/search/search_000001.png --method classical
python -c "import csv; r=next(csv.DictReader(open('reviewer_smoke/annotations.csv'))); print({'true_center_x': r['x'], 'true_center_y': r['y'], 'seed': r['seed']})"
```

The observed coordinate output was `(326, 446)`. The generated annotation recorded the ground-truth fields `x=326.3736081336574`, `y=446.4849240661465`, and `seed=2027`. The complete full-freeze installation command succeeded before generation and inference.
