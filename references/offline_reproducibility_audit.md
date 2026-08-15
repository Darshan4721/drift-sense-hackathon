# Clean Ubuntu offline reproducibility audit

## Simulation setup

The latest release archive was extracted into a fresh directory. The simulation began with no `data/`, no `checkpoints/best.pt`, no evaluation results, and no example images. Python 3.12.3 and pip 24.0 were used in an isolated virtual environment.

The required CPU dependencies installed successfully during setup from `requirements.txt`. The optional training manifest installed PyTorch 2.13.0+cu130, with CUDA unavailable; all training and inference ran on CPU. The absence of an offline wheel cache is not an inference defect: package installation requires either a setup-time package source or preloaded wheels, while inference itself performs no network access.

## README workflow validation

The following workflow was executed in order in the clean clone.

| Step | Command class | Outcome |
|---|---|---|
| 1 | Install `requirements.txt` | Passed; NumPy 2.2.6, OpenCV 4.10.0, SciPy 1.14.1, Matplotlib, and pytest imported. |
| 2 | Generate DRAM data | Passed with 20 pairs for a bounded smoke run; 20 references, 20 searches, annotations, metadata, and environment files were created. |
| 3 | Run inference with no checkpoint | Passed; output matched the strict coordinate pattern `(integer_x, integer_y)` with empty stderr. |
| 4 | Install `requirements-deep.txt` | Passed; PyTorch imported on CPU. |
| 5 | Train verifier | Passed; `best.pt`, `final_metrics.json`, `split.json`, and `training_log.csv` were created. |
| 6 | Run hybrid evaluation | Passed; 20 annotated samples produced metrics, predictions, histogram, success/failure examples, and failure-analysis output. |
| 7 | Run JSON inference | Passed; emitted valid JSON with `x`, `y`, `confidence`, and `method`. |
| 8 | Run visualization CLI | Passed; output PNG was created. |
| 9 | Generate FinFET data | Passed with 3 pairs. |
| 10 | Run inference outside repository working directory | Passed; script-relative checkpoint discovery and package imports worked. |
| 11 | Load packaged checkpoint in clean environment | Passed under PyTorch 2.13.0+cu130 with CUDA unavailable. |

## Offline checks

A source scan found no runtime imports or calls to `requests`, `urllib`, `socket`, `curl`, `wget`, or other network clients in Python source files. Inference accepted absolute paths from `/tmp`, produced coordinate-only stdout, and produced no stderr. The default checkpoint path is resolved relative to `localize.py`, so running the script from another working directory does not break weight loading.

## Corrections made or confirmed

The repository already contained the necessary package initializer, shared modules, CLI entry points, requirements manifests, optional checkpoint fallback, and script-relative checkpoint discovery. The clean run confirmed that no additional missing file, import, path, dependency, weight-loading, or command-line correction was necessary after the audit. The only failed probe was an audit harness assertion that incorrectly expected a coordinate from a different dataset seed; the repository correctly returned the newly generated sample’s actual coordinate. That probe was replaced with a syntax and output-contract assertion.

The final archive includes this report, `requirements.txt`, `requirements-deep.txt`, the trained-checkpoint workflow, and the held-out evaluation artifacts. Inference remains usable when both the dataset and checkpoint are absent.
