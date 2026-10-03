# Drift-Sense: Reference-in-Search Localization & Semiconductor Metrology

## 1. Project Purpose & Scope
Developed for the Erode Hackathon (Semiconductor / VLSI domain), **Drift-Sense** solves sub-pixel wafer defect and pattern localization under non-linear drift, varying illumination, sensor noise, and multi-pattern degradation across DRAM and FinFET architectures. It features:
- Standalone synthetic SEM pattern generator (`generate_dataset.py`).
- Hybrid classical-deep verification pipeline (`localize.py`).
- Calibrated confidence gating & prefilter evaluations.
- Complete benchmark suite across degraded DRAM and FinFET datasets.

---

## 2. Technology Stack & Current Status
- **Languages & Frameworks**: Python 3.10–3.12, PyTorch, Torchvision, OpenCV, NumPy, SciPy, Pandas, Scikit-learn.
- **Model Checkpoints**: Embedded weights in `checkpoints/best.pt` and `model_weights/drift_sense_verifier.pt`.
- **Status**: **100% Complete & Synchronized on GitHub** (`https://github.com/Darshan4721/drift-sense-hackathon.git`).

---

## 3. Quick Start & Dependency Reinstallation

### Step 1: Virtual Environment Setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
# Or minimal CPU classical inference: pip install -r requirements-inference.txt
```

### Step 3: Regenerate Datasets & Run Inference
```powershell
# 1. Generate sample DRAM wafer pairs
python generate_dataset.py --architecture DRAM --num-pairs 2 --output-dir sample_data --seed 42

# 2. Run localization inference
python localize.py --reference sample_data/reference/000000.png --search sample_data/search/000000.png --method hybrid
```
