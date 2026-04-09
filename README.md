# Altermagnet Inverse Design

**Machine-learning-driven inverse design of altermagnetic materials with large spin-splitting energy**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)

---

<p align="center">
  <img src="workflow.pdf" alt="Workflow" width="800">
</p>

## Overview

This repository provides the data, trained model, and code for quantitative inverse design of altermagnetic materials.

We introduce three continuous, DFT-free structural descriptors — **MSBI** (Motif Symmetry-Breaking Index), **MPF** (Motif Packing Fraction), and **p/d electron ratio** — that predict spin-splitting energy (SSE) from crystal structure and composition alone. An interpretable XGBoost surrogate trained on 3,851 DFT-labeled structures achieves R² = 0.70 and MAE = 123.1 meV. Bayesian optimization over this descriptor space yields six magnetically stable candidates with DFT-validated SSE above 0.5 eV, including three above 1 eV.

## Repository Contents

| File | Description |
|------|-------------|
| `descriptor.ipynb` | Compute 52 structural descriptors from POSCAR files (bond geometry, coordination angles, motif symmetry, elemental properties) |
| `BO.py` | Resumable Bayesian optimization pipeline with Optuna TPE and prototype matching |
| `final_model_all.json` | Pre-trained XGBoost model for SSE prediction |
| `POSCARS.zip` | 3,851 crystal structures in VASP POSCAR format |

## Requirements

```
python >= 3.8
numpy
pandas
pymatgen
scipy
scikit-learn
xgboost >= 2.0
optuna
matplotlib
```

## Usage

### 1. Compute Descriptors

Use `descriptor.ipynb` to extract 52 structural descriptors from POSCAR files. The notebook computes nine descriptor categories and merges them into a single DataFrame. Motifs $\mathcal{M}_1$ and $\mathcal{M}_2$ denote the coordination polyhedra of the two antiparallel magnetic sublattices; $M$ and $X$ refer to the magnetic and non-magnetic species.

| Category (# features) | Descriptors |
|------------------------|-------------|
| **Symmetry** (2) | MSBI (motif symmetry-breaking index), σ_inhom (asymmetry inhomogeneity) |
| **Bond** (7) | MX_avg, MX_max, MX_min, MX_std, MX_spread, MX_CV, MX_elong |
| **Angle** (9) | XMX angles (max, min, avg, std, spread), XXX angles (max, min, std, spread) |
| **Topology** (6) | Coordination number N_X, motif dimensionality, convex-hull volume, XX_long, XX_short, shape anisotropy η |
| **Distortion** (2) | Shape distortion σ_shape, off-center displacement δ_off |
| **Chemical** (8) | Z_M, Z_X, χ_M, χ_X, ΔZ, \|ΔZ\|, Δχ, \|Δχ\| |
| **Electronic** (5) | d-electrons n_d, p-electrons n_p, unpaired electrons, spin-only moment μ, p/d ratio |
| **Global** (9) | Inter-motif centroid distances (1st–3rd), magnetic neighbor distances (1st–3rd), cell volume, MPF, motif alignment angle θ_align |
| **Hybrid** (4) | χ–elongation coupling Γ, sublattice displacements Δ_sub (1st–3rd shell) |

```python
# Input:  POSCAR files (unzip POSCARS.zip)
# Output: merged DataFrame with 52 descriptors per structure
```

### 2. Predict SSE

```python
import xgboost as xgb

model = xgb.XGBRegressor()
model.load_model("final_model_all.json")

# X: descriptor matrix from step 1
sse_pred = model.predict(X)
```

### 3. Inverse Design via Bayesian Optimization

`BO.py` runs a resumable Optuna TPE optimization that maximizes predicted SSE in descriptor space, then maps optimized descriptors to real crystal structures via cosine similarity + Euclidean distance prototype matching, stratified by coordination number (CN = 4 or 6).

```bash
# Basic run (1000 trials, 8 parallel jobs)
python BO.py --model final_model_all.json --data fin_data.csv --trials 1000 --jobs 8

# Resume from checkpoint
python BO.py --trials 5000 --checkpoint-dir ./checkpoints

# Analysis only (skip optimization)
python BO.py --skip-optimization --output ./results
```

Key outputs: `top5_by_coordination.csv`, `similarity_analysis_by_coordination.csv`, `monitoring_dashboard.png`, `final_report.txt`

## Key Descriptors

- **MSBI** (Motif Symmetry-Breaking Index) — Extends the discrete motif-interconvertibility criterion into a continuous symmetry-breaking measure via Hungarian matching of two magnetic sublattice motifs. Higher MSBI → stronger spin splitting.
- **MPF** (Motif Packing Fraction) — Encodes the magnetic exchange scale through motif packing geometry, computed from the convex hull volume of coordination polyhedra.
- **p/d ratio** — Ratio of ligand p-electrons to magnetic-ion d-electrons; selects covalent chemistries with spin-asymmetric hybridization.

## Citation

Paper in preparation. Citation information will be updated upon publication.

## License

This project is licensed under the [MIT License](LICENSE).
