# Altermagnet Inverse Design

**Quantifying symmetry breaking as a design variable for giant altermagnetic spin splitting**

[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost%203.4.1-orange.svg)](https://xgboost.readthedocs.io/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19488475.svg)](https://doi.org/10.5281/zenodo.19488475)

---

<p align="center">
  <img src="workflow.png" alt="Workflow" width="800">
</p>

## Overview

This repository holds the data, trained surrogate, and code behind a quantitative inverse-design
workflow for altermagnetic materials.

We introduce three continuous, DFT-free descriptors — **MSBI** (Motif Symmetry-Breaking Index),
**MPF** (Motif Packing Fraction), and the **p/d electron ratio** — that predict the spin-splitting
energy (SSE) from crystal structure and composition alone. An interpretable XGBoost surrogate is
trained on a **symmetry-labelled screened cohort of 3,845 DFT-labelled structures**, comprising
2,841 altermagnets together with 811 conventional antiferromagnets and 193
compensated-ferrimagnet candidates retained as symmetry controls. Bayesian optimization over the
descriptor space, followed by prototype matching and held-out DFT calculations, yields six
magnetically stable candidates.

MSBI is the descriptor whose removal costs the most held-out accuracy, and it does so as a
**gate**: it separates the symmetry-preserving low-splitting limit from the altermagnetic regime.
Within the altermagnetic population alone, variation in SSE is more strongly associated with
packing and covalency. `analysis/cohort/` contains the experiments that establish this.

### Performance

| Quantity | Value |
|---|---|
| R² on the full cohort, log(1+x) space | 0.7351 ± 0.0138 |
| R² on the full cohort, eV | 0.6933 ± 0.0171 |
| R² restricted to held-out altermagnetic rows | 0.586 (log1p) / 0.562 (eV) |
| MAE | 123.1 meV |
| Out-of-fold Spearman ρ · precision@20 | 0.871 · 1.000 |
| Top-decile bias | −31 % (the model under-predicts the largest splittings) |

All figures are pooled over 20 random seeds × 5 parent-grouped folds. The surrogate is a
screening and ranking tool, not a substitute for a DFT calculation.

## Pipeline

| Stage | Location | Content |
|---|---|---|
| 1. SSE extraction and screening | `screening/` | VASP `EIGENVAL` → SSE, Γ-point average splitting |
| 2. Descriptor computation | `descriptors/` | structure → 52 descriptors, MSBI included |
| 3. Model training and evaluation | `model/` | XGBoost, GroupKFold, ablation, nested CV, SHAP |
| 4. Inverse design | `optimization/` | Optuna BO + coordination-matched prototype matching |
| 5. DFT assessment | `vasp/`, `validation/` | input generation, convex hull, U scan, structure checks |
| 6. Symmetry classification | `symmetry/` | sublattice-connecting operation, magnetic space group |
| 7. Cohort-composition tests | `analysis/cohort/` | altermagnet-only and size-matched training cohorts |

## Directory layout

```
data/          fin_data.csv                   3,845 rows, 322 parents, 52 descriptors + SSE
               candidate_master.csv           six candidates: SG/MSG, lattice, moments, energy differences
               magnetic_symmetry_*.csv        symmetry classification results
               raw/, derived/                 SSE variants, orbital matching, convergence and
                                              per-target SHAP tables read by
                                              analysis/point2_make_figures.py
descriptors/   descriptor.ipynb               descriptor computation (notebook, 25 cells)
               run_descriptors.py             batch execution of the notebook cells
               msbi_calc.py                   MSBI alone
               mpf_audit.py                   audit of the MPF face-selection stability
screening/     parse_eigenval.py              EIGENVAL → SSE (canonical extractor)
               SSE_ax.py                      strain-variant batch driver (Python)
               SSE_calc.sh                    shell driver that generates and runs the extractor
optimization/  BO.py                          Optuna BO (defines FEATURE_ORDER)
               optimization_results_resumable.json   100,000 trials; all 52 descriptors for the top 5
               similarity_analysis_final_results_top50.csv  nearest 50 prototypes per trial
model/         final_model_all_named.json     trained XGBoost (feature_names embedded)
               ablation_grouped.py            ablation + drop-column + grouped permutation
               stability_selection_100_parallel.py   20 seeds x 5 folds stability selection
               nested_hp_validation.py        nested hyperparameter validation
               retrain_targets.py             alternative-target retraining + SHAP
               table2.py, mpf_impact.py       prediction re-evaluation, Williams leverage
               model_check.py, fix_model.py   model verification, feature_names assignment
               verify_model.py                loads the artifact and reports what it reproduces
               slurm/                         submission scripts
symmetry/      altermag_sym.py                sublattice-connecting operation
               altermag_batch.py              batch execution over the whole cohort
               msg_type.py                    magnetic space group type (I–IV)
               cand_table.py                  builds the candidate master table
analysis/      point2_make_figures.py         reviewer-response figures 1-8 (reads data/raw, data/derived)
               prototype_match.py             structure-based nearest-prototype search
               nias_classify.py               NiAs-type classification and distortion metrics
               structure_check.py             coordination, planarity and space-group checks
               msbi_threshold_bootstrap.py    bootstrap intervals for the MSBI gate statistics
               cohort/am_only.py              altermagnet-only refit on identical held-out rows
               cohort/size_control.py         size- and parent-matched training-cohort controls
validation/    hull_quick.py, mu_test.py      Ni–S convex hull, reference-phase independence
               analyze_uscan.py               Hubbard U scan
               isif3_extract.py, sse_calc.py  relaxation behaviour, SSE extraction
               data/                          the hull and U-scan tables the above read
                                              (hull_final.json/.tsv, mp_reference.json,
                                              uscan_hull_summary.json, uscan_report.txt,
                                              report_template.md, run_manifest.json)
vasp/          gen_incar.sh, POT.sh           INCAR/POTCAR generation (per-element U mapping)
               templates/                     the production INCAR and KPOINTS files
structures/    *_relaxed.vasp                 the six final candidates
               POSCARS.zip                    structure archive, 5,945 files
figures/       plotdata/                      flat CSV tables for redrawing each figure
               reference_plots/               verification plots and their scripts
               archive/                       the original notebooks, outputs cleared
```

## Data

**`data/fin_data.csv`** — 3,845 rows across 322 parents. `filename` defines the cohort and
`parent` is the GroupKFold group key.

| Item | Value |
|---|---|
| Target | `sse` (eV): the maximum of \|e↑−e↓\| over same-band-index pairs for which **both** eigenvalues lie inside `[E_F−2, E_F]`. **No occupancy condition is imposed.** |
| `gamma point average splitting` | in **meV** (a different unit from `sse`): the mean of \|E↑−E↓\| over all bands at Γ |
| Descriptors | 52; the order is `FEATURE_ORDER` in `optimization/BO.py` |

**`data/magnetic_symmetry_all.csv`** and **`magnetic_symmetry_parents.csv`** give the
sublattice-connecting operation for every row and parent at `symprec = 0.05`. The `verdict`
column takes four values: `ALTERMAGNET` (2,841 rows), `conventional_AFM_inversion` (687),
`conventional_AFM_translation` (124) and `compensated_FiM_candidate` (193). The two
conventional-AFM classes are the symmetry-preserving controls retained in the training cohort;
they are not claimed to be altermagnets.

### The magnetic classification of the 322 parents, in one table

**`data/derived/altermagnet_classification_parents.csv`** (322 rows) and
**`data/derived/altermagnet_classification.csv`** (3,845 rows) join the magnetic symmetry,
the moments and the labels that are otherwise spread over five files. Built by
`python data/derived/make_classification_tables.py`; nothing in them is recomputed.

Per parent: `verdict`, `ops` (the operation relating the two spin sublattices), `n_ops`,
`nsym`, `spacegroup`, `msg_type`, `bns`, `uni`, and, aggregated over its strained children,
the SSE range, the Γ-point average, the moment range and the moment asymmetry. Per
structure the same, plus the signed `m1_muB` / `m2_muB` and the `antiparallel` flag.

What the tables say:

| | |
|---|---|
| parent verdicts | `ALTERMAGNET` 243, `conventional_AFM_inversion` 57, `compensated_FiM_candidate` 12, `conventional_AFM_translation` 10 |
| MSG type | type III 300, type I 12, type IV 9, unresolved 1 |
| sublattice operation | `mirror/rotoinv\|rotation` 211, `inversion\|mirror/rotoinv\|rotation` 22, `rotation` 19, none found 12 |
| **strain changes the verdict** | **87 of 3,524 strained children (2.5 %)**; 78 of them are parents classified `ALTERMAGNET` whose strained child falls back to `compensated_FiM_candidate` |
| **the two local moments** | known for **all 3,845**; antiparallel in **3,839 (99.8 %)**, and \|\|m₁\|−\|m₂\|\| never exceeds **0.006 μB** (median 0.0000, p99 0.0040) |
| the six that are not antiparallel | every one of them has \|m\| ≤ 0.008 μB — the weakly spin-polarized numerical solutions the referee asks about, now identifiable rather than hidden. 101 structures in all have \|m₁\| < 0.1 μB; the median over the cohort is 2.76 μB, and 2.59 μB over the rows classified `ALTERMAGNET` |
| Γ-point average | of **absolute** differences; median 0.89 meV, 90th percentile 3.68, max 9.96 |

**Signs, and where the moments come from.** `fin_data.csv` carries magnitudes - its
`ion1 tot` is \|m₁\| and its `tot_mag` is \|m_total\|. Three files carry the signed pair, all
of them reading the last `magnetization (x)` block of the OUTCAR through the same
`check_magnetization()` in `screening/parse_eigenval.py`:

| File | Rows | How it was made |
|---|---|---|
| `data/raw/local_moments.csv` | 4,211 | `screening/extract_moments.py` re-reads every stored OUTCAR. This is what the tables use |
| `data/raw/spin_splitting_summary.csv` | 3,207 | the direct output of `screening/SSE_ax.py` during screening; the fallback for structures whose OUTCAR is not on hand |
| `data/raw/magnetization.csv` | 4,491 | \|m₁\| only, the column that became `fin_data.csv`'s `ion1 tot` |

They agree. Over the 3,292 dataset structures whose OUTCAR was re-read, \|m₁\| reproduces
`fin_data.csv` exactly for 3,290; the two exceptions are `POSCAR_Cr2F4_cluster2` (3.793
against 3.795) and `POSCAR_Cr2F8_cluster2_st050` (2.250 against 2.251), which the build
script prints. Together the three files give both moments for all 3,845 structures.

The OUTCARs themselves are about 78 GB and are not deposited; `screening/extract_moments.py`
is, and takes `--root` or `$OUTCAR_ROOT`.

**`structures/POSCARS.zip`** holds 5,945 structures = 1,010 unstrained + 4,935 strained
variants of those 1,010 parents. The 322 parents and 3,845 structures that passed screening are
the ones listed in `fin_data.csv`. Strain suffixes take the 17 forms
`_(st|x|y|z)(025|050|950|975)`; **only `POSCAR_Cu2O2_1_st05` uses a two-digit suffix**, and that
file duplicates `_st050` and was excluded from the dataset (see below).

## Reproducing

```bash
# recompute descriptors (from a directory containing POSCAR_* files)
python descriptors/run_descriptors.py <structure-dir> <notebook-cell-dir> out.csv

# verify the model — check the feature order first
python model/model_check.py model/final_model_all_named.json data/fin_data.csv

# ablation (20 seeds x 5 folds; about 20 min on 20 cores including drop-column)
python model/ablation_grouped.py data/fin_data.csv --outdir out --workers 20

# cohort-composition tests
python analysis/cohort/am_only.py --data data/fin_data.csv --sym data/magnetic_symmetry_all.csv
python analysis/cohort/size_control.py --data data/fin_data.csv --sym data/magnetic_symmetry_all.csv

# symmetry classification
python symmetry/altermag_batch.py <structure-dir> names.txt out.csv 0.05
python symmetry/msg_type.py       <structure-dir> names.txt out.csv 0.05

# reviewer-response figures 1-8 (writes into figures/)
python analysis/point2_make_figures.py

# Ni-S convex hull: report, TSV and figures (reads validation/data/ only)
python validation/make_report.py
python validation/mk_tsv.py
python validation/hull_make_figures.py

# SHAP recomputation, then the reference SHAP plots
python figures/shap_recompute.py                 # writes shap_rank.csv, shap_heldout.npz, shap_meta.json
python figures/reference_plots/plot_shap_bar.py
python figures/reference_plots/plot_shap_figure.py
```

Every script resolves its inputs relative to its own file, so it runs from any working
directory. Two escape hatches exist for the parts that read data too large to deposit:

| Variable | Default | Used by |
|---|---|---|
| `NIS_HULL_DIR` | `~/NiS_hull` | `validation/hull_quick.py`, `analyze_uscan.py`, `repair.py`, `make_inputs.py` — the raw VASP run trees, which are **not** deposited |
| `FIN_DATA`, `SSE_VARIANTS`, `SHAP_OUTDIR`, `BOPY` | the deposited paths | overriding an input or output location |

## Known issues and caveats

- **`fin_data.csv` has 3,845 rows.** Six rows were removed from the initial 3,851-row release:
  five `Cr2F8_cluster3` rows (numerically identical duplicates of `cluster2`) and one
  `Cu2O2_1_st05` row (the same strain variant as `st050`, of unknown provenance). The parent
  count is correspondingly 323 → **322**.
- **The symmetry classification is stable for `symprec ≥ 0.05`.** The relaxed CONTCAR positions
  sit 10⁻³–10⁻⁴ off ideal Wyckoff positions, so detection breaks down at `symprec = 0.01`.
- **Magnetic space group type alone does not identify an altermagnet.** Type III contains both
  altermagnets and PΘ-symmetric conventional antiferromagnets; what decides is whether the
  connecting operation is an inversion or a rotation (`symmetry/altermag_sym.py`).
- **The 2D MPF face selection is numerically unstable when two face normals are nearly
  degenerate.** `descriptors/mpf_audit.py` quantifies the exposure, and
  `data/fin_data_mpf_deterministic.csv` is the table with the selection made deterministic.
- `packing_fraction` in `optimization/BO.py` is computed in 2D as area ÷ volume (1/Å), which
  differs from the training-side definition of area ÷ cell-face area (dimensionless). The
  residual effect on the square-planar candidate predictions is −13.6 to −14.7 meV.

## What runs from this repository alone

Everything under `figures/`, `analysis/`, `model/`, `optimization/`, `symmetry/` and
`descriptors/` reads only deposited files. So does the reporting half of `validation/`
(`make_report.py`, `mk_tsv.py`, `mu_test.py`, `hull_make_figures.py`), which reads
`validation/data/`; `mk_tsv.py` reproduces the deposited `hull_final.tsv` byte for byte.

The parts that cannot run from the deposit alone are the ones that consume raw VASP output:
`validation/hull_quick.py`, `analyze_uscan.py`, `repair.py` and `make_inputs.py` need the run
trees under `$NIS_HULL_DIR`, and `screening/`, `vasp/` and the `descriptors/` batch driver need
the calculation directories they were written for. Their *results* are deposited, so every
number in the manuscript can be checked without rerunning them.

## Computational environment

DFT used VASP 5.4.4 (candidates and substitution scans) and 6.4.2 (convex hull, HSE06); phonon
calculations used phonopy 2.48.0. Python package versions are pinned in `requirements.txt` and
`environment.yml`.

`environment.yml` pins **Python 3.13**, which is what the training and evaluation
(`model/`, `optimization/`) were reproduced on. The figure code was additionally checked on
**Python 3.12.3 with matplotlib 3.11.1**, which is why `requirements.txt` asks only for
`matplotlib>=3.8` rather than pinning it: below 3.8 the figures still draw, but the panel
proportions and tight-crop boxes shift by a fraction of a point, so a set of panels rendered
on an older matplotlib does not compose cleanly with one rendered on a newer one.

## Citation

If you use this code or data, please cite the archived release:

> Chun, K. and Kim, G. *Altermagnet inverse design - code and data.* Zenodo.
> https://doi.org/10.5281/zenodo.19488475

The same details are in [`CITATION.cff`](CITATION.cff), which GitHub reads for its
"Cite this repository" button. It has no `repository-code:` line yet - add the repository
URL there once it exists.

## License

MIT - see [`LICENSE`](LICENSE).
