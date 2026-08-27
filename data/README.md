# `data/`

Three tiers. Which tier a file sits in tells you what it is.

| | | One row is |
|---|---|---|
| **`data/`** | the cohort | a structure, with its descriptors and its label |
| **`data/raw/`** | one family of measured or determined quantities per file | a structure |
| **`data/derived/`** | joins of the two above — nothing here recomputes a physical quantity | a structure, a parent, or a k-point |

## `data/` — the cohort

| File | Rows | |
|---|---:|---|
| `fin_data.csv` | 3,845 | **what everything is trained and reported on.** 322 parents, 52 descriptors, `sse` (eV), `gamma point average splitting` (meV), `ion1 tot`, `tot_mag`, and `parent` — the `GroupKFold` key. md5 `e6efa647c916f6b4ab74efac02849e7b` |
| `fin_data_mpf_deterministic.csv` | 3,845 | the same with the MPF face selection made deterministic. One row differs materially; no reported result changes. md5 `2ddbb1d9c98c6d6a7bb767368083a0f4` |
| `prescreening_table.csv` | 4,483 | the table **before** screening, so the four thresholds can be checked. The 3,845 are a strict subset. md5 `ebbf7b479368f78aa3885926ba32de6d` |
| `Altermagnetism_full_data.csv` | 4,507 | the earlier full extraction the cohort was cut from |
| `candidate_master.csv` | 6 | the six candidates: space group, MSG, cell, coordinates, both moments, ΔE<sub>FM</sub>, ΔE<sub>NM</sub>, SSE |

## `data/raw/` — per-structure properties

| File | Rows | | Written by |
|---|---:|---|---|
| `magnetic_symmetry_all.csv` | 3,845 | the sublattice-connecting operation and its verdict, symprec **0.05** | `symmetry/altermag_batch.py` |
| `magnetic_symmetry_all_symprec001.csv` | 3,845 | the same at symprec **0.01** | ” , fourth argument `0.01` |
| `magnetic_symmetry_parents.csv` | 322 | the same, parents only | ” |
| `magnetic_spacegroup_type_parents.csv` | 322 | MSG type, BNS, UNI — moments as **collinear scalars** | `symmetry/msg_type.py` |
| `magnetic_spacegroup_type_parents_c.csv` | 322 | the same with moments along **c**; 96 of 322 differ | ” , fifth argument `c` |
| `local_moments.csv` | 4,211 | **both signed moments** and the total, from every stored OUTCAR | `screening/extract_moments.py` |
| `magnetization.csv` | 4,491 | \|m₁\| only — the source of `fin_data.csv`'s `ion1 tot` | the screening pipeline |
| `spin_splitting_summary.csv` | 3,207 | the screening run's own signed m₁, m₂, m<sub>total</sub> | `screening/SSE_ax.py` |
| `sse_variants_all.csv` | 3,845 | the splitting under ten definitions | the SSE extractor |
| `orbital_match_all.csv` | 3,851 | rank pairing vs. orbital-character-optimal pairing | the band-pairing audit |
| `convergence_126runs.csv` | 126 | k-density, ENCUT, SIGMA and EDIFF sweep | the convergence study |

## `data/derived/` — joins

| File | Rows × cols | |
|---|---:|---|
| `altermagnet_classification.csv` | 3,845 × 29 | per structure: verdict and operation at **both** tolerances, the parent's MSG with and without a Néel axis, both signed moments with an antiparallel flag and the asymmetry, SSE, Γ average |
| `altermagnet_classification_parents.csv` | 322 × 25 | the same per parent, aggregated over its strain variants |
| `typeA_all.csv` | 3,851 | the band-pairing audit: `type` is `rank_ok` (3,635) or `A` (216, the two pairings pick different bands) |
| `candidates_sse.csv` | 6 | the ten SSE definitions for the six candidates |
| `shap200*.csv` | 52 each | SHAP sweeps under alternative targets |

Built by `python data/derived/make_classification_tables.py`. It recomputes nothing, stops if a
join drops a row, and checks the signed moments against `fin_data.csv`'s magnitudes.

## Two things that are easy to get wrong

**`ion1 tot` and `tot_mag` are magnitudes** — \|m₁\| and \|m_total\|. Signed values and the second
moment are in `raw/local_moments.csv`.

**The published SSE is `sse_max_win_both`, not `sse_max`.** In `raw/sse_variants_all.csv` and
`derived/candidates_sse.csv`, `sse_max` is the loose window (*either* eigenvalue inside it).
They differ on 504 of 3,845 rows; among the six candidates only CrS — **0.411 eV**, not 0.443.

## Regenerating `raw/`

```bash
python symmetry/altermag_batch.py <poscar dir> <name list> out.csv 0.05   # or 0.01
python symmetry/msg_type.py       <poscar dir> <name list> out.csv 0.05 c # or collinear
python screening/extract_moments.py --root <dir of POSCAR_*/OUTCAR>
```

POSCARs are in `structures/POSCARS.zip`. The OUTCARs are ~78 GB and are not deposited.
