# `data/` — what is here and how it is layered

Three tiers, and the tier a file sits in tells you what it is.

| Tier | Where | One row is | Written by |
|---|---|---|---|
| **Cohort** | `data/` | a structure, with its descriptors and its label | the descriptor and screening pipeline |
| **Per-structure properties** | `data/raw/` | a structure, with one family of measured or determined quantities | one parser or one determination, each named below |
| **Joins and aggregates** | `data/derived/` | whatever the join produces — a structure, a parent, a k-point | a script that only combines the two tiers above |

Nothing in `derived/` recomputes a physical quantity. If a number appears there it was carried
over from `data/` or `data/raw/`, and the script that did it says so.

**Five files sit at the top level that by this rule belong in `raw/`** — the magnetic-symmetry
and magnetic-space-group tables. They stay because the manuscript's Data Availability statement
cites them by path (`data/magnetic_symmetry_*.csv`,
`data/magnetic_spacegroup_type_parents.csv`). Moving them would make a published sentence wrong.
They are listed under *Per-structure properties* below, which is what they are.

---

## Cohort

| File | Rows | What it is |
|---|---:|---|
| `fin_data.csv` | 3,845 | **the table everything is trained and reported on.** 322 parents, 52 descriptors, `sse` (eV), `gamma point average splitting` (meV), `ion1 tot` and `tot_mag`, and `parent`, the `GroupKFold` group key. md5 `e6efa647c916f6b4ab74efac02849e7b` |
| `fin_data_mpf_deterministic.csv` | 3,845 | the same with the two-dimensional MPF face selection made deterministic. Differs materially in one row; refitting on it changes no reported result. md5 `2ddbb1d9c98c6d6a7bb767368083a0f4` |
| `prescreening_table.csv` | 4,483 | the table **before** screening, keeping the Γ average and total magnetisation of the rows the filter removed, so the four thresholds can be checked independently. The 3,845 are a strict subset. md5 `ebbf7b479368f78aa3885926ba32de6d` |
| `Altermagnetism_full_data.csv` | 4,507 | the earlier full extraction the cohort was cut from |
| `candidate_master.csv` | 6 | the six candidates: space group, MSG, cell, coordinates, both local moments, ΔE<sub>FM</sub>, ΔE<sub>NM</sub>, SSE |

### Two things to know about the columns

**`ion1 tot` and `tot_mag` are magnitudes.** `ion1 tot` is |m₁| and `tot_mag` is |m_total|. The
signed values, and the second moment, are in `raw/local_moments.csv`.

**`sse` and the Γ column are in different units** — eV and meV respectively. `sse` is the
maximum of |e↑ − e↓| over same-band-index pairs for which **both** eigenvalues lie in
[E_F − 2 eV, E_F], with no occupancy condition.

---

## Per-structure properties

| File | Rows | What it is | Written by |
|---|---:|---|---|
| `magnetic_symmetry_all.csv` | 3,845 | the operation connecting the two magnetic sublattices, and the verdict it implies, at **symprec 0.05** | `symmetry/altermag_batch.py` |
| `magnetic_symmetry_all_symprec001.csv` | 3,845 | the same at **symprec 0.01** | the same, fourth argument `0.01` |
| `magnetic_symmetry_parents.csv` | 322 | the same for the parents only, symprec 0.05 | the same |
| `magnetic_spacegroup_type_parents.csv` | 322 | MSG type, BNS and UNI numbers, moments given as **collinear scalars** | `symmetry/msg_type.py` |
| `magnetic_spacegroup_type_parents_c.csv` | 322 | the same with the moments along **c**, the direction the calculations impose. 96 of the 322 differ | the same, fifth argument `c` |
| `raw/local_moments.csv` | 4,211 | **both signed moments** and the total, read from the last `magnetization (x)` block of every stored OUTCAR | `screening/extract_moments.py` |
| `raw/magnetization.csv` | 4,491 | \|m₁\| only — the column that became `fin_data.csv`'s `ion1 tot` | the screening pipeline |
| `raw/spin_splitting_summary.csv` | 3,207 | the screening run's own signed m₁, m₂, m_total together with the splitting it found | `screening/SSE_ax.py` |
| `raw/sse_variants_all.csv` | 3,845 | the splitting under **ten** definitions — max, P99/P95/P90/P50, two BZ averages, and both window conventions | the SSE extractor |
| `raw/orbital_match_all.csv` | 3,851 | rank pairing against the orbital-character-optimal pairing, as `cos_l_sum`, `cos_l_ion`, `cos_l_swap`, `cos_m_sum` | the band-pairing audit |
| `raw/convergence_126runs.csv` | 126 | the k-density, ENCUT, SIGMA and EDIFF sweep | the convergence study |

### Which SSE column is the published one

In `raw/sse_variants_all.csv` and `derived/candidates_sse.csv`, read **`sse_max_win_both`**.
`sse_max` is the looser convention — the maximum over pairs with *either* eigenvalue inside the
window. They coincide on 3,341 of the 3,845 rows and differ on 504. Against `fin_data.csv`'s
`sse`, `sse_max_win_both` agrees to 1 meV on 3,817 rows and `sse_max` on 3,497. Among the six
candidates only CrS differs: **0.411 eV** under the published definition, 0.443 under the loose
one.

---

## Joins and aggregates

| File | Rows × cols | What it joins |
|---|---:|---|
| `derived/altermagnet_classification.csv` | 3,845 × 29 | one row per structure: the symmetry verdict and connecting operation at **both** tolerances, the parent's MSG with and without a Néel axis, both signed moments with an antiparallel flag and the moment asymmetry, the SSE and the Γ average |
| `derived/altermagnet_classification_parents.csv` | 322 × 25 | one row per parent: the same, aggregated over its strain variants, plus how many children change verdict under strain |
| `derived/typeA_all.csv` | 3,851 | the band-pairing audit per structure — `type` is `rank_ok` (3,635) or `A` (216, where the two pairings pick different bands) |
| `derived/candidates_sse.csv` | 6 | the ten SSE definitions for the six candidates |
| `derived/shap200*.csv` | 52 each | SHAP sweeps under alternative targets |

`derived/make_classification_tables.py` builds the first two. It recomputes nothing, it stops if
a join drops a row, and it checks the signed moments against `fin_data.csv`'s magnitudes — two
structures differ by 0.002 and 0.001 μB and it prints them rather than passing over them.

```bash
python data/derived/make_classification_tables.py
```

---

## Regenerating the per-structure tables

```bash
# the symmetry classification, at either tolerance
python symmetry/altermag_batch.py <poscar dir> <name list> out.csv 0.05

# the magnetic space group, with or without a Neel axis
python symmetry/msg_type.py <poscar dir> <name list> out.csv 0.05 c

# both local moments, from a directory of finished calculations
python screening/extract_moments.py --root <dir of POSCAR_*/OUTCAR>
```

The POSCAR files are in `structures/POSCARS.zip`. The OUTCARs are about 78 GB and are not
deposited; `extract_moments.py` is, so the moments are regenerable wherever they are kept.
