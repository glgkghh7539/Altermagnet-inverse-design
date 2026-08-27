# `plotdata/` — flat tables for redrawing the figures

These are not plotting scripts. They are **flat CSV files ready to plot**. All of them are
keyed to the canonical cohort in `../../data/fin_data.csv` (3,845 rows, 322 parents, md5
`ff17adab…`), and every `sse` column in this directory is that table's.

Their provenance differs by file. The SHAP tables come from `../shap_recompute.py`; the parity
and decile tables from `../reference_plots/plot_si3_parity.py`; the descriptor scatters are
slices of `fin_data.csv` itself. Two are not derived from the descriptor table at all:
`fig4b_pd_hybridization.csv` is extracted from the archived VASP `PROCAR` files, and
`si4_bo_progress.csv` is the Optuna trial history.

**SHAP values are given in the training space, `log1p`.** The manuscript's conversion to eV by
local linearization has been withdrawn: the linearization is state-dependent, so the rescaled
quantities no longer sum to the model output. Attributions are also computed **only on the
held-out folds** of a parent-grouped `GroupKFold(5)` rather than in-sample on a model fitted to
all rows, and averaged over 20 seeds. Those two changes together account for the difference
between the manuscript's top-three total of 37.4 % and the revised 42.04 %.

## Files

### `fig2a_shap_ranking.csv` — Fig. 2a bars, Table 1 (52 rows)

| Column | Meaning |
|---|---|
| `rank` | rank by descending mean\|SHAP\| |
| `feature` | dataset column name |
| `symbol` | symbol used in the manuscript (MSBI, MPF, sigma_inhom, d_CC_1 …); for descriptors with no manuscript symbol this repeats `feature` |
| `mean_abs_shap`, `mean_abs_shap_sd` | mean and standard deviation over 20 seeds, in `log1p` units |
| `share_pct`, `share_pct_sd` | share of the total (%) |
| `cumulative_pct` | running total (%) |

Total Σ mean\|SHAP\| = **0.3240**, top three = **42.04 %**, top twelve cumulative = **75.50 %**.
`table_shap_top12.csv` is the first twelve rows of this table.

### `fig2b_beeswarm_long.csv` — Fig. 2b beeswarm (46,140 rows = 12 features × 3,845)

Long format: group by `feature` and draw one row of the swarm per group.

| Column | Meaning |
|---|---|
| `filename`, `sse` | row identifier and DFT SSE (eV) |
| `feature`, `symbol`, `rank` | the feature and its rank in Fig. 2a |
| `feature_value` | raw feature value for that row — **use for colour** |
| `shap` | held-out SHAP value for that row, in `log1p` units — **use for horizontal position** |

Colour is normally normalized to the 1st–99th percentile of the feature value.

### `fig3abc_scatter.csv` — Fig. 3 a–c (3,845 rows)

Scatter of `sse` against `msbi` / `mpf` / `pd_ratio`, with point colour taken from the matching
`*_shap` column. The manuscript uses a logarithmic *x* axis. **Some rows have `msbi` exactly
zero, so they must be excluded or a lower bound imposed before taking the logarithm.**

### `fig4a_packing_scatter.csv` — Fig. 4a (3,845 rows)

| Column | Meaning |
|---|---|
| `d_CC_1` | `labelled_1st` — distance **between the two magnetic sites** = d_CC⁽¹⁾ in the manuscript |
| `d_MM_1` | `global_1st` — nearest metal–metal distance including periodic self-images = d_MM⁽¹⁾ |
| `mpf`, `msbi`, `pd_ratio`, `dimension` | for colour and filtering |

Three independent checks support that identification: (i) `labelled_*` counts only M1↔M2
cross pairs whereas `global_*` includes self-images (`descriptors/descriptor.ipynb`, cell 3);
(ii) `labelled_1st ≥ global_1st` holds for **all** 3,845 rows; and (iii) the attribution ranks
(labelled 6th, global 11th) reproduce the manuscript's ordering of d_CC 5th above d_MM 8th.
The caption condition "d_CC⁽¹⁾ < 3.5 Å" selects 47.3 % of rows on `d_CC_1` and 62.8 % on
`d_MM_1`.

### `fig4b_pd_hybridization.csv` — Fig. 4b (7,690 rows = 3,845 structures × 2 spins)

One row per (structure, spin channel), extracted from the archived `PROCAR` files at the
(k-point, band) where the SSE is attained. Pivot on `spin` to get one point per structure:

    H_max   = max(H_up, H_dn)          delta_H = |H_up - H_dn|

with `H = pd_hybrid_minimum`. `plot_fig4b.py` does exactly this.

| Column | Meaning |
|---|---|
| `filename` | structure; the cohort key, matching `data/fin_data.csv` |
| `spin` | `up` or `down`. Not read from a spin header: at the target (k, band) the first ion table is up and the second is down |
| `kp_idx`, `band` | the PROCAR k-index and band index at which the SSE occurs |
| `mag_elem`, `nonmag_elem` | the metal and the ligand |
| `s_mag_tot`, `d_mag_tot` | summed s and d weights over the two magnetic ions (ions 1 and 2) |
| `s_nonmag_tot`, `p_nonmag_tot` | summed s and p weights over the ligands (ions 3 and up) |
| `s_tot`, `p_tot`, `d_tot` | `s_mag+s_nonmag`, `p_nonmag`, `d_mag`. **`p_tot` is ligand p only and `d_tot` is metal d only** — that asymmetry is the definition, not an omission |
| `total_occ` | `s_tot + p_tot + d_tot` |
| `s_global`, `p_global`, `d_global` | the three above divided by `total_occ`; they sum to 1 |
| **`pd_hybrid_minimum`** | **`2 * min(p_global, d_global)`** — the H_σ of the manuscript; **this is the column Fig. 4b uses** |
| `pd_hybrid_geometric` | `2 * sqrt(p_global * d_global)`, an alternative not used in the paper |
| `pd_hybrid_product` | `4 * p_global * d_global`, likewise unused |
| `p_to_d_ratio` | `p_tot / d_tot`, **empty where `d_tot` is 0** (2 rows: `Cu2O2_1_st050` and `Cu2O2_1_z050`, spin down, which carry no metal d weight at that band). Drop or mask these before plotting |
| `sse`, `pd_ratio` | joined from `fin_data.csv` for colouring and for the p/d split |

`H` is bounded by construction: `p_global + d_global <= 1`, so `2*min(...) <= 1`, reached only
at perfect 1:1 mixing. The two annotated case studies of the manuscript are
`POSCAR_O2V2_3` (VO: p/d = 1.000, SSE = 0.165 eV, H_max = 0.412) and `POSCAR_Cr2Sb2_1`
(CrSb: p/d = 0.600, SSE = 1.194 eV, H_max = 0.808, delta_H = 0.514).

The table was regenerated against the canonical cohort: five `Cr2F8_cluster3` structures that
predate the deduplication were removed, and `sse` was taken from `fin_data.csv`. The
PROCAR-derived columns were re-extracted from the archive and match to machine precision.

### `si4_bo_progress.csv` — SI Fig. S4, BO convergence (99,960 rows)

The Optuna trial history of the inverse-design search.

| Column | Meaning |
|---|---|
| `trial_number` | 0 to 99,999; **40 trials are absent**, having been pruned or failed |
| `value` | the surrogate's predicted SSE for that trial, in eV |
| `best_sse_overall` | Optuna's best-so-far as recorded when the trial completed |

Two things to know before plotting. `best_sse_overall` is **not exactly** a running maximum of
`value`: it disagrees on 24 of 99,960 rows and steps backwards 9 times, by at most 0.046 eV and
typically 0.009 eV. That is the usual artefact of a parallel study, where a worker writes its
`best_value` before another worker's better trial has been committed. Counting improvements
from this column gives 32 new bests; counting them from the cumulative maximum of `value` gives
25. The endpoints agree either way — the search runs from 0.513 eV to **1.061 eV**, attained at
trial 42,747 — so the convergence curve is unaffected in shape, but a strictly monotone line
should be drawn as `value.cummax()` rather than from this column.

### `si3_oof_predictions.csv` — SI Fig. S3a parity (3,845 rows)

`filename`, `sse_dft`, `sse_pred`. Out-of-fold predictions from a parent-grouped
`GroupKFold(5)`, **averaged over 20 seeds row by row**. Scored on these seed-averaged
predictions, R²(eV) = **0.7168** and MAE = **117.7 meV**. This is not the same estimator as the
headline figure: pooling each seed separately and then averaging the 20 resulting values gives
R²(eV) = 0.6933 ± 0.0171. Seed averaging removes model-to-model variance, so the parity plot
necessarily looks slightly better than the headline number. Both are stated in the
Supplementary Information.

### `si3_decile.csv` — SI Fig. S3b decile bias (10 rows)

`decile`, `n`, `sse_mean`, `pred_mean`, `bias_pct`, computed from the same seed-averaged
predictions: 1st decile **+515 %** → 7th **+0.1 %** → 10th **−31.5 %**.

## Which figures need redrawing

| Figure | Redraw | Reason |
|---|---|---|
| Fig. 1 | no | schematic, not generated by code |
| **Fig. 2 a·b + Table 1** | **yes** | SHAP space and computation both changed |
| **Fig. 3 a–c** | **yes** | point colour is a SHAP value; the caption said "eV" |
| Fig. 3 d–e | no | CuO and FeSi band structures, independent of SHAP |
| Fig. 4a | check | independent of SHAP; **confirm only that the abscissa is `d_CC_1`** |
| Fig. 4b | check | PROCAR-derived and independent of the model, but the table itself was regenerated against the 3,845-row cohort (five duplicate structures removed, `sse` taken from `fin_data.csv`) |
| Fig. 4 c–d, Fig. 5 | no | fatbands, unaffected |
| **SI Fig. S3** | **yes** | depends on model and cohort; cited directly by the reviewer |
| SI Fig. S1 | check | 3,851 → 3,845 (six rows; no visible change) |
| SI Fig. S2 | no | one cell of 160 combinations shifts by 0.016 meV |
| SI Fig. S4 | no | BO convergence; the search was not re-run |
| SI Figs. S5–S6 | no | candidate structures and bands, unaffected |

`../reference_plots/` contains the plots we drew to verify these numbers together with the
scripts that produced them. **They do not follow the formatting of the published figures** and
should be used only to check values.
