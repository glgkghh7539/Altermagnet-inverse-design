# NiS Formation Energy — Ni–S Convex Hull

Where NiAs-type NiS (α-NiS, the altermagnet candidate) sits thermodynamically, quantified in a
form directly comparable with the Materials Project.

- Calculations: VASP 6.4.2 — 9 phases for the hull, plus 40 runs for the Hubbard *U* scan
- Status: **complete**

---

## 1. Headline result

**NiAs-type NiS lies {nias_hull:.1f} meV/atom above the 0 K hull, Ef = {nias_ef:.4f} eV/atom**
(on the MP2020 correction scale).

Millerite, the low-temperature stable polymorph, lies {mil_hull:.1f} meV/atom above the hull, so
the separation between the two polymorphs is **{poly_gap:.0f} meV/atom** with millerite lower —
in agreement with experiment.

![convex hull](fig1_convex_hull.png)

---

## 2. Calculation settings

**Identical settings were applied to every phase.** That is the premise that makes the formation
energies comparable at all.

| Item | Value |
|---|---|
| Functional | PBE, **no Hubbard U** (kept comparable with MP) |
| POTCAR | PAW 54, `Ni_pv` / `S` |
| ENCUT | 520 eV, PREC=Accurate, LASPH, LREAL=.FALSE. |
| k-points | **KSPACING**: 0.25 for relaxation, 0.15 for static; KGAMMA=.TRUE. |
| Relaxation | ISIF=3, IBRION=2, NSW=120, EDIFFG=−0.01, ISMEAR=1/σ=0.2, **two passes** |
| Static | NSW=0, ISMEAR=−5 (tetrahedron), EDIFF=1e-6, LORBIT=11 |
| Spin | ISPIN=2, ferromagnetic initialization (MAGMOM Ni 5.0 / S 0.6) |

k-points are set through KSPACING because the cells differ greatly in size — from one atom for
Ni to 34 for Ni₉S₈ — and the comparison requires the **same k-point density**, not the same
grid. Fixing a grid per phase by hand would not be a like-for-like comparison. The grids that
result run from 21×21×21 for Ni (static) down to 5×5×5 for α-S.

The relaxation is run in two passes because the plane-wave basis of an ISIF=3 run is fixed to
the starting cell (Pulay stress). Restarting from the CONTCAR of the first pass regenerates the
basis at the current volume.

---

## 3. Full results

{hull_table}

★ marks a phase on the hull. Ef is in eV/atom and E_hull in meV/atom. `Ef_raw` is the plain PBE
value; `Ef (MP2020)` has the MP2020 sulfide anion correction applied.

**Stable phases**: Ni, Ni₃S₂, Ni₉S₈, Ni₃S₄, NiS₂ (P2₁/c), S — the same set as MP.

Decomposition paths of the metastable phases:

```
NiS millerite     15.0 meV/atom  ->  0.708 Ni9S8 + 0.292 Ni3S4
NiS2 pyrite       15.6 meV/atom  ->  NiS2 (polymorph transition to P2_1/c)
NiS NiAs-type    106.8 meV/atom  ->  0.708 Ni9S8 + 0.292 Ni3S4
                                     (= 1 NiS -> 1/12 Ni9S8 + 1/12 Ni3S4)
```

---

## 4. The MP2020 correction — the key finding of this work

The formation energies from plain PBE came out **higher than the MP values by exactly 0.503 eV
per S atom.** That value coincides with the Materials Project's MP2020 sulfide anion correction,
which we confirmed directly in pymatgen.

With the correction applied, the calculation reproduces MP to **{mad:.1f} meV/atom on average**
across {n_dev} phases (range {dev_lo:+.1f} to {dev_hi:+.1f}). The hull distances agree too:
millerite 15.0 against MP's 14.2, pyrite 15.6 against 15.2, and NiAs-type 106.8 against 104.7
for mp-594.

![MP validation](fig2_mp_validation.png)

> **The column to quote is `Ef (MP2020)`.** Comparing raw PBE values with MP leaves a systematic
> error proportional to the S content.

---

## 5. E_hull does not depend on the reference states

Ef shifts with the choice of reference chemical potentials (μ_Ni, μ_S), but **E_above_hull does
not.**

An error δ in μ_S shifts every Ef by −x_S·δ. That is affine in composition, and the convex hull
construction is invariant under an affine transformation, so neither the set of hull vertices
nor the distance of any phase above the hull moves.

Numerical confirmation (`mu_test.py`), forcing μ_S by ±0.3 eV:

```
  d(mu_S)   NiAs-type Ef    E_hull    |  stable set
   -0.30      -0.1159      106.8 meV  |  unchanged
    0.00      -0.2659      106.8 meV  |  unchanged
   +0.30      -0.4159      106.8 meV  |  unchanged
```

**So computing α-S with PBE and no van der Waals correction does not affect the conclusion.**
α-S is a molecular crystal of S₈ rings, and its cell is over-expanded in PBE without vdW (+38 %
against experiment), but that error enters only μ_S and cancels in E_hull. MP computes mp-77 the
same way, so adding vdW would in fact break the comparison with MP.

The cancellation does **not** hold where elemental S appears in the equilibrium itself — an S
partial pressure, or a decomposition such as NiS₂ → NiS + S — and vdW would matter there. The
decomposition path of the NiAs-type phase involves neither elemental S nor Ni metal.

---

## 6. Hubbard U — it must not be used for the thermodynamics

Since the phonon calculation uses U = 7, we checked whether the hull should use U as well.
**It should not.**

Because E_hull here is a distance to the Ni₉S₈–Ni₃S₄ tie-line, the elemental references cancel,
which lets us apply U to **the seven nickel sulfides only** and avoid the unphysical situation of
putting U on nickel metal. The scan is U = 0, 2, 4, 6, 7 × {{7 sulfides FM + NiAs-type AFM}} = 40
runs.

Self-consistency check: at U = 0 the sulfide-only construction gives 106.7 / 106.8 meV/atom,
matching the 106.8 of the full nine-phase hull.

{u_table}

Units are meV/atom. **A negative value would mean the NiAs-type phase is more stable than its
decomposition products, which is physically wrong** — α-NiS is a high-temperature phase and
cannot be stable at 0 K.

### Polymorph ordering — the decisive test

Experimentally millerite is the stable phase below about 379 °C.

{order_table}

**Only U = 0 reproduces experiment.** At U ≥ 2 the calculation predicts the high-temperature
phase to be more stable than the low-temperature one.

### Cell volume

Experimental α-NiS is about 13.7 Å³/atom (a ≈ 3.44, c ≈ 5.35 Å). U = 0 comes closest at 13.28
(−3 %), and the cell expands with U as the moment develops, reaching 15.26 (+11 %) at U = 7.

![U dependence](fig3_U_dependence.png)

### Summary

| Criterion | U = 0 | U ≥ 2 |
|---|:---:|:---:|
| Agreement with MP ({mad:.1f} meV/atom) | yes | not comparable |
| millerite below NiAs-type (experiment) | yes | reversed |
| Cell volume | −3 % | +2 to +11 % |

**Thermodynamics at U = 0; magnetism and phonons at U = 7.** Report the two side by side and do
not combine their free energies.

---

## 7. Magnetic states

Every sulfide on the hull converged to a **non-magnetic** solution from a ferromagnetic start;
only fcc Ni retained a moment, 0.63 μB. That is the expected behaviour for PBE without U and
matches MP's treatment.

NiAs-type NiS was additionally computed in the antiferromagnetic configuration used for the
phonon calculation (Ni at z = 0 up, z = ½ down):

| State | U | E₀ (eV / 4 atoms) | V (Å³) | m(Ni) μB |
|---|---:|---:|---:|---:|
| NM | 0 | −20.31556799 | 53.10 | 0.000 |
| AFM | 0 | −20.31544426 | 53.10 | **0.058** |
| AFM | 7 | −13.60008236 | 61.15 | **1.626** |
| FM | 7 | −13.39569832 | 62.43 | 1.728 |
| NM | 7 | −10.64006705 | 52.34 | 0.000 |

**At U = 0 the AFM solution barely exists** — a moment of 0.058 μB, and an energy 0.031 meV/atom
*above* NM, a degeneracy at the level of numerical noise. The value {nias_hull:.1f} meV/atom is
therefore a determination, not an upper bound.

At U = 7 a robust AFM state appears (1.63 μB), lying 51 meV/atom below FM and 740 meV/atom below
NM. The cell expansion is driven by **moment formation rather than by U itself**: at U = 7 the NM
solution actually contracts, to 52.34 Å³.

> Millerite has three Ni atoms per cell, an odd number, so a compensated collinear AFM state is
> impossible; it would need at least a doubled cell, and in R3m the three Ni sites are
> symmetry-equivalent, which makes collinear AFM geometrically frustrated. Millerite is however a
> Pauli-paramagnetic metal experimentally, so NM is its true ground state, and placing each phase
> in its own magnetic ground state — as done here — is the correct treatment.

---

## 8. Scope of the polymorph search

Twenty-seven structures were retrieved from MP and nine of them computed. The eighteen that were
not computed all lie above the hull according to MP, and their margin — at least 21 meV/atom once
elemental S is excluded — is far larger than this calculation's {mad:.1f} meV/atom deviation from
MP, so including them would not change the hull.

| Composition | Structures not used (MP E_hull, meV/atom) |
|---|---|
| Ni (1) | hcp P6₃/mmc (45.8) |
| NiS₂ (3) | Pnnm (21.3), R-3m (35.0), Fd-3m (43.8) |
| S (14) | P2₁ (0.4), P2/c (0.9), P2 (6.3), Pnnm (10.3), P2₁/c (12.3), … R-3 (51.2) |

The only compositions for which more than one structure was computed are **NiS (2) and NiS₂ (2)**.
MP likewise holds only two NiS entries, mp-594 (NiAs-type) and mp-1547 (millerite), so resting
the argument on "agrees with every MP entry" is the safe formulation.

---

## 9. Known limitations

- **Two of the 40 U-scan runs** are not fully converged. `U6/Ni₉S₈` had its force criterion
  relaxed to −0.05 eV/Å — the energy spread across the oscillating interval is 0.24 meV/atom, so
  it does not matter — and `U6/NiS₂ P2₁/c` exhausted NSW during relaxation, although its static
  electronic step converged normally. The latter is not a hull vertex at U = 6 (pyrite is
  0.239 eV/atom lower), so it does not enter any reported number.
- The decomposition products in the U scan are initialized ferromagnetically. At large U only the
  NiAs-type phase gains an AFM stabilization, so the reversal of the polymorph ordering may be
  partly an artefact of that asymmetry. It is nevertheless physically reasonable, since millerite
  carries no moment experimentally and α-NiS does.
- The MP2020 correction is an empirical value fitted to experimental formation enthalpies. When
  quoting an absolute Ef it is safer to state that it is on the same scale as MP.

---

## 10. Files

```
hull/
├── fig1_convex_hull.png         Ni-S convex hull
├── fig2_mp_validation.png       validation against MP (parity and deviation)
├── fig3_U_dependence.png        U dependence, three panels
├── hull_final.tsv               result table (15 columns)
├── hull_final.json              machine-readable
├── hull_final_report.txt        full report
├── uscan/                       U scan (report, figures, summary JSON)
└── scripts/                     the scripts needed to reproduce all of the above
```

Reproducing the analysis, from the directory holding `hull_final.json`:

```bash
python hull_quick.py      # table and hull figure
python mu_test.py         # reference-state independence
python analyze_uscan.py   # U scan
python make_report.py     # regenerate this document
```

These require pymatgen; version 2026.8.13 was used here, and the pinned environment is in the
repository root. One practical note for anyone rebuilding VASP: some compute partitions lack AVX
support, and a binary built with it dies immediately with `illegal instruction`, so check the
target architecture before building.
