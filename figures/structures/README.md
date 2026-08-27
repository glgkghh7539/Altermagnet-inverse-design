# `figures/structures/` — drop-in replacements for the crystal-structure images

`compose_figures.py` looks here first for the structure that goes above each band or
fatband panel. Put a file named after the compound and it is used; leave the folder empty
and the composite falls back to the image embedded in the published PDF.

| File | Panel | Cropped from | ppi in the composite |
|---|---|---|---|
| `CuO.png` | Fig. 3d | itself — supplied by hand | 718 |
| `FeSi.png` | Fig. 3e | itself | 882 |
| `VO.png` | Fig. 4c | itself | 620 |
| `CrSb.png` | Fig. 4d | itself | 636 |
| `FeS.png` | Fig. 5a | `figures/final_structure/POSCAR_FeS.png` | 449 |
| `NiS.png` | Fig. 5b | `figures/final_structure/POSCAR_NiS.png` | 509 |
| `CoO.png` | SI S5a | `figures/final_structure/POSCAR_CoO.png` | 444 |
| `CrS.png` | SI S5b | `figures/final_structure/POSCAR_CrS.png` | 470 |
| `CoS.png` | SI S6a | `figures/final_structure/POSCAR_CoS.png` | 490 |
| `FeAs.png` | SI S6b | `figures/final_structure/POSCAR_FeAs.png` | 489 |

The published figures carry all ten at about 200 ppi.

All ten are cropped by the same script:

```bash
python figures/crop_structures.py               # all ten
python figures/crop_structures.py CuO FeSi      # just these
```

The six with a `figures/final_structure/POSCAR_*.png` are cropped out of it. The other four have no
`.vesta` file and no render anywhere in the project, so they are supplied by hand and cropped
**in place**: drop a fresh VESTA export over `CuO.png` and re-run, and the composite picks it
up. Re-running on an already-cropped file changes nothing — the ink box is the same. The
`.vasp` files below are the structures themselves, for re-rendering those four.

## What to provide

A VESTA render, PNG. Transparent or white background both work, and the crop does not have
to be exact: the composite measures where the ink starts and stops inside the file and places
*that* on the published rectangle, so a margin of your own simply hangs outside it. Cropping
close still helps — it is what keeps the file small.

Resolution is the only thing that limits sharpness here. The rectangles the composite
uses are about 1.8 x 2.9 in for the Fig. 3 pair and 2.3 x 2.9 in for the Fig. 4 pair, so:

| Render width | Resulting ppi | |
|---|---|---|
| 400 px | ~220 | about what the published figure has |
| 900 px | ~500 | comfortably past what any journal asks |
| 4130 px | ~2300 | what the `final_structure/POSCAR_*.png` renders are |

The six structures in `../../final_structure/` (CoO, CoS, CrS, FeAs, FeS, NiS) are already
4130 x 1995 px, which is where that last row comes from. Matching that for CuO, FeSi, VO
and CrSb would put every structure in the paper at the same quality.

Nothing else needs changing — re-run `python figures/compose_figures.py` and it will say
which files it picked up.

## The structures themselves, for re-rendering

`VO.vasp`, `CrSb.vasp`, `CuO.vasp` and `FeSi.vasp` are here so the four missing renders can
be made without hunting for the cells. Each is the **relaxed** structure the calculation
produced - the CONTCAR, taken from `../../structures/POSCARS.zip` - with the fractional
coordinates wrapped into [0, 1); the cell is exactly as calculated. The first line of each
file records which panel it belongs to, its descriptor values, and the name it has in the
archive.

| File | Panel | Cell | Space group (symprec 0.05) |
|---|---|---|---|
| `VO.vasp` | Fig. 4c | V₂O₂, a = 3.101, c = 5.293 Å | **P6₃/mmc (#194)** |
| `CrSb.vasp` | Fig. 4d | Cr₂Sb₂, a = 4.274, c = 5.861 Å | **P6₃/mmc (#194)** |
| `CuO.vasp` | Fig. 3d | Cu₂O₂, a = 2.759, c = 4.928 Å | C2/c (#15) |
| `FeSi.vasp` | Fig. 3e | Fe₂Si₂, a = 3.147, b = 2.991, c = 5.120 Å | Pmmn (#59) |

VO and CrSb come out in the same space group, which is the point of that comparison in the
text - they are evaluated in the same P6₃/mmc prototype, so the difference between them is
covalency rather than geometry. CuO and FeSi sit at the two ends of the MSBI range, C2/c
distorted against Pmmn symmetric.

In every file the two magnetic atoms come first and carry opposite moments; that is what
the arrows in the published renders show. `fatband/CrSb/POSCAR` is the same CrSb cell with
its off-diagonal lattice components zeroed and its coordinates wrapped - the same
structure, tidied for the band run.
