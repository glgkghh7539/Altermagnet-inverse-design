#!/usr/bin/env python3
"""Crop the VESTA renders in `final_structure/` into `figures/structures/`.

`final_structure/POSCAR_<name>.png` is a 4130 x 1995 VESTA export: the structure sits in
the middle of a wide white canvas with the a/b/c axis triad off to the left. The composites
place the structure on a fixed rectangle, so any white around it becomes white in the
figure - it has to be cropped, and the triad has to go with it, exactly as in the published
figures.

The crop is found rather than typed in. Ink is anything either coloured (max - min channel
above a threshold, which finds the atoms, bonds and polyhedra but not the grey axis text)
or genuinely dark. Its columns fall into two groups with white between them - the triad and
the structure - and the wider group is the structure. A small margin is added so the
outermost atoms are not clipped by their own antialiasing.

    python figures/crop_structures.py            # all ten -> figures/structures/
    python figures/crop_structures.py FeS NiS

CuO, FeSi, VO and CrSb have no `.vesta` file and no render in `final_structure/`, so they are
supplied by hand as `figures/structures/<name>.png` and cropped **in place**: drop a fresh
VESTA export over the old file, re-run this, and the composites pick it up. Re-running on an
already-cropped file is harmless - the ink box is the same, so the crop is the same.

Cropping is no longer needed for the composite to place a structure correctly - it measures
the ink itself - but it is what keeps the files small. A 9300 x 3612 export with the structure
in the middle is 33 Mpx of mostly nothing.
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
# The VESTA exports. figures/final_structure/ is the copy that ships here; the working
# tree this was developed in keeps them one level up as well.
SRC = next((d for d in (os.path.join(HERE, 'final_structure'),
                        os.path.join(HERE, '..', '..', 'final_structure'))
            if os.path.isdir(d)), os.path.join(HERE, 'final_structure'))
DST = os.path.join(HERE, 'structures')

# Cropped out of final_structure/POSCAR_<name>.png
VESTA = ('CoO', 'CoS', 'CrS', 'FeAs', 'FeS', 'NiS')
# Supplied by hand as figures/structures/<name>.png and cropped in place, because these
# four have no .vesta file and no render in final_structure/ to crop from
HAND = ('CuO', 'FeSi', 'VO', 'CrSb')
NAMES = VESTA + HAND
SAT = 12          # channel spread above which a pixel counts as coloured ink
DARK = 200        # or a maximum channel below which it counts as dark ink
GAP = 0.02        # column runs closer than this fraction of the width are one object
MARGIN = 0.015    # of the crop's longer side, added on every edge


def ink(a):
    return ((a.max(2) - a.min(2)) > SAT) | (a.max(2) < DARK)


def structure_box(m):
    """(x0, y0, x1, y1) of the widest run of ink columns - the structure, not the triad."""
    cols = m.any(0)
    runs, i = [], 0
    while i < len(cols):
        if cols[i]:
            j = i
            while j < len(cols) and cols[j]:
                j += 1
            runs.append([i, j - 1])
            i = j
        else:
            i += 1
    if not runs:
        raise ValueError('no ink found')
    gap = int(len(cols) * GAP)
    merged = [runs[0]]
    for s, e in runs[1:]:
        if s - merged[-1][1] < gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    x0, x1 = max(merged, key=lambda r: r[1] - r[0])
    rows = np.where(m[:, x0:x1 + 1].any(1))[0]
    return x0, rows[0], x1, rows[-1]


def source_of(name):
    """Where `name` is cropped from: its VESTA export, or its own file, cropped in place."""
    vesta = os.path.join(SRC, f'POSCAR_{name}.png')
    if os.path.isfile(vesta):
        return vesta
    return os.path.join(DST, f'{name}.png')


def crop(name):
    src = source_of(name)
    if not os.path.isfile(src):
        print(f'  {name:5s} no source - skipped')
        return
    im = Image.open(src)
    a = np.asarray(im.convert('RGB')).astype(int)
    x0, y0, x1, y1 = structure_box(ink(a))
    pad = int(round(max(x1 - x0, y1 - y0) * MARGIN))
    box = (max(0, x0 - pad), max(0, y0 - pad),
           min(a.shape[1], x1 + 1 + pad), min(a.shape[0], y1 + 1 + pad))
    out = os.path.join(DST, f'{name}.png')
    # keep transparency if the render has it; the composites place it on white either way
    cropped = im.crop(box)
    im.close()
    cropped.save(out)
    w, h = box[2] - box[0], box[3] - box[1]
    print(f'  {name:5s} {a.shape[1]}x{a.shape[0]} -> {w}x{h}  '
          f'(aspect {w / h:.3f})   {os.path.relpath(out, os.path.dirname(HERE))}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('names', nargs='*', metavar='NAME', default=None)
    a = ap.parse_args()
    os.makedirs(DST, exist_ok=True)
    for n in (a.names or NAMES):
        crop(n)
    return 0


if __name__ == '__main__':
    sys.exit(main())
