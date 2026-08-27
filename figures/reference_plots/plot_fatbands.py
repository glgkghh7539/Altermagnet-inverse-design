#!/usr/bin/env python3
"""Fig. 4c-d, Fig. 5 and SI Figs. S5-S6 - the fatband panels.

A plain runner for `archive/plotfat_original.ipynb`. It executes the notebook's code cells
in `figures/fatband/`, which is the relative layout they expect (`VO/`, `CrSb/` beside
them, `../final_structure/...` above), and collects the 15 PNGs they write.

The notebook is deposited as it was run, with one exception, recorded in
`archive/ORIGINAL_MD5.txt`: its colorbar labels were set with `rotation=270`, which reads
top-to-bottom and comes out upside down beside the colorbar numbers. They now use
matplotlib's default 90, so they read bottom-to-top like every other colorbar here - `MPF`
on Fig. 4a, `Maximum SSE (eV)` on SI S2, `<SSE> (eV)` on Fig. 4b.

    python figures/reference_plots/plot_fatbands.py

Inputs  : ../fatband/{VO,CrSb}/ and ../final_structure/*/ PBAND_SUM_{UP,DW}.dat
Output  : 15 PNGs, written into ../fatband/ (override with $FATBAND_OUTDIR)
"""
import json
import os
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK = os.path.join(HERE, '..', 'archive', 'plotfat_original.ipynb')
RUNDIR = os.path.join(HERE, '..', 'fatband')      # the layout the notebook expects
OUTDIR = os.environ.get('FATBAND_OUTDIR', RUNDIR)


def _keep_overwritten(rundir, before, seen, cell):
    """Keep a panel that a later cell is about to overwrite, tagged with its cell.

    Two cells of this notebook write `SI_fatband_combined_CoO_CrS.png`: cell 6, whose
    output is the one in the published SI Fig. S5, and cell 8, a re-run of the same pair
    with the third k-label changed from `U` to `U$_2$`. Running the notebook straight
    through leaves only cell 8's, and the published panel cannot be reproduced at all.
    Every version is now kept: the last one under the notebook's own name, the earlier
    ones as `<stem>__cell<N>.png`.
    """
    for f in sorted(os.listdir(rundir)):
        if f in before or not f.lower().endswith(('.png', '.pdf')):
            continue
        data = open(os.path.join(rundir, f), 'rb').read()
        prev = seen.get(f)
        if prev is None:
            seen[f] = (cell, data)
        elif prev[1] != data:
            stem, ext = os.path.splitext(f)
            open(os.path.join(rundir, f'{stem}__cell{prev[0]}{ext}'), 'wb').write(prev[1])
            seen[f] = (cell, data)
        # unchanged: keep the cell index that actually wrote it


def main():
    rundir = os.path.abspath(RUNDIR)
    outdir = os.path.abspath(OUTDIR)
    os.makedirs(outdir, exist_ok=True)
    before = set(os.listdir(rundir))

    nb = json.load(open(NOTEBOOK, encoding='utf-8'))
    plt.show = lambda *args, **kw: None
    g = {'__name__': '__main__'}
    cwd = os.getcwd()
    os.chdir(rundir)
    seen = {}                         # filename -> (cell index, bytes) of the last version
    try:
        for i, c in enumerate(nb['cells']):
            if c['cell_type'] != 'code':
                continue
            exec(compile(''.join(c['source']), f'<cell {i}>', 'exec'), g)
            plt.close('all')          # 15 panels in one process; do not hold them all
            _keep_overwritten(rundir, before, seen, i)
    finally:
        os.chdir(cwd)

    made = sorted(f for f in set(os.listdir(rundir)) - before
                  if f.lower().endswith(('.png', '.pdf')))
    if outdir != rundir:
        for f in made:
            shutil.move(os.path.join(rundir, f), os.path.join(outdir, f))
    for f in made:
        p = os.path.join(outdir, f)
        print(f'  written: {p}   ({os.path.getsize(p) / 1e6:.1f} MB)')
    print(f'  {len(made)} panels')
    return 0 if made else 1


if __name__ == '__main__':
    sys.exit(main())
