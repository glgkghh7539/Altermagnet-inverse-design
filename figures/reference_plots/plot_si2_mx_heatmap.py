#!/usr/bin/env python3
"""SI Fig. S2 - maximum DFT-calculated SSE (eV) for each M-X elemental combination.

The published S2 has no generating script in the deposit, so this one is written against
`data/fin_data.csv`. Both axes run in ascending atomic number, as the published ones do:
magnetic species Sc...Cd across, non-magnetic species B...Bi up. A cell holds the maximum
SSE over every structure with that pair; pairs that never occur are left blank.

The values differ from the published figure only where one of the six rows removed after
that figure was made (five `Cr2F8_cluster3`, one `Cu2O2_1_st05`) held the maximum for its
pair - see the "Known issues" section of the top-level README.

    python figures/reference_plots/plot_si2_mx_heatmap.py

Input  : ../../data/fin_data.csv          (override with $FIN_DATA)
Output : SI_2.pdf / SI_2.png next to this script   (override with $SI_OUTDIR)
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# the _cohort / _fonts / _save helpers live beside this file; make that true
# however the script is invoked, not just via `python path/to/this.py`
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _cohort
import _fonts
import _save

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.environ.get('SI_OUTDIR', HERE)

CMAP = 'Greys'
TEXT_SWITCH = 0.55        # fraction of vmax above which the cell label turns white

_fonts.use()

# Text size. Everything in this figure is set relative to FS_SCALE, so one number
# changes them all. 1.0 - the default - is the size the published panel used.
# 414 annotated cells cannot hold more text at a fixed canvas size, so the canvas
# scales with FS_SCALE; the printed glyphs then grow with it.
FS_SCALE = 1.0
FS_CELL, FS_TICK, FS_AXIS, FS_CBAR = (s * FS_SCALE for s in (6.2, 9, 13, 12))
# canvas chosen so the cropped output matches the published page, 362 x 414 pt
FIGSIZE = (5.1 * FS_SCALE, 5.85 * FS_SCALE)


def main():
    df = _cohort.load()
    cols = _cohort.by_atomic_number(df.M)       # magnetic, x axis
    rows = _cohort.by_atomic_number(df.X)       # non-magnetic, y axis
    print(f'rows: {len(df)}   grid: {len(cols)} magnetic x {len(rows)} non-magnetic')

    grid = np.full((len(rows), len(cols)), np.nan)
    for (m, x), g in df.groupby(['M', 'X']):
        grid[rows.index(x), cols.index(m)] = g.sse.max()
    filled = np.isfinite(grid)
    vmax = np.nanmax(grid)
    print(f'  occupied cells {filled.sum()} of {grid.size}   max SSE {vmax:.3f} eV')

    fig, ax = plt.subplots(figsize=FIGSIZE)
    # pcolormesh, not imshow: imshow resamples the 23 x 18 array up to the full canvas,
    # which on this size of figure costs about 1 GB and bakes the cells into a bitmap.
    # pcolormesh draws 414 quads as vector, uses a fraction of the memory, and keeps the
    # cell edges crisp at any zoom in the PDF.
    im = ax.pcolormesh(np.arange(len(cols) + 1) - 0.5, np.arange(len(rows) + 1) - 0.5,
                       np.ma.masked_invalid(grid), cmap=CMAP, vmin=0.0, vmax=vmax,
                       shading='flat')
    ax.set_xlim(-0.5, len(cols) - 0.5)
    ax.set_ylim(-0.5, len(rows) - 0.5)

    # outline every occupied cell, so a blank pair reads as "never occurs" rather than 0
    for i, j in zip(*np.nonzero(filled)):
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor='0.55', linewidth=0.4, zorder=2))
        v = grid[i, j]
        ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=FS_CELL, zorder=3,
                color='white' if v > TEXT_SWITCH * vmax else 'black')

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=FS_TICK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=FS_TICK)
    ax.set_xlabel('Magnetic atom', fontsize=FS_AXIS)
    ax.set_ylabel('Non-magnetic atom', fontsize=FS_AXIS)
    ax.tick_params(length=0)          # the published panel keeps its frame

    # make_axes_locatable, not fig.colorbar(ax=ax, aspect=...): `aspect` fixes the bar's
    # length-to-width ratio, so the bar comes out whatever height that implies and ends up
    # inset from the heatmap at both ends - about 7.5 pt at each end here, which is what it
    # looked like. The divider ties the bar to the heatmap's own box instead, and it spans
    # exactly the frame, as the published panel does. Its width and its gap are set as
    # fractions of the heatmap width, measured off that panel: 18.5 and 15.1 pt against a
    # 244.1 pt frame.
    cax = make_axes_locatable(ax).append_axes('right', size='7.58%', pad='6.19%')
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('Maximum SSE (eV)', fontsize=FS_CBAR)
    cb.ax.tick_params(labelsize=FS_TICK)

    fig.tight_layout()
    _save.save(fig, OUTDIR, 'SI_2')


if __name__ == '__main__':
    main()
