#!/usr/bin/env python3
"""Fig. 4b - hybridization spin asymmetry against maximum spin-resolved p-d hybridization.

Ported from the cell of the working notebook that produced the published panel
(`fig_heatmap_filled`). Only three things are changed: the input path, the output
location, and two vestigial `% mask.sum()` applied to legend strings that carry no format
placeholder - as written those raise `TypeError: not all arguments converted during string
formatting`, so the cell cannot be executed as it stands. The counts they were meant to
print are reported on stdout instead.

For each structure the PROCAR-derived table carries one `pd_hybrid_minimum` per spin
channel. Pivoting on spin gives H_up and H_dn per structure, from which

    H_max   = max(H_up, H_dn)
    delta_H = |H_up - H_dn|

The background is the mean SSE over a 100 x 100 binning of that plane, extended to the
y = x boundary by nearest-neighbour padding and then interpolated with a thin-plate-spline
RBF; it is masked above y = x, which delta_H <= H_max forbids. Points are split by p/d
electron ratio - black circles below 1, red triangles at or above 1 - and VO and CrSb, the
controlled same-prototype comparison in the text, are marked and labelled.

    python figures/reference_plots/plot_fig4b.py

Input  : ../plotdata/fig4b_pd_hybridization.csv   (override with $FIG4B_DATA)
         one row per (structure, spin); deposited as the notebook's `pd_hybri_sse.csv`
Output : SI-style fig4b.pdf / .png next to this script (override with $FIG4B_OUTDIR)
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable

# the _cohort / _fonts / _save helpers live beside this file; make that true
# however the script is invoked, not just via `python path/to/this.py`
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _fonts
import _save

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('FIG4B_DATA',
                      os.path.join(HERE, '..', 'plotdata', 'fig4b_pd_hybridization.csv'))
OUTDIR = os.environ.get('FIG4B_OUTDIR', HERE)

# Text size. Everything in this figure is set relative to FS_SCALE, so one number
# changes them all. 1.0 - the default - is the size of the type in Fig. 4a, which sits
# directly above this panel in the composed figure.
#
# The two are placed with their axes frames the same width, so the type has to be the same
# size too, and this panel's used to be about a tenth smaller. Fig. 4a sets its axis labels,
# tick labels and colorbar label at 15 / 13 / 12 pt scaled by its own FS4_SCALE = 1.54, on a
# frame 281.4 pt across; this panel's frame is 277.9 pt, so the same type here is those base
# sizes times FS_MATCH. The published panel's own sizes were 20 / 18 / 18 / 16.
#
# The layout adapts: past CROWDED the legend sits in the empty half above the
# diagonal but its two lines run past the axes, so it is scaled more gently, its
# labels are shortened and the canvas grows.
FS_MATCH = 1.54 * 277.9 / 281.4
FS_SCALE = 1.0
CROWDED = FS_SCALE > 1.3
FS_AXIS, FS_TICK, FS_CBAR = (s * FS_MATCH * FS_SCALE for s in (15, 13, 12))
# the annotation boxes and the legend have no counterpart in Fig. 4a; they keep the
# proportion to the tick labels that the published panel gave them
FS_ANNOT = FS_TICK * 16 / 18
FS_LEGEND = FS_TICK * 17.2 / 18 * (1 + (FS_SCALE - 1) * 0.45)
FIGSIZE = (6 * (1.45 if CROWDED else 1.0), 5 * (1.45 if CROWDED else 1.0))
CBAR_LABELPAD = 18 * FS_SCALE if CROWDED else 30
LEGEND_LABELS = ((r'$p/d < 1$', r'$p/d \geq 1$') if CROWDED else
                 (r'$p/d$ electron ratio < 1', r'$p/d$ electron ratio$\geq 1$'))

NBINS = 100        # binning of the (H_max, delta_H) plane for the mean-SSE background
MIN_COUNT = 2      # a bin needs at least this many structures to enter the interpolation
NGRID = 300        # RBF output grid
SMOOTHING = 1.0    # RBFInterpolator smoothing

# The two annotated cases, as read off the data in the original notebook.
VO = (0.4115, 0.2865)      # p/d = 1.0  -> triangle
CRSB = (0.8076, 0.5140)    # p/d = 0.6  -> circle


def main():
    df = pd.read_csv(DATA)
    piv = df.pivot_table(index=['filename', 'sse', 'pd_ratio'], columns='spin',
                         values='pd_hybrid_minimum', aggfunc='mean').reset_index()
    piv = piv.dropna(subset=['up', 'down'])
    piv['H_max'] = piv[['up', 'down']].max(axis=1)
    piv['delta_H'] = np.abs(piv['up'] - piv['down'])

    x = piv['H_max'].values
    y = piv['delta_H'].values
    sse = piv['sse'].values
    low = piv['pd_ratio'].values < 1.0
    high = ~low
    print(f'structures with both spin channels: {len(piv)}   ({DATA})')
    print(f'  p/d < 1: {low.sum()}   p/d >= 1: {high.sum()}')

    # ---- mean SSE over a fine binning of the plane -------------------------
    edges = np.linspace(0, 1, NBINS + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    sum_z, _, _ = np.histogram2d(x, y, bins=[edges, edges], weights=sse)
    count, _, _ = np.histogram2d(x, y, bins=[edges, edges])
    with np.errstate(invalid='ignore'):
        mean_z = np.where(count >= MIN_COUNT, sum_z / count, np.nan)

    Xc, Yc = np.meshgrid(centres, centres, indexing='ij')
    valid = ~np.isnan(mean_z)
    pts = np.column_stack([Xc[valid], Yc[valid]])
    vals = mean_z[valid]
    print(f'  occupied bins: {valid.sum()} of {mean_z.size}')

    # ---- pad the y = x boundary and the far corner by nearest neighbour ----
    tree = cKDTree(pts)
    edge_x = np.linspace(0.05, 0.95, 20)
    edge_y = edge_x - 0.02
    _, idx = tree.query(np.column_stack([edge_x, edge_y]))
    corner = np.array([[0.95, 0.90], [0.90, 0.85], [0.85, 0.80],
                       [0.98, 0.95], [0.95, 0.93], [0.92, 0.90]])
    _, idx_c = tree.query(corner)
    pts_aug = np.vstack([pts, np.column_stack([edge_x, edge_y]), corner])
    vals_aug = np.concatenate([vals, vals[idx], vals[idx_c]])

    # ---- thin-plate-spline interpolation onto the display grid -------------
    rbf = RBFInterpolator(pts_aug, vals_aug, kernel='thin_plate_spline',
                          smoothing=SMOOTHING)
    xi = yi = np.linspace(0, 1, NGRID)
    Xi, Yi = np.meshgrid(xi, yi)
    Z = np.clip(rbf(np.column_stack([Xi.ravel(), Yi.ravel()])).reshape(NGRID, NGRID),
                0, None)
    Z[Yi > Xi] = np.nan            # delta_H <= H_max

    # ---- plot --------------------------------------------------------------
    _fonts.use()

    fig, ax = plt.subplots(figsize=FIGSIZE)
    im = ax.pcolormesh(xi, yi, Z, cmap='jet', vmin=0, vmax=1, shading='gouraud',
                       rasterized=True, zorder=1)

    # NOT rasterized. A single-colour rasterized layer is written into the PDF as a
    # 1-bit indexed image with a soft mask, and viewers render that combination
    # wrong - the points come out white and streaked, while the PNG is fine. 3,845
    # markers draw cheaply as vector and are correct everywhere.
    ax.scatter(x[low], y[low], s=10, c='k', alpha=0.3, edgecolors='black',
               marker='o', rasterized=False, zorder=2)
    ax.scatter(x[high], y[high], s=80, c='red', alpha=0.8, edgecolors='r',
               linewidths=0.3, marker='^', rasterized=False, zorder=3)

    for name, (px, py), marker, size in (('VO', VO, '^', 80),
                                         ('CrSb', CRSB, 'o', 100)):
        ax.scatter(px, py, s=size, marker=marker,
                   c='red' if marker == '^' else 'k',
                   edgecolors='white', linewidths=2, zorder=5)
        ax.annotate(name, xy=(px, py), xytext=(px - 0.35, py + 0.15),
                    fontsize=FS_ANNOT, fontweight='bold', color='k',
                    arrowprops=dict(arrowstyle='->', color='k', lw=1.5),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='k', alpha=0.85),
                    zorder=6)

    ax.plot([0, 1], [0, 1], 'w-', lw=1.5, alpha=0.6)
    ax.set_aspect('equal')

    # make_axes_locatable, not fig.colorbar(ax=ax): this axes is set_aspect('equal'),
    # so its visible box is shorter than the position rectangle matplotlib hands the
    # colorbar, and a plain colorbar comes out taller than the plot it belongs to. The
    # divider measures the aspect-adjusted box, so the bar matches it exactly.
    cax = make_axes_locatable(ax).append_axes('right', size='4.5%', pad=0.07)
    cbar = fig.colorbar(im, cax=cax)
    # rotation is left at matplotlib's default 90, so the colorbar label reads
    # bottom-to-top like the y-axis label opposite it - the same as 'MPF' on Fig. 4a
    # and 'Maximum SSE (eV)' on SI Fig. S2. The original notebook set rotation=270,
    # which reads top-to-bottom.
    cbar.set_label(r'$\langle$SSE$\rangle$ (eV)', fontsize=FS_CBAR, labelpad=6)
    cbar.ax.tick_params(labelsize=FS_TICK)

    ax.set_xlabel(r'$\mathrm{max}(H_{\uparrow},\, H_{\downarrow})$', fontsize=FS_AXIS)
    ax.set_ylabel(r'$\Delta H$', fontsize=FS_AXIS)
    ax.tick_params(labelsize=FS_TICK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # both axes every 0.2, as published. Left to itself matplotlib thins the abscissa to
    # 0.00 / 0.25 / 0.50 / 0.75 / 1.00 once the tick labels are set at Fig. 4a's size,
    # which leaves the two axes of the same square panel on different steps
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_yticks(np.arange(0, 1.01, 0.2))

    ax.legend(handles=[
        Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=8,
               ls='', label=LEGEND_LABELS[0]),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='red', markersize=10,
               markeredgecolor='white', markeredgewidth=0.5,
               ls='', label=LEGEND_LABELS[1]),
    ], loc='upper left', fontsize=FS_LEGEND, frameon=False)

    fig.tight_layout()
    _save.save(fig, OUTDIR, 'fig4b')


if __name__ == '__main__':
    main()
