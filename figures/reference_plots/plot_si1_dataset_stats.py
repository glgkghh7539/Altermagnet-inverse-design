#!/usr/bin/env python3
"""SI Fig. S1 - statistical distribution of the screened cohort.

    (a) histogram of SSE
    (b) count by magnetic atom
    (c) count by chemical composition

The published S1 has no generating script in the deposit, so this one is written against
`data/fin_data.csv` to match it panel for panel. It reproduces the published bars exactly
apart from the six rows removed after that figure was made (five `Cr2F8_cluster3`
duplicates and one `Cu2O2_1_st05`), which is why the counts here are 3,845 rather than
3,851:

    (a) first bin   1665 -> 1660,  fifth bin 135 -> 134
    (b) Cr           625 ->  620,  Cu        116 -> 115
    (c) MX          1518 -> 1517,  MX4       445 -> 440

Every other bar is unchanged. See the "Known issues" section of the top-level README for
why those rows went.

    python figures/reference_plots/plot_si1_dataset_stats.py

Input  : ../../data/fin_data.csv          (override with $FIN_DATA)
Output : SI_1.pdf / SI_1.png next to this script   (override with $SI_OUTDIR)
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# the _cohort / _fonts / _save helpers live beside this file; make that true
# however the script is invoked, not just via `python path/to/this.py`
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _cohort
import _fonts
import _save

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.environ.get('SI_OUTDIR', HERE)

BAR_BLUE = '#2077b4'          # the blue of the published panels
SSE_BINS = np.arange(0.0, 1.751, 0.25)
NCOMP = 10                    # composition classes shown, most populated first

_fonts.use()

# Text size. Everything in this figure is set relative to FS_SCALE, so one number
# changes them all. 1.0 - the default - is the size the published panel used.
# The layout below adapts to it: above CROWDED the crowded labels are set
# vertically and the canvas grows, because at twice the size they collide.
FS_SCALE = 1.0
CROWDED = FS_SCALE > 1.3
FS_TICK, FS_LABEL, FS_COUNT, FS_PANEL = (s * FS_SCALE for s in (11, 13, 9, 15))
ROT = 90 if CROWDED else 0                       # x tick labels in (b) and (c)
# canvas chosen so the cropped output matches the published page, 289 x 542 pt
# 6.0 in wide, not the published 4.0: at the published width the 18 element labels of
# panel (b) and the 10 subscripted composition labels of panel (c) run into each other
# at 11 pt. The text is left at the published size and the canvas widened instead.
_grow = 1.5 if CROWDED else 1.0
FIGSIZE = (6.0 * _grow, 9.3 * (1.4 if CROWDED else 1.0))
YSTEP_A_C = 400 if CROWDED else 200              # y ticks thinned only when needed
HSPACE = 0.55 if CROWDED else 0.42


def panel_letter(ax, s):
    ax.text(-0.13, 1.06, s, transform=ax.transAxes, ha='left', va='bottom',
            fontsize=FS_PANEL, fontweight='bold')


def annotate(ax, xs, heights, pad_frac=0.02, rotation=0):
    """Print the count above each bar, as the published panels do.

    At FS_SCALE = 2 the counts no longer fit side by side over 18 adjacent bars, so the
    crowded panels print them vertically instead of dropping them.
    """
    top = max(heights) if len(heights) else 1
    for x, h in zip(xs, heights):
        if h == 0:
            continue
        ax.text(x, h + pad_frac * top, f'{int(h)}', ha='center', va='bottom',
                fontsize=FS_COUNT, rotation=rotation)


def main():
    df = _cohort.load()
    print(f'rows: {len(df)}')

    fig, axes = plt.subplots(3, 1, figsize=FIGSIZE)

    # ---- (a) SSE histogram -------------------------------------------------
    ax = axes[0]
    counts, edges = np.histogram(df.sse.to_numpy(float), bins=SSE_BINS)
    centres = 0.5 * (edges[:-1] + edges[1:])
    ax.bar(centres, counts, width=np.diff(edges), color=BAR_BLUE,
           edgecolor='black', linewidth=0.8)
    annotate(ax, centres, counts)
    ax.set_xlabel('SSE (eV)', fontsize=FS_LABEL)
    ax.set_ylabel('Count of Materials', fontsize=FS_LABEL)
    ax.set_xlim(edges[0], edges[-1])
    ax.set_xticks(edges)
    # the count sits above the bar, so the top of the axes has to clear it: at
    # ylim = 1800 the 1660 label runs into the frame
    ax.set_ylim(0, 1950)
    ax.set_yticks(np.arange(0, 1801, YSTEP_A_C))
    panel_letter(ax, '(a)')
    print('  (a)', dict(zip([f'{e:.2f}' for e in edges[:-1]], counts.tolist())))

    # ---- (b) magnetic atom -------------------------------------------------
    ax = axes[1]
    order = _cohort.by_atomic_number(df.M)
    heights = [int((df.M == m).sum()) for m in order]
    xs = np.arange(len(order))
    ax.bar(xs, heights, width=0.75, color=BAR_BLUE, edgecolor='black', linewidth=0.8)
    annotate(ax, xs, heights, pad_frac=0.03 if CROWDED else 0.02, rotation=ROT)
    ax.set_xticks(xs)
    ax.set_xticklabels(order, fontsize=FS_TICK, rotation=ROT)
    ax.set_xlabel('Magnetic Atom', fontsize=FS_LABEL)
    ax.set_ylabel('Count of Materials', fontsize=FS_LABEL)
    ax.set_xlim(-0.7, len(order) - 0.3)
    ax.set_ylim(0, 1000 if CROWDED else 760)     # headroom for the count labels
    ax.set_yticks(np.arange(0, 801, 200 if CROWDED else 100))
    panel_letter(ax, '(b)')
    print('  (b)', dict(zip(order, heights)))

    # ---- (c) composition ---------------------------------------------------
    ax = axes[2]
    vc = df.composition.value_counts().head(NCOMP)
    labels = []
    for lab in vc.index:                                # MX2 -> $\mathrm{MX_2}$
        s, i = '', 0
        while i < len(lab):
            s += lab[i]
            j = i + 1
            while j < len(lab) and lab[j].isdigit():
                j += 1
            if j > i + 1:
                s += '_{%s}' % lab[i + 1:j]
            i = j
        labels.append(r'$\mathrm{%s}$' % s)
    xs = np.arange(len(vc))
    ax.bar(xs, vc.values, width=0.65, color=BAR_BLUE, edgecolor='black', linewidth=0.8)
    annotate(ax, xs, vc.values, pad_frac=0.03 if CROWDED else 0.02)
    ax.set_xticks(xs)
    # the composition labels carry subscripts and are the widest set in the figure,
    # so this panel alone takes them down a notch - as the published one does
    ax.set_xticklabels(labels, fontsize=FS_TICK * 0.9, rotation=ROT)
    ax.set_xlabel('Composition', fontsize=FS_LABEL)
    ax.set_ylabel('Count of Materials', fontsize=FS_LABEL)
    ax.set_xlim(-0.7, len(vc) - 0.3)
    ax.set_ylim(0, 1750)
    ax.set_yticks(np.arange(0, 1601, YSTEP_A_C))
    panel_letter(ax, '(c)')
    print('  (c)', dict(zip(vc.index, vc.values.tolist())))

    for ax in axes:
        ax.tick_params(labelsize=FS_TICK)

    fig.subplots_adjust(hspace=HSPACE)
    _save.save(fig, OUTDIR, 'SI_1')


if __name__ == '__main__':
    main()
