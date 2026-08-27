#!/usr/bin/env python3
"""SI Fig. S4 - convergence of the Bayesian optimization.

Ported verbatim from the cell of the working notebook that produced the published panel
(`FIG_OPT_PROGRESS.png`); only the input path and the output location are changed.

Grey dots are the surrogate-predicted SSE of each of the 10^5 TPE trials, the blue curve is
the running best, and a red marker sits on every trial that improved on it.

    python figures/reference_plots/plot_si4_bo_progress.py

Input  : ../plotdata/si4_bo_progress.csv   (override with $SI4_DATA)
         the three columns the figure uses - trial_number, value, best_sse_overall -
         taken from the optimizer's `checkpoints/progress_log.csv`
Output : SI_4.pdf / SI_4.png next to this script   (override with $SI_OUTDIR)
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# the _cohort / _fonts / _save helpers live beside this file; make that true
# however the script is invoked, not just via `python path/to/this.py`
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _fonts
import _save

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('SI4_DATA',
                      os.path.join(HERE, '..', 'plotdata', 'si4_bo_progress.csv'))
OUTDIR = os.environ.get('SI_OUTDIR', HERE)

_fonts.use()

# Text size. Everything in this figure is set relative to FS_SCALE, so one number
# changes them all. 1.0 - the default - is the size the published panel used.
# The layout below adapts to it: above CROWDED the crowded labels are set
# vertically and the canvas grows, because at twice the size they collide.
FS_SCALE = 1.0
CROWDED = FS_SCALE > 1.3
FS_LABEL, FS_TICK = (s * FS_SCALE for s in (20, 15))
# the legend is three entries wide; enlarged it runs off the axes, so past CROWDED
# it is scaled more gently and moved to a band of its own above the data
FS_LEGEND = 12 * (1 + (FS_SCALE - 1) * 0.6)
FIGSIZE = (6 * (1.45 if CROWDED else 1.0), 4 * (1.45 if CROWDED else 1.0))
YMAX = 1.2 + (0.25 * (FS_SCALE - 1) if CROWDED else 0)
LEGEND_KW = (dict(loc='upper center', ncol=3, frameon=False,
                  handletextpad=0.4, columnspacing=1.4) if CROWDED
             else dict(loc='lower center', ncol=3))


def comma_fmt(x, pos):
    return f'{int(x):,}'


def main():
    df = pd.read_csv(DATA).sort_values('trial_number').reset_index(drop=True)
    x = df['trial_number'].values
    y = df['value'].values
    best = df['best_sse_overall'].values

    # a trial counts as a new best when the running maximum moves
    improved = np.zeros(len(best), dtype=bool)
    improved[0] = True
    improved[1:] = best[1:] > best[:-1]
    print(f'trials: {len(df):,}   new bests: {improved.sum()}   '
          f'final best: {best[-1]:.4f} eV   ({DATA})')

    fig, ax = plt.subplots(figsize=FIGSIZE)
    # The trial cloud is 100,000 markers. Drawn as vector they become 100,000 drawing
    # operations in the PDF, which several viewers take tens of seconds to render or
    # give up on entirely.
    #
    # Rasterizing the cloud ALONE is not the fix either: a single-colour rasterized
    # layer is written into the PDF as a 1-bit indexed image with a soft mask carrying
    # the alpha, and viewers render that with pale diagonal seams across the cloud.
    # set_rasterization_zorder puts the cloud and the blue best-so-far line into ONE
    # rasterized layer instead, which has more than two colours in it and is written as
    # an ordinary 8-bit image. The red new-best markers sit above the threshold and stay
    # vector, as do the axes, the frame and every piece of text.
    ax.set_rasterization_zorder(2.5)
    ax.scatter(x, y, c='black', s=0.8, alpha=0.15, zorder=1, label='Trial SSE')
    ax.plot(x, best, linewidth=2.0, c='#0000FF', zorder=2, label='Best SSE')
    ax.scatter(x[improved], best[improved], c='#FF0000', s=40, zorder=3,
               edgecolors='white', linewidths=0.8, label='New best', marker='o')

    ax.xaxis.set_major_formatter(FuncFormatter(comma_fmt))
    ax.set_xlabel('Trial number', fontsize=FS_LABEL)
    ax.set_ylabel('Predicted SSE (eV)', fontsize=FS_LABEL)
    ax.legend(fontsize=FS_LEGEND, **LEGEND_KW)
    ax.tick_params(axis='both', labelsize=FS_TICK)
    ax.set_xlim(0, 100000)
    ax.set_ylim(0, YMAX)

    fig.tight_layout()
    _save.save(fig, OUTDIR, 'SI_4')


if __name__ == '__main__':
    main()
