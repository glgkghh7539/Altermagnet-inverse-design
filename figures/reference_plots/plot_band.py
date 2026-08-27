#!/usr/bin/env python3
"""Fig. 3d/3e - CuO and FeSi band structures from VASPKIT `BAND.dat`.

This is the maintained version of `archive/plotband_CuO_original.ipynb` and
`archive/plotband_FeSi_original.ipynb`. The plotting code is theirs, unchanged; the two
notebooks differ only in the per-material constants collected in MATERIALS below, so
they are folded into one script here.

The CuO notebook cannot be executed as deposited: it calls `plt.rcParams` before
`import matplotlib.pyplot as plt`, which only worked because the kernel it was run in
already had pyplot imported. The archive is kept byte-for-byte as it was run (see
`archive/ORIGINAL_MD5.txt`), so the ordering is corrected here instead.

Spin-up is drawn red and spin-down blue wherever the local splitting exceeds THRESHOLD,
and both black where it does not - the black stretches are the spin-degenerate ones.

    python figures/reference_plots/plot_band.py                   # all four
    python figures/reference_plots/plot_band.py CuO              # the published panel
    python figures/reference_plots/plot_band.py CuO_full         # the whole k-path

Inputs  : ../data/<material>/BAND.dat, ../data/<material>/BAND_full.dat
Outputs : <material>_bandstructure.png / .pdf, written next to this script
          (override with $BAND_OUTDIR)
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

import _fonts
import _save

_fonts.use()

# Text size. Everything in this figure is set relative to FS_SCALE, so one number
# changes them all. 1.0 - the default - is the size the published panel used.
FS_SCALE = 1.0
_grow = 1.25 if FS_SCALE > 1.3 else 1.0   # more canvas only when the text grows


HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('BAND_DATA', os.path.join(HERE, '..', 'data'))
OUTDIR = os.environ.get('BAND_OUTDIR', HERE)

# Per-material constants. 'CuO' and 'FeSi' are taken verbatim from the two notebooks and
# reproduce the published Fig. 3d/3e, which zoom on the two segments around Gamma.
#
# 'CuO_full' and 'FeSi_full' draw the whole k-path instead - Gamma-X|Y-Gamma-Z|R-Gamma-
# T|U-Gamma-V, 280 k-points against 80 - from `BAND_full.dat`. That is the path the
# calculation actually ran and the one the group's own gnuplot script labels; the
# published panels use only its first two segments, which is why they carry two blank
# tick labels either side of Gamma. The y range and line width follow that gnuplot
# script (`set yr [-4:4]`, `lw 3`).
MATERIALS = {
    # The archived CuO notebook draws a wide -2..2 eV panel with 10 pt ribbons and no
    # tick labels. That is NOT what Fig. 3d shows: measured off the published PDF, panels
    # d and e are the same size (175.7 x 316.1 pt, portrait) and the CuO panel is drawn
    # exactly like the FeSi one - T-Gamma-U, -4..4 eV, 1.5 pt lines, black where the two
    # spins are degenerate. Those settings are used here; the archived variant is kept as
    # 'CuO_wide' below.
    'CuO': dict(file='BAND.dat', dirname='CuO',
                hs_positions=[0.000, 1.314, 2.613],
                hs_labels=['T', 'Γ', 'U'],
                threshold=0.03,          # eV; below this the pair is drawn black
                ylim=(-4, 4),
                lw=1.5,
                figsize=(4 * _grow, 7 * _grow),   # taller than the published
                fontsize=20,                      # panel, to fill the column
                shade_below_ef=False,
                vlines='middle',
                stem='CuO'),
    'CuO_wide': dict(file='BAND.dat', dirname='CuO',
                     hs_positions=[0.000, 1.314, 2.613],
                     hs_labels=['T', 'Γ', 'U'],
                     threshold=0.00,
                     ylim=(-2, 2),
                     lw=10,
                     figsize=(6 * _grow, 4 * _grow),
                     fontsize=30,
                     shade_below_ef=True,
                     vlines='middle',
                     stem='CuO_wide'),
    # ylabel=False: in the published Fig. 3 the two band panels sit side by side and
    # only the left one is labelled. Dropping it here also keeps this panel's left
    # margin narrow enough that, placed at the published position, its page does not
    # overlap the neighbouring panel and clip that panel's right spine.
    'FeSi': dict(file='BAND.dat', dirname='FeSi', ylabel=False,
                 hs_positions=[0.000, 1.131, 2.303],
                 hs_labels=['T', 'Γ', 'U'],
                 threshold=0.03,
                 ylim=(-4, 4),
                 lw=1.5,
                 # 3.51 in, not 4: without the ylabel the freed margin would go to the
                 # axes and make this panel wider than the CuO one, so the two would be
                 # scaled differently when placed side by side and their type would not
                 # match. Narrowing the canvas by the width of the label keeps the axes
                 # at CuO's 211 pt.
                 figsize=(3.51 * _grow, 7 * _grow),
                 fontsize=20,
                 shade_below_ef=False,
                 vlines='middle',
                 stem='FeSi'),
    'CuO_full': dict(file='BAND_full.dat', dirname='CuO',
                     hs_positions=[0.000, 1.143, 2.286, 2.924, 4.721, 6.035, 7.334, 9.018],
                     hs_labels=['Γ', 'X|Y', 'Γ', 'Z|R', 'Γ', 'T|U', 'Γ', 'V'],
                     threshold=0.03,
                     ylim=(-4, 4),
                     lw=1.5,
                     figsize=(9 * _grow, 5 * _grow),
                     fontsize=18,
                     shade_below_ef=False,
                     vlines='all',
                     stem='CuO_full'),
    'FeSi_full': dict(file='BAND_full.dat', dirname='FeSi',
                      hs_positions=[0.000, 0.998, 1.948, 2.562, 4.071, 5.202, 6.373, 7.752],
                      hs_labels=['Γ', 'X|Y', 'Γ', 'Z|R', 'Γ', 'T|U', 'Γ', 'V'],
                      threshold=0.03,
                      ylim=(-4, 4),
                      lw=1.5,
                      figsize=(9 * _grow, 5 * _grow),
                      fontsize=18,
                      shade_below_ef=False,
                      vlines='all',
                      stem='FeSi_full'),
}


def read_band_dat(path):
    """Parse VASPKIT BAND.dat: blank lines and '#' headers separate bands.

    Columns are k-path, spin-up (eV), spin-down (eV).
    """
    bands, current = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or line == '':
                if current:
                    bands.append(np.array(current))
                    current = []
            else:
                current.append(list(map(float, line.split())))
        if current:
            bands.append(np.array(current))
    return bands


def plot(name, cfg):
    src = os.path.join(DATA, cfg['dirname'], cfg['file'])
    bands = read_band_dat(src)
    print(f'{name}: bands {len(bands)}, k-points {len(bands[0])}  ({src})')

    # VASPKIT repeats the k value at a high-symmetry boundary; joining across that
    # repeat would draw a spurious vertical line, so those segments are skipped.
    kk0 = bands[0][:, 0]
    skip = {i for i in range(len(kk0) - 1) if abs(kk0[i] - kk0[i + 1]) < 1e-5}

    hs_positions, lw = cfg['hs_positions'], cfg['lw']
    ymin, ymax = cfg['ylim']
    fs = cfg['fontsize'] * FS_SCALE

    fig, ax = plt.subplots(figsize=cfg['figsize'])
    for band in bands:
        kk, eup, edn = band[:, 0], band[:, 1], band[:, 2]
        for i in range(len(kk) - 1):
            if i in skip:
                continue
            # colour from the mean splitting of the two k-points spanned
            diff_avg = 0.5 * (abs(eup[i] - edn[i]) + abs(eup[i + 1] - edn[i + 1]))
            if diff_avg < cfg['threshold']:
                col_up = col_dn = 'black'
            else:
                col_up, col_dn = '#FF0000', '#0000FF'
            seg_x = [kk[i], kk[i + 1]]
            ax.plot(seg_x, [eup[i], eup[i + 1]], color=col_up, lw=lw, solid_capstyle='round')
            ax.plot(seg_x, [edn[i], edn[i + 1]], color=col_dn, lw=lw, solid_capstyle='round')

    ax.set_ylim(ymin, ymax)
    ax.set_xlim(hs_positions[0], hs_positions[-1])
    ax.margins(x=0)
    interior = hs_positions[1:-1] if cfg['vlines'] == 'all' else [hs_positions[1]]
    for xp in interior:
        ax.axvline(x=xp, color='black', lw=1.0, linestyle='--', dashes=(4, 3), zorder=1)
    ax.axhline(y=0, color='black', ls='--', lw=1.2, zorder=1)

    ax.set_xticks(hs_positions)
    ax.set_xticklabels(cfg['hs_labels'], fontsize=fs)   # unicode Γ, upright
    ax.tick_params(axis='x', which='both', length=0)

    yticks = list(range(ymin, ymax + 1))
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(v) for v in yticks], fontsize=fs)
    ax.tick_params(axis='y', length=4)
    if cfg.get('ylabel', True):
        ax.set_ylabel(r'$\mathrm{E - E_F\ (eV)}$', fontsize=fs)

    if cfg['shade_below_ef']:
        ax.axhspan(ymin, 0, color='gray', alpha=0.5, zorder=0)

    plt.tight_layout()
    _save.save(fig, OUTDIR, f"{cfg['stem']}_bandstructure")


if __name__ == '__main__':
    wanted = sys.argv[1:] or list(MATERIALS)
    for name in wanted:
        if name not in MATERIALS:
            sys.exit(f'unknown material {name!r}; choose from {list(MATERIALS)}')
        plot(name, MATERIALS[name])
