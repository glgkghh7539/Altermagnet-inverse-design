#!/usr/bin/env python3
"""Regenerate every code-generated figure panel in the manuscript, in one command.

    python figures/make_all_figures.py                 # everything -> figures/generated/
    python figures/make_all_figures.py --outdir OUT    # somewhere else
    python figures/make_all_figures.py --list          # what would run
    python figures/make_all_figures.py --only fig2 si4 # a subset

Each generator runs in its own subprocess, so a 1000 dpi render releases its ~1.1 GB before
the next one starts and a failure in one panel does not stop the rest. The exit status is
non-zero if any generator failed, and the run always ends with a table saying what was
produced and what was not.

What this does NOT produce, because no generating code exists for it:

  * Fig. 1 in full - a schematic, drawn in a graphics program.
  * The crystal-structure sub-panels of Fig. 4c-d, Fig. 5 and SI Figs. S5-S6 - rendered in
    VESTA. The band and fatband panels of those same figures ARE produced here.
  * Final assembly. Panel lettering, arrangement and annotation arrows were done in
    presentation software; that step changes layout, never a plotted value.

See figures/README.md for the panel-by-panel map.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, 'reference_plots')
NOTEBOOK = os.path.join(HERE, 'redraw_figures_R1.ipynb')
PLOTFAT = os.path.join(HERE, 'archive', 'plotfat_original.ipynb')
FATBAND_DIR = os.path.join(HERE, 'fatband')

# key, description, kind, payload
#   'script'   -> (path, env overrides)
#   'cells'    -> (notebook, [cell indices], cwd, env overrides)
JOBS = [
    ('fig2',   'Fig. 2 - SHAP bars + beeswarm',
     'cells', (NOTEBOOK, [3], HERE)),
    ('fig3abc', 'Fig. 3a-c - SSE vs MSBI / MPF / p-d',
     'cells', (NOTEBOOK, [5], HERE)),
    ('fig3de', 'Fig. 3d-e - CuO and FeSi band structures',
     'script', (os.path.join(REF, 'plot_band.py'), 'BAND_OUTDIR')),
    ('fig4a',  'Fig. 4a - SSE vs d_CC, coloured by MPF',
     'cells', (NOTEBOOK, [7], HERE)),
    ('fig4b',  'Fig. 4b - hybridization spin asymmetry',
     'script', (os.path.join(REF, 'plot_fig4b.py'), 'FIG4B_OUTDIR')),
    # run through the adapter, not the archived notebook directly: it serves the two
    # colorbar labels at rotation 90 instead of 270, so they read the same way round as
    # every other colorbar here. Nothing else about the panels changes.
    ('fatbands', 'Fig. 4c-d, Fig. 5, SI S5-S6 - fatband panels (16 files)',
     'script', (os.path.join(REF, 'plot_fatbands.py'), 'FATBAND_OUTDIR')),
    ('si1',    'SI Fig. S1 - dataset statistics',
     'script', (os.path.join(REF, 'plot_si1_dataset_stats.py'), 'SI_OUTDIR')),
    ('si2',    'SI Fig. S2 - M-X maximum-SSE heatmap',
     'script', (os.path.join(REF, 'plot_si2_mx_heatmap.py'), 'SI_OUTDIR')),
    ('si3',    'SI Fig. S3 - parity plot + decile bias',
     'cells', (NOTEBOOK, [9], HERE)),
    ('si4',    'SI Fig. S4 - Bayesian-optimization convergence',
     'script', (os.path.join(REF, 'plot_si4_bo_progress.py'), 'SI_OUTDIR')),
    # cell 11 reuses the ranking table cell 3 builds, so both run; '-' as the output
    # directory stubs out save() so Fig. 2 is not rendered a second time.
    ('table1', 'Table 1 - the LaTeX rows that go with Fig. 2a',
     'cells', (NOTEBOOK, [3, 11], HERE, '-')),
]


def run_cells(nb_path, cells, cwd, outdir):
    """Execute a notebook's code cells in this process (called via --exec-cells)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.show = lambda *a, **k: None

    nb = json.load(open(nb_path, encoding='utf-8'))
    code = [i for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code']
    wanted = code if cells is None else cells
    g = {'__name__': '__main__'}
    os.chdir(cwd)

    setup = code[0] if code else None
    if nb_path.endswith('redraw_figures_R1.ipynb'):
        exec(compile(''.join(nb['cells'][setup]['source']), '<setup>', 'exec'), g)
        if outdir == '-':
            g['save'] = lambda *a, **k: None      # numbers only, no files
        elif outdir:
            g['OUTDIR'] = outdir
            os.makedirs(outdir, exist_ok=True)
        wanted = [i for i in wanted if i != setup]
    for i in wanted:
        exec(compile(''.join(nb['cells'][i]['source']), f'<cell {i}>', 'exec'), g)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--outdir', default=os.path.join(HERE, 'generated'))
    ap.add_argument('--only', nargs='*', metavar='KEY')
    ap.add_argument('--list', action='store_true')
    # internal: re-entry point for the notebook subprocesses
    ap.add_argument('--exec-cells', nargs=4, help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a.exec_cells:
        nb_path, cells, cwd, outdir = a.exec_cells
        run_cells(nb_path, None if cells == '-' else [int(c) for c in cells.split(',')],
                  cwd, outdir or None)
        return 0

    if a.list:
        for key, desc, _, _ in JOBS:
            print(f'  {key:<9} {desc}')
        return 0

    jobs = [j for j in JOBS if not a.only or j[0] in a.only]
    unknown = set(a.only or []) - {j[0] for j in JOBS}
    if unknown:
        sys.exit(f'unknown key(s): {", ".join(sorted(unknown))}')

    outdir = os.path.abspath(a.outdir)
    os.makedirs(outdir, exist_ok=True)
    print(f'output directory: {outdir}\n')

    results = []
    for key, desc, kind, payload in jobs:
        print(f'--- {key}: {desc}')
        t0 = time.time()
        before = set(os.listdir(outdir))
        env = dict(os.environ)

        if kind == 'script':
            path, var = payload
            env[var] = outdir
            cmd = [sys.executable, path]
            cwd = HERE
        else:
            nb_path, cells, cwd = payload[:3]
            sub = payload[3] if len(payload) > 3 else (outdir if nb_path != PLOTFAT else '')
            cmd = [sys.executable, os.path.abspath(__file__), '--exec-cells', nb_path,
                   '-' if cells is None else ','.join(map(str, cells)), cwd, sub]

        fat_before = set(os.listdir(FATBAND_DIR)) if nb_path_is_fat(kind, payload) else None
        p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
        dt = time.time() - t0

        if p.returncode != 0:
            if p.returncode < 0:
                # killed by a signal; -9 is the OOM killer, which is what a 1000 dpi
                # render hits on a machine with less than ~2 GB free
                why = f'killed by signal {-p.returncode}'
                if p.returncode == -9:
                    why += ' (SIGKILL - almost always out of memory)'
            else:
                why = f'exit status {p.returncode}'
            print(f'    FAILED - {why}')
            lines = [ln for ln in (p.stderr or '').strip().splitlines()
                     if 'timestamp seems very low' not in ln and 'findfont' not in ln]
            for ln in lines[-4:]:
                print('     ', ln[:110])
            results.append((key, 'FAILED', [], dt))
            continue

        # plotfat writes beside its data; collect those files into outdir
        if fat_before is not None:
            for f in sorted(set(os.listdir(FATBAND_DIR)) - fat_before):
                if f.lower().endswith(('.png', '.pdf')):
                    shutil.move(os.path.join(FATBAND_DIR, f), os.path.join(outdir, f))

        made = sorted(set(os.listdir(outdir)) - before)
        for f in made:
            print(f'    {f}')
        if key == 'table1':
            rows = [ln for ln in p.stdout.splitlines() if ln.rstrip().endswith(r'\\')]
            print(f'    {len(rows)} LaTeX table rows on stdout')
        results.append((key, 'ok', made, dt))

    print('\n' + '=' * 72)
    nfail = sum(1 for _, st, _, _ in results if st != 'ok')
    for key, st, made, dt in results:
        mark = 'ok  ' if st == 'ok' else 'FAIL'
        print(f'  {mark}  {key:<9} {len(made):>2} file(s)   {dt:5.1f} s')
    total = sum(len(m) for _, _, m, _ in results)
    print(f'\n  {total} files in {outdir}')
    print(f'  {len(results) - nfail} of {len(results)} generators succeeded')
    if nfail:
        print('\n  A generator killed by SIGKILL ran out of memory. Re-run just that one\n'
              '  with --only, or free memory first; Fig. 2 peaks near 1.1 GB and Fig. 3\n'
              '  near 1.3 GB, and bbox_inches=\'tight\' renders each canvas twice.')
    print('\n  not produced here (no generating code exists): Fig. 1, and the VESTA')
    print('  crystal-structure sub-panels of Fig. 4c-d, Fig. 5 and SI S5-S6.')
    return 1 if nfail else 0


def nb_path_is_fat(kind, payload):
    return kind == 'cells' and payload[0] == PLOTFAT


if __name__ == '__main__':
    sys.exit(main())
