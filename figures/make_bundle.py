#!/usr/bin/env python3
"""Collect every manuscript figure as a PDF, with the code that makes it, into one folder.

The result is meant to be handed over on its own - to a co-author, or with a submission -
and it is not just a pile of files: the layout is the one the code expects, so the bundle
regenerates its own figures.

    python figures/make_bundle.py                    # -> ../Figures_R1_bundle
    python figures/make_bundle.py --out /some/where
    python figures/make_bundle.py --verify           # rebuild the composites inside it

What lands in it:

    pdf/                    the ten manuscript figures, one PDF each, named as the
                            manuscript names them. Fig. 1 is not here - it is a schematic
                            with no generating code.
    deposit/figures/panels/ the individual panels the composites are built from
    reviewer_verification/  the published PDFs, if they are reachable. They are not part
                            of the repository; the H_sigma labels take their Cambria Math
                            out of figure5.pdf when it is there, and fall back to Times
                            New Roman Italic when it is not.
    deposit/figures/        the code and its inputs

Three files are deliberately NOT copied: reference_plots/{figure2,figure3abc,SI_3}.{pdf,png}.
They are older outputs of the three reference scripts that need `shap_recompute.py` to have
been run first, and they were written before those scripts were put on the shared font
setup, so they are in DejaVu Sans rather than Times New Roman. The maintained versions of
the same three panels - redrawn/figure2, redrawn/figure3abc, redrawn/SI_3 - are copied and
are the ones the manuscript uses.
"""
import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..'))          # the deposit
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))    # its parent, in this working tree
DEFAULT_OUT = os.path.join(ROOT, 'Figures_R1_bundle')

sys.path.insert(0, HERE)
import compose_figures        # noqa: E402  - for the panel and published-figure paths

PANELS = compose_figures.PANELS
PUBLISHED = compose_figures.PUBLISHED
COMPOSED = os.path.join(HERE, 'composed')

# manuscript name -> where it comes from: composed/ for the composites, PANELS for the rest
FIGURES = [
    ('Fig2.pdf',   'Fig2(a-b).pdf'),
    ('Fig3.pdf',   'composed/Fig3.pdf'),
    ('Fig4.pdf',   'composed/Fig4.pdf'),
    ('Fig5.pdf',   'composed/Fig5.pdf'),
    ('SI_S1.pdf',  'SI1(a-c).pdf'),
    ('SI_S2.pdf',  'SI2.pdf'),
    ('SI_S3.pdf',  'SI3(a-b).pdf'),
    ('SI_S4.pdf',  'SI4.pdf'),
    ('SI_S5.pdf',  'composed/SI_5.pdf'),
    ('SI_S6.pdf',  'composed/SI_6.pdf'),
]

# the published PDFs the code still reads
PUBLISHED_FILES = ('figure3.pdf', 'figure4.pdf', 'figure5.pdf', 'SI_5.pdf', 'SI_6.pdf')

# superseded, and in the wrong font; see the module docstring
SKIP = {os.path.join('reference_plots', f'{s}.{e}')
        for s in ('figure2', 'figure3abc', 'SI_3') for e in ('pdf', 'png')}

IGNORE = shutil.ignore_patterns('__pycache__', '_tmp', 'composed',
                                '.ipynb_checkpoints')


README = """# Figures for the revised manuscript

Every figure in the paper as a PDF, and the code that produced it, in one place.

## `pdf/` - the figures

| File | Manuscript | Made of |
|---|---|---|
| `Fig2.pdf` | Fig. 2 | one panel: mean\\|SHAP\\| bars + beeswarm |
| `Fig3.pdf` | Fig. 3 | a-c plotted, d-e band structures under CuO and FeSi structures |
| `Fig4.pdf` | Fig. 4 | a-b plotted, c-d one fatband image under VO and CrSb structures |
| `Fig5.pdf` | Fig. 5 | FeS and NiS structures over one fatband image |
| `SI_S1.pdf` | SI Fig. S1 | dataset statistics |
| `SI_S2.pdf` | SI Fig. S2 | M-X maximum-SSE heatmap |
| `SI_S3.pdf` | SI Fig. S3 | parity plot + decile bias |
| `SI_S4.pdf` | SI Fig. S4 | Bayesian-optimization convergence |
| `SI_S5.pdf` | SI Fig. S5 | CoO and CrS |
| `SI_S6.pdf` | SI Fig. S6 | CoS and FeAs |

**Fig. 1 is not here.** It is a schematic drawn in a graphics program; no code produces it.

The plotted content is vector throughout - text is text, lines are lines. The bitmaps are
the point clouds (400 dpi), the fatband panels (~380 ppi) and the crystal structures
(440-880 ppi, against about 200 ppi in the published figures). Every font is embedded:
Times New Roman for the text, Computer Modern for the mathematics, and Cambria Math for the
H_sigma labels, which is the face the published figures use.

## Regenerating them

```bash
python deposit/figures/make_all_figures.py     # every panel -> deposit/figures/generated/
python deposit/figures/compose_figures.py      # the five composites -> composed/
```

`deposit/figures/README.md` is the full account: what each panel is drawn from, how the
type sizes and panel proportions were matched to the published figures, which point clouds
are rasterized and why, and how the memory was brought down.

Needs Python with the packages in `deposit/requirements.txt` (matplotlib >= 3.8), and
**Times New Roman** - not redistributable, so it is not here. Drop `times.ttf`,
`timesbd.ttf`, `timesi.ttf`, `timesbi.ttf` into `deposit/figures/fonts/` or point
`$TIMES_FONT_DIR` at them. Without it the figures still draw, in the metric-compatible
Liberation Serif, and every script says so.

## The other folders

`deposit/` is the code repository, complete: the figure code, its input data, and
`figures/panels/` - the individual panels the composites are assembled from, named as the
manuscript names them, PDF and PNG (the fatbands are PNG only, because the notebook that
draws them writes PNG). `figures/panels/extra/` holds outputs the manuscript does not use
but which check the ones it does.

`reviewer_verification/`, if it is here, holds the published PDFs. They are not part of the
code repository. The only thing still read out of them is the Cambria Math subset for the
H_sigma labels; without it those labels are set in Times New Roman Italic instead, at sizes
matched to the published glyphs.

## Not copied here

`deposit/figures/reference_plots/{figure2,figure3abc,SI_3}.{pdf,png}` in the full deposit
are older outputs of three scripts that need `shap_recompute.py` to have been run first,
and they predate those scripts being put on the shared font setup - they are in DejaVu Sans,
not Times New Roman. The maintained versions of the same three panels are here, as
`pdf/Fig2.pdf`, `deposit/figures/panels/Fig3(a-c).pdf` and `pdf/SI_S3.pdf`.

## `CHECKSUMS.md5`

```bash
cd <this folder> && md5sum -c CHECKSUMS.md5
```
"""


def md5(path):
    with open(path, 'rb') as fh:
        return hashlib.md5(fh.read()).hexdigest()


def clear_files(root):
    """Delete every file under `root`, leaving the directories.

    Not shutil.rmtree: on a VirtualBox shared folder the guest cannot rmdir at all
    ("Operation not permitted"), however the directory is owned or moded. Files delete
    fine, so the bundle is rebuilt by emptying it and writing over it rather than by
    removing and recreating it. Directories left behind are harmless - a stale one ends
    up empty - and CHECKSUMS.md5 lists files, so it stays exact either way.
    """
    n = 0
    for base, _, files in os.walk(root):
        for f in files:
            os.remove(os.path.join(base, f))
            n += 1
    return n


def copy_tree(src, dst, skip=()):
    shutil.copytree(src, dst, ignore=IGNORE, dirs_exist_ok=True)
    for rel in skip:
        p = os.path.join(dst, rel)
        if os.path.exists(p):
            os.remove(p)
    return sum(len(f) for _, _, f in os.walk(dst))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--verify', action='store_true',
                    help='rebuild the composites inside the bundle and diff them')
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    if os.path.isdir(out):
        print(f'  cleared {clear_files(out)} file(s) from the previous build')
    os.makedirs(os.path.join(out, 'pdf'), exist_ok=True)
    if not all(os.path.isfile(os.path.join(COMPOSED, f'{k}.pdf'))
               for k in ('Fig3', 'Fig4', 'Fig5', 'SI_5', 'SI_6')):
        print('  composites not built yet - running compose_figures.py')
        subprocess.run([sys.executable, os.path.join(HERE, 'compose_figures.py')],
                       check=True)
    for name, src in FIGURES:
        s = (os.path.join(COMPOSED, os.path.basename(src)) if src.startswith('composed/')
             else os.path.join(PANELS, src))
        shutil.copyfile(s, os.path.join(out, 'pdf', name))
        print(f'  pdf/{name:12s} <- {src}')

    kept = [f for f in PUBLISHED_FILES if os.path.isfile(os.path.join(PUBLISHED, f))]
    if kept:
        os.makedirs(os.path.join(out, 'reviewer_verification'), exist_ok=True)
        for f in kept:
            shutil.copyfile(os.path.join(PUBLISHED, f),
                            os.path.join(out, 'reviewer_verification', f))
    print(f'  reviewer_verification/ {len(kept)} files'
          f'{" (not present - skipped)" if not kept else ""}')

    n = copy_tree(HERE, os.path.join(out, 'deposit', 'figures'), skip=SKIP)
    print(f'  deposit/figures/       {n} files')

    shutil.copyfile(os.path.join(REPO, 'requirements.txt'),
                    os.path.join(out, 'deposit', 'requirements.txt'))
    with open(os.path.join(out, 'README.md'), 'w', encoding='utf-8') as fh:
        fh.write(README)
    print('  README.md')

    files = []
    for root, dirs, fs in os.walk(out):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for f in sorted(fs):
            if f != 'CHECKSUMS.md5':
                files.append(os.path.relpath(os.path.join(root, f), out))
    files.sort()
    with open(os.path.join(out, 'CHECKSUMS.md5'), 'w') as fh:
        fh.write('\n'.join(f'{md5(os.path.join(out, p))}  {p}' for p in files) + '\n')
    print(f'  CHECKSUMS.md5          {len(files)} entries')

    if a.verify:
        # into a temp directory, not into the bundle: see clear_files() on why a stray
        # directory inside it could not be removed again
        tmp = tempfile.mkdtemp(prefix='bundle_verify_')
        env = dict(os.environ, COMPOSE_OUTDIR=tmp)
        r = subprocess.run([sys.executable,
                            os.path.join(out, 'deposit', 'figures', 'compose_figures.py')],
                           env=env, capture_output=True, text=True)
        print(r.stdout.strip() or r.stderr.strip()[-800:])
        # A PDF is not byte-reproducible - it carries a creation timestamp - so the check
        # is that each composite was rebuilt and came out the same size as the one shipped.
        for k, shipped in (('Fig3', 'Fig3.pdf'), ('Fig4', 'Fig4.pdf'), ('Fig5', 'Fig5.pdf'),
                           ('SI_5', 'SI_S5.pdf'), ('SI_6', 'SI_S6.pdf')):
            built = os.path.join(tmp, f'{k}.pdf')
            if not os.path.exists(built):
                print(f'    {k:5s} NOT REBUILT'); continue
            a = os.path.getsize(built)
            b = os.path.getsize(os.path.join(out, 'pdf', shipped))
            ok = abs(a - b) < max(2048, b * 0.01)
            print(f'    {k:5s} rebuilt {a/1e6:5.2f} MB vs shipped {b/1e6:5.2f} MB'
                  f'   {"ok" if ok else "DIFFERENT"}')
        shutil.rmtree(tmp, ignore_errors=True)
    print(f'\n  {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
