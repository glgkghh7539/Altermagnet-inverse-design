#!/usr/bin/env python3
"""Read the two local moments out of every stored OUTCAR.

`SSE_ax.py` already reports `ion1_tot`, `ion2_tot` and `overall_tot` while it screens, but
it only wrote them for the structures in one run, so the deposited
`data/raw/spin_splitting_summary.csv` carries the second moment for 2,565 of the 3,845
structures. The referee asked for both moments of every structure. This walks a directory
of finished calculations and reads them again, through the same
`parse_eigenval.check_magnetization()` the screening used - the last `magnetization (x)`
block of the OUTCAR - so the numbers come from the same place, not a second convention.

    python screening/extract_moments.py                       # -> data/raw/local_moments.csv
    OUTCAR_ROOT=/path/to/runs python screening/extract_moments.py

The layout it expects is one directory per structure, named as the structure is named in
`fin_data.csv`, each holding an `OUTCAR`:

    <root>/POSCAR_Ag2F6_3/OUTCAR
    <root>/POSCAR_Ag2F6_3_st950/OUTCAR

The OUTCARs are far too large to deposit - about 78 GB for the set this was run on - so
this script is deposited and its output is deposited, but its input is not.

Sign convention: these are the signed values. `fin_data.csv` carries |m1| and |m_total|.
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_eigenval import check_magnetization        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get('OUTCAR_ROOT',
                      os.path.join(HERE, '..', '..', 'packaged_calculations'))
OUT = os.environ.get('MOMENTS_OUT',
                     os.path.join(HERE, '..', 'data', 'raw', 'local_moments.csv'))
HEADER = ['filename', 'm1_muB', 'm2_muB', 'm_total_muB', 'condition_met']


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default=ROOT)
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    if not os.path.isdir(root):
        sys.exit(f'no such directory: {root}\nSet --root or $OUTCAR_ROOT.')

    dirs = sorted(d for d in os.listdir(root)
                  if d.startswith('POSCAR_')
                  and os.path.isfile(os.path.join(root, d, 'OUTCAR')))
    print(f'  {len(dirs):,} structures with an OUTCAR under {root}')

    rows, bad = [], 0
    for i, d in enumerate(dirs, 1):
        try:
            ok, tot, m1, m2 = check_magnetization(os.path.join(root, d, 'OUTCAR'))
        except Exception as exc:                       # a truncated or half-written OUTCAR
            print(f'  [warn] {d}: {exc}')
            bad += 1
            continue
        if m1 is None and m2 is None and tot is None:
            bad += 1
            continue
        rows.append([d, m1, m2, tot, ok])
        if i % 250 == 0:
            print(f'    {i:,} / {len(dirs):,}', flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        w.writerows(rows)
    anti = sum(1 for r in rows if r[1] is not None and r[2] is not None and r[1] * r[2] < 0)
    both = sum(1 for r in rows if r[1] is not None and r[2] is not None)
    print(f'\n  written: {a.out}   ({len(rows):,} rows, {bad} unreadable)')
    print(f'  both moments read for {both:,}; antiparallel in {anti:,} '
          f'({100 * anti / max(1, both):.1f} %)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
