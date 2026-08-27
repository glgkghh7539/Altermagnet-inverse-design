"""Shared helper: identify the magnetic (M) and non-magnetic (X) species of every row.

`fin_data.csv` carries `magnetic_atomic_number` and `nonmagnetic_atomic_number` but not the
symbols, and the `filename` carries the formula but not which of its two elements is the
magnetic one (`POSCAR_Ag2F6_3` leads with the metal, `POSCAR_F6Ag2_3`-style names do not).
Combining the two settles it, and the atomic numbers double as a check: every one of the
3,845 rows resolves, and a row that did not would raise here rather than be dropped silently.

Used by plot_si1_dataset_stats.py and plot_si2_mx_heatmap.py.
"""
import math
import os
import re

import pandas as pd

_SYMBOLS = (
    'H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn '
    'Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce '
    'Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn'
).split()
SYMBOL = {z + 1: s for z, s in enumerate(_SYMBOLS)}
NUMBER = {s: z + 1 for z, s in enumerate(_SYMBOLS)}

HERE = os.path.dirname(os.path.abspath(__file__))
FIN_DATA = os.environ.get('FIN_DATA', os.path.join(HERE, '..', '..', 'data', 'fin_data.csv'))

_FORMULA = re.compile(r'^POSCAR_([A-Z][a-z]?)(\d+)([A-Z][a-z]?)(\d+)')


def composition_label(a, b):
    """(1, 2) -> 'MX2';  (2, 3) -> 'M2X3'."""
    return ('M' if a == 1 else f'M{a}') + ('X' if b == 1 else f'X{b}')


def load(path=None):
    """Return fin_data with M, X, the reduced M:X counts and the composition label added."""
    path = path or FIN_DATA
    df = pd.read_csv(path, usecols=['filename', 'sse', 'magnetic_atomic_number',
                                    'nonmagnetic_atomic_number'])
    M, X, A, B = [], [], [], []
    for fn, zm, zx in zip(df.filename, df.magnetic_atomic_number,
                          df.nonmagnetic_atomic_number):
        m = _FORMULA.match(fn)
        if m is None:
            raise ValueError(f'cannot read a binary formula out of {fn!r}')
        e1, n1, e2, n2 = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
        sm, sx = SYMBOL[int(zm)], SYMBOL[int(zx)]
        if (e1, e2) == (sm, sx):
            nm, nx = n1, n2
        elif (e2, e1) == (sm, sx):
            nm, nx = n2, n1
        else:
            raise ValueError(f'{fn}: formula {e1}{n1}{e2}{n2} does not contain '
                             f'the tabulated pair {sm}/{sx}')
        g = math.gcd(nm, nx)
        M.append(sm); X.append(sx); A.append(nm // g); B.append(nx // g)
    df['M'], df['X'], df['a'], df['b'] = M, X, A, B
    df['composition'] = [composition_label(a, b) for a, b in zip(A, B)]
    return df


def by_atomic_number(symbols):
    """Sort element symbols the way the published axes do - ascending atomic number."""
    return sorted(set(symbols), key=lambda s: NUMBER[s])
