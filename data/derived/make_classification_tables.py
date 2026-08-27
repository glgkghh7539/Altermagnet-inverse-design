#!/usr/bin/env python3
"""Build the magnetic-classification tables that answer referee point 1.

The referee asked, of the released dataset, for an explicit magnetic-symmetry
classification of the parent structures, the operation that relates the two spin
sublattices, the same check repeated on the strained children (strain can remove the
operation), and the two local moments. Those facts are in the deposit already, but spread
over five files keyed on two different columns. This joins them into two tables:

    altermagnet_classification.csv          one row per structure   (3,845)
    altermagnet_classification_parents.csv  one row per parent      (322)

Nothing is recomputed here - every column is carried over or aggregated.

    python data/derived/make_classification_tables.py

Inputs, all under data/:
    fin_data.csv                            sse, gamma-point average, |m1|, |total|
    raw/magnetic_symmetry_all.csv           verdict + sublattice operation, per structure,
                                            at symprec 0.05
    raw/magnetic_symmetry_all_symprec001.csv the same at symprec 0.01, the tolerance the
                                            response letter reports alongside it
    raw/magnetic_symmetry_parents.csv       the same, per parent
    raw/magnetic_spacegroup_type_parents.csv MSG type and the BNS / UNI numbers, with the
                                            moments given as collinear scalars
    raw/magnetic_spacegroup_type_parents_c.csv the same with the moments along c, the direction
                                            the calculations impose. The MSG acts on them as
                                            axial vectors, so the type differs: 96 of the 322
                                            parents change, all of them between type I and
                                            type III. The two splitting-forbidding classes do
                                            not move.
    raw/local_moments.csv                   the signed moments, m1 and m2, read back out
                                            of every stored OUTCAR by
                                            screening/extract_moments.py
    raw/spin_splitting_summary.csv          the same, from the screening run, for the
                                            subset it covered - used only as a fallback

A note on signs. fin_data.csv carries magnitudes: its `ion1 tot` is |m1| and its `tot_mag`
is |m_total|. The signed values come from raw/local_moments.csv where it exists and from
raw/spin_splitting_summary.csv otherwise; the two agree, and both agree with fin_data.csv
once the sign is taken off, because all three trace back to the same
parse_eigenval.check_magnetization().
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..')
OUT = os.environ.get('CLASSIFICATION_OUTDIR', HERE)


def stem(name):
    """`POSCAR_Ag2F6_3_x950` -> `Ag2F6_3_x950`.

    fin_data.csv keys on the POSCAR file name; the symmetry tables key on the structure
    name without that prefix. Joining them without stripping it silently produces an empty
    intersection, and every downstream comparison then reads as "changed".
    """
    return name[len('POSCAR_'):] if name.startswith('POSCAR_') else name


def strain_tag(filename, parent):
    """`POSCAR_Ag2F6_3_x950` under parent `Ag2F6_3` -> `x950`; the parent itself -> `none`."""
    stem = filename[len('POSCAR_'):] if filename.startswith('POSCAR_') else filename
    if stem == parent:
        return 'none'
    return stem[len(parent) + 1:] if stem.startswith(parent + '_') else ''


def main():
    fin = pd.read_csv(os.path.join(DATA, 'fin_data.csv'))
    sym = pd.read_csv(os.path.join(DATA, 'raw', 'magnetic_symmetry_all.csv'))
    tight = pd.read_csv(os.path.join(DATA, 'raw', 'magnetic_symmetry_all_symprec001.csv'))
    par = pd.read_csv(os.path.join(DATA, 'raw', 'magnetic_symmetry_parents.csv'))
    msg = pd.read_csv(os.path.join(DATA, 'raw', 'magnetic_spacegroup_type_parents.csv'))
    msg_c = pd.read_csv(os.path.join(DATA, 'raw', 'magnetic_spacegroup_type_parents_c.csv'))
    sgn = pd.read_csv(os.path.join(DATA, 'raw', 'spin_splitting_summary.csv'))
    lm_path = os.path.join(DATA, 'raw', 'local_moments.csv')
    lm = pd.read_csv(lm_path) if os.path.isfile(lm_path) else None

    t = fin[['filename', 'parent', 'sse', 'gamma point average splitting',
             'ion1 tot', 'tot_mag']].rename(columns={
                 'sse': 'sse_eV',
                 'gamma point average splitting': 'gamma_avg_meV',
                 'ion1 tot': 'm1_abs_muB',
                 'tot_mag': 'm_total_abs_muB'})
    t['strain'] = [strain_tag(f, p) for f, p in zip(t.filename, t.parent)]
    t['is_parent'] = t.strain == 'none'
    t['structure'] = t.filename.map(stem)

    # per-structure magnetic symmetry, i.e. the strained children checked again
    t = t.merge(sym.rename(columns={'filename': 'structure', 'sg': 'spacegroup'}),
                on='structure', how='left')

    # the same classification at the tighter tolerance. The connecting operation of a
    # relaxed CONTCAR sits 1e-3 to 1e-4 off the ideal Wyckoff position, so which structures
    # keep it at 0.01 is the tolerance sensitivity the response letter reports.
    t = t.merge(tight[['filename', 'verdict', 'ops', 'n_ops']].rename(columns={
        'filename': 'structure', 'verdict': 'verdict_symprec001',
        'ops': 'ops_symprec001', 'n_ops': 'n_ops_symprec001'}),
        on='structure', how='left')

    # the parent's magnetic space group, carried onto every child
    pm = par[['filename', 'verdict']].rename(columns={'filename': 'parent',
                                                      'verdict': 'parent_verdict'})
    t = t.merge(pm, on='parent', how='left')
    t = t.merge(msg.rename(columns={'filename': 'parent'}), on='parent', how='left')
    t = t.merge(msg_c.rename(columns={'filename': 'parent', 'msg_type': 'msg_type_c',
                                      'bns': 'bns_c', 'uni': 'uni_c'}),
                on='parent', how='left')

    for col, label in ((t.verdict, 'magnetic_symmetry_all'),
                       (t.parent_verdict, 'magnetic_symmetry_parents'),
                       (t.verdict_symprec001, 'magnetic_symmetry_all_symprec001'),
                       (t.msg_type, 'magnetic_spacegroup_type_parents')):
        if col.isna().any():
            sys.exit(f'{int(col.isna().sum())} of {len(t)} rows did not join against '
                     f'{label}.csv - check the key columns')

    # the signed moments: the OUTCAR sweep first, the screening run for anything it missed
    s = sgn[['structure', 'ion1_tot', 'ion2_tot']].rename(columns={
        'structure': 'filename', 'ion1_tot': 'm1_muB', 'ion2_tot': 'm2_muB'})
    if lm is not None:
        s = (lm[['filename', 'm1_muB', 'm2_muB']]
             .set_index('filename')
             .combine_first(s.set_index('filename'))
             .reset_index())
    t = t.merge(s, on='filename', how='left')
    # The magnitudes in fin_data.csv are the check on the signed values: |m1_muB| has to
    # reproduce `ion1 tot`. It does, to the last digit, for all but two structures, where
    # the stored OUTCAR gives 3.793 against 3.795 and 2.250 against 2.251 - a re-run, and
    # 0.002 muB either way. Anything past 0.005 muB is a different calculation, not
    # rounding, and stops the build rather than going into the table unnoticed.
    off = (t.m1_muB.abs() - t.m1_abs_muB).abs()
    if (off > 5e-3).any():
        sys.exit(f'{int((off > 5e-3).sum())} rows where |m1_muB| disagrees with '
                 f"fin_data's `ion1 tot` by more than 0.005 muB")
    near = t[(off > 5e-4) & (off <= 5e-3)]
    for _, r in near.iterrows():
        print(f'  [note] {r.filename}: OUTCAR gives |m1| = {abs(r.m1_muB):.3f}, '
              f"fin_data has {r.m1_abs_muB:.3f}")
    both = t.m1_muB.notna() & t.m2_muB.notna()
    t['antiparallel'] = np.where(both, t.m1_muB * t.m2_muB < 0, None)
    t['m_asymmetry_muB'] = (t.m1_muB.abs() - t.m2_muB.abs()).abs()
    # strain changes the operation: does the child still classify as the parent does?
    t['verdict_same_as_parent'] = t.verdict == t.parent_verdict

    cols = ['filename', 'structure', 'parent', 'is_parent', 'strain',
            'verdict', 'ops', 'n_ops', 'nsym', 'spacegroup',
            'verdict_symprec001', 'ops_symprec001', 'n_ops_symprec001',
            'parent_verdict', 'verdict_same_as_parent',
            'msg_type', 'bns', 'uni', 'msg_type_c', 'bns_c', 'uni_c',
            'sse_eV', 'gamma_avg_meV',
            'm1_muB', 'm2_muB', 'm1_abs_muB', 'm_asymmetry_muB', 'antiparallel',
            'm_total_abs_muB']
    t = t[cols].sort_values(['parent', 'strain']).reset_index(drop=True)

    # ---- per parent -------------------------------------------------------------
    g = t.groupby('parent')
    p = pd.DataFrame({
        'n_structures': g.size(),
        'n_altermagnet': g.verdict.apply(lambda v: (v == 'ALTERMAGNET').sum()),
        'n_altermagnet_symprec001': g.verdict_symprec001.apply(
            lambda v: (v == 'ALTERMAGNET').sum()),
        'n_verdict_changed_by_strain': g.verdict_same_as_parent.apply(lambda v: (~v).sum()),
        'sse_max_eV': g.sse_eV.max(),
        'sse_mean_eV': g.sse_eV.mean(),
        'gamma_avg_max_meV': g.gamma_avg_meV.max(),
        'm1_abs_min_muB': g.m1_abs_muB.min(),
        'm1_abs_max_muB': g.m1_abs_muB.max(),
        'm_asymmetry_max_muB': g.m_asymmetry_muB.max(),
        'm_total_abs_max_muB': g.m_total_abs_muB.max(),
        'n_with_m2': g.m2_muB.apply(lambda v: v.notna().sum()),
        'n_antiparallel': g.antiparallel.apply(lambda v: (v == True).sum()),
    }).reset_index()
    p = (par.rename(columns={'filename': 'parent', 'sg': 'spacegroup'})
            .merge(msg.rename(columns={'filename': 'parent'}), on='parent')
            .merge(msg_c.rename(columns={'filename': 'parent', 'msg_type': 'msg_type_c',
                                         'bns': 'bns_c', 'uni': 'uni_c'}), on='parent')
            .merge(p, on='parent'))
    p = p[['parent', 'verdict', 'ops', 'n_ops', 'nsym', 'spacegroup',
           'msg_type', 'bns', 'uni', 'msg_type_c', 'bns_c', 'uni_c'] + [
               c for c in p.columns if c.startswith(
               ('n_', 'sse_', 'gamma_', 'm1_', 'm_'))
               and c not in ('n_ops', 'nsym')]]
    p = p.sort_values('sse_max_eV', ascending=False).reset_index(drop=True)

    os.makedirs(OUT, exist_ok=True)
    for df, name in ((t, 'altermagnet_classification.csv'),
                     (p, 'altermagnet_classification_parents.csv')):
        path = os.path.join(OUT, name)
        df.to_csv(path, index=False)
        print(f'  written: {path}   ({len(df):,} rows x {len(df.columns)} cols)')

    print(f'\n  parents {len(p)}   structures {len(t)}')
    print('  parent verdict :', p.verdict.value_counts().to_dict())
    print('  MSG type       :', p.msg_type.value_counts().sort_index().to_dict(),
          ' (collinear)')
    print('  MSG type along c:', p.msg_type_c.value_counts().sort_index().to_dict(),
          f' - {int((p.msg_type != p.msg_type_c).sum())} parents differ')
    print(f'  m2 known for   : {int(t.m2_muB.notna().sum()):,} of {len(t):,} structures')
    k = t.antiparallel == True
    print(f'  antiparallel   : {int(k.sum()):,} of {int(t.m2_muB.notna().sum()):,} '
          f'({100 * k.sum() / max(1, t.m2_muB.notna().sum()):.1f} %)')
    print(f'  |m1|-|m2| max  : {t.m_asymmetry_muB.max():.4f} muB')
    ch = t[~t.is_parent]
    print(f'  strained children whose verdict differs from their parent: '
          f'{int((~ch.verdict_same_as_parent).sum()):,} of {len(ch):,}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
