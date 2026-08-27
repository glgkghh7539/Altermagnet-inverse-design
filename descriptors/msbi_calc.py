#!/usr/bin/env python
"""Compute MSBI for the given structures using the calculation of descriptor.ipynb cell 16 verbatim."""
import os, glob, sys, numpy as np
from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

MAG = ["Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Y","Zr","Nb","Mo","Ru","Rh","Pd","Ag","Cd"]

def find_motifs(f, num_neighbors=12):
    st = Structure.from_file(f)
    mi = [i for i,s in enumerate(st) if str(s.specie) in MAG][:2]
    if len(mi) < 2: raise ValueError('fewer than two magnetic atoms')
    cnn = CrystalNN(); out = {}
    for m in mi:
        c = st[m]
        nn = cnn.get_nn_info(st, m)
        co = np.array([i["site"].coords for i in nn if str(i["site"].specie) not in MAG])
        if co.shape[0] == 0: raise ValueError('no non-magnetic neighbours')
        d = np.linalg.norm(co - c.coords, axis=1)
        k = np.argsort(d)[:min(num_neighbors, co.shape[0])]
        out[m] = dict(motif_coords=co[k] - c.coords, avg_bond_length=float(np.mean(d[k])), n=len(k))
    return out

def rmsd_std(p, q):
    C = cdist(p, q, 'sqeuclidean')
    r, c = linear_sum_assignment(C)
    sd = C[r, c]
    return float(np.sqrt(sd.mean())), float(np.std(np.sqrt(sd)))

def p_metric(m1, m2, l1, l2):
    if m1.shape[0] != m2.shape[0]: raise ValueError(f'motif size mismatch: {m1.shape[0]} vs {m2.shape[0]}')
    ri, si = rmsd_std(m1, m2)
    rv, sv = rmsd_std(-m1, m2)
    l0 = (l1 + l2) / 2
    pm = (ri/l0) * (rv/l0)
    return pm, (si if ri <= rv else sv)/l0, 0 if ri <= rv else 1

print(f'{"structure":34s} {"n_lig":>5s} {"MSBI":>9s} {"p_std":>8s} {"is_inv":>6s} {"l0":>7s}')
for f in sorted(glob.glob(os.path.join(sys.argv[1], 'POSCAR_*'))):
    try:
        M = find_motifs(f)
        k = list(M)
        pm, ps, iv = p_metric(M[k[0]]['motif_coords'], M[k[1]]['motif_coords'],
                              M[k[0]]['avg_bond_length'], M[k[1]]['avg_bond_length'])
        l0 = (M[k[0]]['avg_bond_length'] + M[k[1]]['avg_bond_length'])/2
        print(f'{os.path.basename(f)[7:]:34s} {M[k[0]]["n"]:5d} {pm:9.4f} {ps:8.4f} {iv:6d} {l0:7.3f}')
    except Exception as e:
        print(f'{os.path.basename(f)[7:]:34s}   ERROR: {e}')
