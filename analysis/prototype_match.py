#!/usr/bin/env python3
"""Exhaustively match each pre-relaxation POSCAR against the parent structures it could derive from.

The comparison uses quantities invariant under species substitution and isotropic scaling:
  - lattice shape: the 3x3 matrix normalised by L/|a1| (captures axial ratios and angles together)
  - fractional coordinates: RMSD after per-species sorting, with periodic boundaries applied
"""
import numpy as np, os, glob, sys

def read_poscar(p):
    L = open(p, errors='replace').read().splitlines()
    if len(L) < 8: return None
    try: s = float(L[1].split()[0])
    except: return None
    lat = np.array([[float(x) for x in L[i].split()[:3]] for i in (2,3,4)]) * s
    i = 5
    try: [float(x) for x in L[5].split()]; sp = None
    except ValueError: sp = L[5].split(); i = 6
    try: cnt = [int(x) for x in L[i].split()]
    except: return None
    i += 1
    if L[i].strip()[:1] in 'sS': i += 1
    direct = L[i].strip()[:1] in 'dD'
    i += 1
    n = sum(cnt)
    if len(L) < i + n: return None
    try: co = np.array([[float(x) for x in L[i+j].split()[:3]] for j in range(n)])
    except: return None
    if not direct: co = co @ np.linalg.inv(lat)
    return dict(lat=lat, cnt=cnt, sp=sp, co=co % 1.0, n=n)

def latnorm(lat):
    return lat / np.linalg.norm(lat[0])

def coord_rmsd(A, B, cnt):
    """Sorted matching within each species group; the origin shift is fixed by the first atom."""
    best = 9e9
    for ref in range(len(A)):
        a = (A - A[ref] + 0.5) % 1.0
        for ref2 in range(len(B)):
            b = (B - B[ref2] + 0.5) % 1.0
            tot, o = 0.0, 0
            ok = True
            for c in cnt:
                ga, gb = a[o:o+c], b[o:o+c]
                ia = np.lexsort(ga.T[::-1]); ib = np.lexsort(gb.T[::-1])
                d = ga[ia] - gb[ib]
                d -= np.round(d)
                tot += (d**2).sum(); o += c
            if not ok: continue
            best = min(best, np.sqrt(tot/len(A)))
    return best

def score(a, b):
    dl = np.linalg.norm(latnorm(a['lat']) - latnorm(b['lat']))
    dc = coord_rmsd(a['co'], b['co'], a['cnt'])
    return dl, dc, dl + 3*dc

init_dir, zip_dir = sys.argv[1], sys.argv[2]
pool = sorted(glob.glob(os.path.join(zip_dir, 'POSCAR_*')))
P = {}
for f in pool:
    d = read_poscar(f)
    if d: P[os.path.basename(f)[7:]] = d
print(f'parent pool: {len(P)} structures read', flush=True)

for f in sorted(glob.glob(os.path.join(init_dir, '*_init.vasp'))):
    name = os.path.basename(f).replace('_init.vasp','')
    a = read_poscar(f)
    title = open(f, errors='replace').readline().strip()
    cand = [(k,v) for k,v in P.items() if v['cnt'] == a['cnt']]
    res = []
    for k, v in cand:
        dl, dc, s = score(a, v)
        res.append((s, dl, dc, k))
    res.sort()
    print(f'\n=== {name}   (title: "{title}")   {len(cand)} candidates with the same composition pattern ===')
    print(f'    {"rank":>4s} {"parent":26s} {"score":>8s} {"d_lattice":>10s} {"d_coord":>9s}')
    for i,(s,dl,dc,k) in enumerate(res[:6],1):
        print(f'    {i:4d} {k:26s} {s:8.4f} {dl:10.4f} {dc:9.4f}')
