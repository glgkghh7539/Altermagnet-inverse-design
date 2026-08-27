#!/usr/bin/env python
"""Quantify the fragility of the MPF face selection and apply a deterministic rule."""
import sys, os, csv, warnings, numpy as np
warnings.filterwarnings('ignore')
from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN
MAG=set("Sc Ti V Cr Mn Fe Co Ni Cu Zn Y Zr Nb Mo Ru Rh Pd Ag Cd".split())

def motif_normal_and_dim(st):
    """Plane normal and dimensionality from the two magnetic motifs (same intent as cell 05)."""
    cnn=CrystalNN(); mi=[i for i,s in enumerate(st) if str(s.specie) in MAG][:2]
    if len(mi)<2: return None
    V=[]
    for m in mi:
        nn=cnn.get_nn_info(st,m)
        lig=[i['site'].coords for i in nn if str(i['site'].specie) not in MAG]
        if not lig: return None
        V.append(np.array(lig)-st[m].coords)
    v=np.vstack(V)
    u,s_,vt=np.linalg.svd(v-v.mean(0))
    dim = 2 if s_[-1]/max(s_[0],1e-12) < 0.05 else 3     # flat motif -> 2D
    return vt[-1], dim, len(V[0])

def faces(st):
    a,b,c = st.lattice.matrix
    N=[np.cross(a,b), np.cross(b,c), np.cross(c,a)]
    A=[np.linalg.norm(x) for x in N]
    n=[x/np.linalg.norm(x) for x in N]
    return n, A, ['ab','bc','ca']

def audit(f):
    st=Structure.from_file(f)
    r=motif_normal_and_dim(st)
    if r is None: return None
    nrm, dim, nlig = r
    n,A,T=faces(st)
    ang=[float(np.degrees(np.arccos(np.clip(abs(np.dot(nrm,x)),0,1)))) for x in n]
    o=np.argsort(ang)
    gap = ang[o[1]]-ang[o[0]]
    a1,a2 = A[o[0]], A[o[1]]
    rel = abs(a1-a2)/max(a1,1e-12)
    return dict(dim=dim, nlig=nlig, ang=[round(x,3) for x in ang], areas=[round(x,4) for x in A],
                pick=T[o[0]], gap=gap, alt=T[o[1]], rel_area_change=rel,
                area=a1, area_alt=a2, median=float(np.median(A)))

if __name__=='__main__':
    zipdir, listfile, out = sys.argv[1], sys.argv[2], sys.argv[3]
    names=[l.strip() for l in open(listfile) if l.strip()]
    rows=[]
    for i,nm in enumerate(names):
        p=os.path.join(zipdir,'POSCAR_'+nm)
        try:
            a=audit(p)
            if a is None: continue
            a['filename']=nm; rows.append(a)
        except Exception: pass
        if (i+1)%800==0: print(f'  {i+1}/{len(names)}',flush=True)
    with open(out,'w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=['filename','dim','nlig','pick','alt','gap','area','area_alt','median','rel_area_change'],extrasaction='ignore')
        w.writeheader(); w.writerows(rows)
    d2=[r for r in rows if r['dim']==2]; d3=[r for r in rows if r['dim']==3]
    print(f'\n{len(rows)} rows total  (2D {len(d2)}, 3D {len(d3)})')
    for tol in (0.1,1.0,5.0):
        fr=[r for r in d2 if r['gap']<tol]
        big=[r for r in fr if r['rel_area_change']>0.01]
        print(f'  2D rows with angle gap < {tol:4.1f} deg : {len(fr):4d}   of which area differs by 1%+ : {len(big):4d}'
              + (f'  (median area change {np.median([r["rel_area_change"] for r in big])*100:.1f}%, max {max(r["rel_area_change"] for r in big)*100:.1f}%)' if big else ''))
    fr3=[r for r in d3 if r['gap']<1.0]
    print(f'  3D rows with angle gap < 1.0 deg : {len(fr3)}')
