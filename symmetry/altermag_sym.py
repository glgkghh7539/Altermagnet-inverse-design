#!/usr/bin/env python
"""Determine the operation connecting the two magnetic sublattices - reviewer Point 1.

For a compensated antiferromagnet (two magnetic atoms, m1 = -m2), find and classify the
crystal symmetry operation that maps one spin sublattice onto the other.

  - translation (W = I)   -> sublattices related by a translation => conventional AFM, splitting forbidden
  - inversion (W = -I)    -> sublattices related by inversion     => conventional AFM, splitting forbidden
  - rotation or mirror    -> related only by a rotation           => ALTERMAGNET
If no connecting operation exists at all, the two sites are symmetry-inequivalent
(a compensated-ferrimagnet candidate).
"""
import sys, os, glob, warnings, numpy as np
warnings.filterwarnings('ignore')
import spglib

MAG=set("Sc Ti V Cr Mn Fe Co Ni Cu Zn Y Zr Nb Mo Ru Rh Pd Ag Cd".split())
Z={s:i+1 for i,s in enumerate(
 "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
 "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd "
 "Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi".split())}

def read_poscar(p):
    L=open(p,errors='replace').read().splitlines()
    s=float(L[1].split()[0])
    lat=np.array([[float(x) for x in L[i].split()[:3]] for i in (2,3,4)])*s
    i=5
    try: [float(x) for x in L[5].split()]; sp=None
    except ValueError: sp=L[5].split(); i=6
    cnt=[int(x) for x in L[i].split()]; i+=1
    if L[i].strip()[:1] in 'sS': i+=1
    direct=L[i].strip()[:1] in 'dD'; i+=1
    n=sum(cnt)
    co=np.array([[float(x) for x in L[i+j].split()[:3]] for j in range(n)])
    if not direct: co=co@np.linalg.inv(lat)
    if sp is None: return None
    species=[]
    for s_,c in zip(sp,cnt): species+= [s_]*c
    return lat, co%1.0, species

def classify(p, symprec):
    r=read_poscar(p)
    if r is None: return None
    lat, co, species = r
    nums=[Z.get(s,0) for s in species]
    cell=(lat, co, nums)
    ds=spglib.get_symmetry_dataset(cell, symprec=symprec)
    if ds is None: return None
    sg = ds.international if hasattr(ds,'international') else ds['international']
    rot= ds.rotations   if hasattr(ds,'rotations')   else ds['rotations']
    tra= ds.translations if hasattr(ds,'translations') else ds['translations']
    mi=[i for i,s in enumerate(species) if s in MAG]
    if len(mi)!=2: return dict(sg=sg, verdict='not_two_magnetic_atoms', nmag=len(mi))
    A,B = co[mi[0]], co[mi[1]]
    I=np.eye(3,dtype=int)
    kinds=[]
    for W,w in zip(rot,tra):
        img=(W@A+w)%1.0
        d=img-B; d-=np.round(d)
        if np.abs(d@lat).max()<symprec*2:      # an operation mapping A onto B
            if np.array_equal(W,I):            kinds.append('translation')
            elif np.array_equal(W,-I):         kinds.append('inversion')
            elif round(np.linalg.det(W))==1:   kinds.append('rotation')
            else:                              kinds.append('mirror/rotoinv')
    k=set(kinds)
    if not k:                                   v='compensated_FiM_candidate'
    elif 'translation' in k:                    v='conventional_AFM_translation'
    elif 'inversion' in k:                      v='conventional_AFM_inversion'
    else:                                       v='ALTERMAGNET'
    return dict(sg=sg, verdict=v, ops=sorted(k), n_ops=len(kinds), nsym=len(rot))

if __name__=='__main__':
    files=sys.argv[2:]
    sp=float(sys.argv[1])
    print(f'{"structure":26s} {"space group":12s} {"connecting op":22s} {"verdict"}')
    for f in files:
        try:
            r=classify(f, sp)
            n=os.path.basename(f).replace('POSCAR_','')
            if r is None: print(f'{n:26s} (read failed)'); continue
            print(f'{n:26s} {r["sg"]:12s} {",".join(r.get("ops",[])) or "-":22s} {r["verdict"]}')
        except Exception as e:
            print(f'{os.path.basename(f):26s} ERROR {e}')
