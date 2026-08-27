#!/usr/bin/env python
"""Extract the magnetic space group type (I-IV) of the 322 parents and cross-check it
against the sublattice-connecting-operation classification."""
import sys, os, csv, warnings, numpy as np
warnings.filterwarnings('ignore')
import spglib
MAG=set("Sc Ti V Cr Mn Fe Co Ni Cu Zn Y Zr Nb Mo Ru Rh Pd Ag Cd".split())
Z={s:i+1 for i,s in enumerate(
 "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
 "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd "
 "Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi".split())}
def read(p):
    L=open(p,errors='replace').read().splitlines(); s=float(L[1].split()[0])
    lat=np.array([[float(x) for x in L[i].split()[:3]] for i in (2,3,4)])*s
    i=5
    try: [float(x) for x in L[5].split()]; sp=None
    except ValueError: sp=L[5].split(); i=6
    cnt=[int(x) for x in L[i].split()]; i+=1
    if L[i].strip()[:1] in 'sS': i+=1
    direct=L[i].strip()[:1] in 'dD'; i+=1
    co=np.array([[float(x) for x in L[i+j].split()[:3]] for j in range(sum(cnt))])
    if not direct: co=co@np.linalg.inv(lat)
    species=[]
    for s_,c in zip(sp,cnt): species+=[s_]*c
    return lat, co%1.0, species
zipdir, listfile, out, sp_ = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
names=[l.strip() for l in open(listfile) if l.strip()]
rows=[]
for i,n in enumerate(names):
    f=os.path.join(zipdir,'POSCAR_'+n)
    if not os.path.exists(f): continue
    try:
        lat,co,species=read(f); nums=[Z.get(x,0) for x in species]
        mi=[j for j,x in enumerate(species) if x in MAG]
        if len(mi)!=2: continue
        mm=[0.0]*len(species); mm[mi[0]]=1.0; mm[mi[1]]=-1.0
        md=spglib.get_magnetic_symmetry_dataset((lat,co,nums,mm), symprec=sp_)
        uni=md.uni_number; t=md.msg_type
        mt=spglib.get_magnetic_spacegroup_type(uni)
        rows.append(dict(filename=n, msg_type=int(t), bns=mt.bns_number, uni=int(uni)))
    except Exception:
        rows.append(dict(filename=n, msg_type=0, bns='determination_failed', uni=0))
    if (i+1)%100==0: print(f'  {i+1}/{len(names)}',flush=True)
csv.DictWriter(open(out,'w',newline=''),fieldnames=['filename','msg_type','bns','uni']).writerows([{'filename':'filename','msg_type':'msg_type','bns':'bns','uni':'uni'}]+rows)
import collections
c=collections.Counter(r['msg_type'] for r in rows)
NAMES={1:'type I (colorless)',2:'type II (grey, paramagnetic)',3:'type III (black-white, no antitranslation)',4:'type IV (black-white, with antitranslation)',0:'determination failed'}
print(f'\n{len(rows)} structures  symprec={sp_}')
for k,v in sorted(c.items()): print(f'  {NAMES[k]:42s} {v:4d}  ({100*v/len(rows):.1f}%)')
