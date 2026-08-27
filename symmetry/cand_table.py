#!/usr/bin/env python
"""SI master table for the six candidates: space group, magnetic space group, lattice,
moments and the energy differences between magnetic configurations."""
import sys, os, csv, json, warnings, numpy as np
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
def cellpar(lat):
    a,b,c=[np.linalg.norm(v) for v in lat]
    ang=lambda u,v: np.degrees(np.arccos(np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v))))
    return a,b,c,ang(lat[1],lat[2]),ang(lat[0],lat[2]),ang(lat[0],lat[1])
def analyze(f, m1, symprec=0.05):
    lat,co,sp=read(f); nums=[Z.get(x,0) for x in sp]
    mi=[i for i,x in enumerate(sp) if x in MAG][:2]
    ds=spglib.get_symmetry_dataset((lat,co,nums), symprec=symprec)
    sg = ds.international if hasattr(ds,'international') else ds['international']
    sgn= ds.number        if hasattr(ds,'number')        else ds['number']
    magmoms=[0.0]*len(sp); magmoms[mi[0]]=abs(m1); magmoms[mi[1]]=-abs(m1)
    msg='?'
    try:
        md=spglib.get_magnetic_symmetry_dataset((lat,co,nums,magmoms), symprec=symprec)
        uni = md.uni_number if hasattr(md,'uni_number') else md['uni_number']
        t   = md.msg_type   if hasattr(md,'msg_type')   else md['msg_type']
        mt  = spglib.get_magnetic_spacegroup_type(uni)
        msg = f"{mt['bns_number']} ({mt['uni_number']}, type {t})"
    except Exception as e:
        msg = 'MSG determination failed: '+str(e)[:30]
    return sg, sgn, msg, cellpar(lat), sp, co, mi
if __name__=='__main__':
    base=sys.argv[1]
    US={}
    for r in csv.DictReader(open(sys.argv[2]),delimiter='\t'):
        US[(r['system'],r['config'],int(r['U']))]=r
    PROD={'NiS':7,'CoS':3,'FeAs':4,'FeS':4,'CrS':4,'CoO_sp':3}
    SSE={'NiS':0.8707,'CoS':1.0824,'CoO_sp':0.9735,'FeS':1.3548,'FeAs':1.4023,'CrS':0.4106}
    out=[]
    for n in ['NiS','CoS','FeAs','FeS','CrS','CoO_sp']:
        f=os.path.join(base,f'POSCAR_{n}__4relaxed')
        U=PROD[n]; a=US.get((n,'AFM',U))
        m1=abs(float(a['m1_tot'])) if a else 0.0
        sg,sgn,msg,cp,sp,co,mi=analyze(f,m1)
        e={c:(float(US[(n,c,U)]['E_sigma0']) if (n,c,U) in US else None) for c in ['AFM','FM','NM']}
        d=lambda c: (e[c]-e['AFM'])*1000/4 if e[c] is not None else float('nan')
        out.append(dict(name=n,U=U,sg=sg,sgn=sgn,msg=msg,
            a=cp[0],b=cp[1],c=cp[2],alpha=cp[3],beta=cp[4],gamma=cp[5],
            m1=float(a['m1_tot']),m2=float(a['m2_tot']),
            dFM=d('FM'),dNM=d('NM'),sse=SSE[n],
            species='/'.join(sorted(set(sp))),
            sites=';'.join(f'{sp[i]}({co[i][0]:.4f},{co[i][1]:.4f},{co[i][2]:.4f})' for i in range(len(sp)))))
    w=csv.DictWriter(open(sys.argv[3],'w',newline=''),fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print(f'{"candidate":10s} {"U":>2s} {"space grp":11s} {"MSG (BNS)":22s} {"a":>7s} {"c":>7s} {"gamma":>7s} {"m(M)":>7s} {"dE(FM)":>8s} {"dE(NM)":>8s} {"SSE":>7s}')
    for r in out:
        print(f'{r["name"]:8s} {r["U"]:2d} {r["sg"]:11s} {r["msg"]:22s} {r["a"]:7.4f} {r["c"]:7.4f} {r["gamma"]:7.2f} '
              f'{r["m1"]:+7.3f} {r["dFM"]:+8.1f} {r["dNM"]:+8.1f} {r["sse"]:7.4f}')
