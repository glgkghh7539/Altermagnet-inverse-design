import sys, numpy as np, re, os
def sse(d):
    eig=os.path.join(d,'EIGENVAL'); out=os.path.join(d,'OUTCAR')
    if not (os.path.exists(eig) and os.path.exists(out)): return None
    t=open(out,errors='replace').read()
    m=re.findall(r'E-fermi\s*:\s*([-\d.]+)',t)
    if not m: return None
    ef=float(m[-1])
    L=open(eig,errors='replace').read().split('\n')
    nel,nk,nb=[int(x) for x in L[5].split()]
    i=6; best=0.0; nb_pairs=0
    for k in range(nk):
        while i<len(L) and L[i].strip()=='' : i+=1
        i+=1
        for b in range(nb):
            p=L[i].split(); i+=1
            if len(p)<3: continue
            eu,ed=float(p[1]),float(p[2])
            if ef-2.0<=eu<=ef and ef-2.0<=ed<=ef:
                nb_pairs+=1
                best=max(best,abs(eu-ed))
        i+=0
    return ef,nel,nk,nb,nb_pairs,best
for d in sys.argv[1:]:
    r=sse(d)
    n=os.path.basename(d.rstrip('/'))
    if r is None: print(f'{n:12s} (file not found)'); continue
    ef,nel,nk,nb,np_,b=r
    print(f'{n:12s} E_F={ef:8.4f}  NELECT={nel:3d}  NKPTS={nk:5d}  NBANDS={nb:3d}  eligible_pairs={np_:6d}  SSE={b:.4f} eV')
