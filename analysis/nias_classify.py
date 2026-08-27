import numpy as np, glob, os, sys, re
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'match.py')).read().split('init_dir, zip_dir')[0])

def nias_metrics(d):
    """Deviation from the NiAs type (P6_3/mmc, M at 2a and X at 2c)."""
    lat, co, cnt = d['lat'], d['co'], d['cnt']
    if cnt != [2,2]: return None
    a,b,c = [np.linalg.norm(v) for v in lat]
    g = np.degrees(np.arccos(np.dot(lat[0],lat[1])/(a*b)))
    M, X = co[:2], co[2:]
    # are the M sites at (0,0,0) and (0,0,1/2)?  (origin shift allowed: first M taken as origin)
    M0 = (M - M[0]) % 1.0
    dM = min(np.abs(M0[1]-[0,0,0.5]).max(), np.abs(M0[1]-[0,0,-0.5]).max())
    Xs = (X - M[0]) % 1.0
    ideal = np.array([[1/3,2/3,0.25],[2/3,1/3,0.75]])
    best = 9e9
    for p in ([0,1],[1,0]):
        dd = Xs[p] - ideal; dd -= np.round(dd)
        best = min(best, np.sqrt((dd**2).sum()/2))
    return dict(a=a,b=b,c=c,gamma=g, ba=b/a, ca=c/a,
                dM=dM, dX=best,
                hexdev=abs(b/a-1)+abs(g-120)/120)

zip_dir, init_dir = sys.argv[1], sys.argv[2]
rows=[]
for f in glob.glob(os.path.join(zip_dir,'POSCAR_*')):
    d=read_poscar(f)
    if not d: continue
    m=nias_metrics(d)
    if m and m['dM']<0.02 and m['dX']<0.12:      # classified as NiAs type
        m['name']=os.path.basename(f)[7:]; rows.append(m)
fam=lambda k: re.sub(r'_(st|x|y|z)\d+$','',k)
print(f'parents classified as NiAs type: {len(rows)} structures / {len({fam(r["name"]) for r in rows})} compositions')
print()
print('--- the six candidates (initial, pre-relaxation structures) ---')
print(f'{"":10s} {"a":>7s} {"c/a":>6s} {"b/a":>7s} {"gamma":>7s} {"dX(vs ideal)":>12s}')
for f in sorted(glob.glob(os.path.join(init_dir,'*_init.vasp'))):
    d=read_poscar(f); m=nias_metrics(d)
    n=os.path.basename(f).replace('_init.vasp','')
    if m: print(f'{n:10s} {m["a"]:7.3f} {m["ca"]:6.3f} {m["ba"]:7.4f} {m["gamma"]:7.2f} {m["dX"]:12.4f}'
                + ('   <- ideal NiAs' if m['dX']<0.005 and abs(m['ba']-1)<0.005 else ''))
    else: print(f'{n:10s}   (not NiAs type)')
print()
print('--- Cr2S2_1 family ---')
for r in sorted([r for r in rows if fam(r['name'])=='Cr2S2_1'], key=lambda r:r['name']):
    print(f'{r["name"]:20s} {r["a"]:7.3f} {r["ca"]:6.3f} {r["ba"]:7.4f} {r["gamma"]:7.2f} {r["dX"]:12.4f}')
print()
print('--- NiAs-type parents ranked by closeness to the ideal type (top 10 by dX) ---')
for r in sorted(rows,key=lambda r:r['dX'])[:10]:
    print(f'{r["name"]:20s} {r["a"]:7.3f} {r["ca"]:6.3f} {r["ba"]:7.4f} {r["gamma"]:7.2f} {r["dX"]:12.4f}')
print()
import numpy as _n
dx=_n.array([r['dX'] for r in rows])
print(f'dX distribution: median {_n.median(dx):.4f}  90th pct {_n.percentile(dx,90):.4f}  max {dx.max():.4f}')
cr=[r for r in rows if fam(r['name'])=='Cr2S2_1']
if cr:
    q=_n.mean([(dx<r['dX']).mean() for r in cr])
    print(f'dX percentile of the Cr2S2_1 family: {100*q:.0f}%  (distortion is within the top {100*(1-q):.0f}%)')
