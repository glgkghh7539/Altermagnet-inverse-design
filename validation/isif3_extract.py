import sys, os, re, numpy as np, glob
def rdpos(p):
    L=open(p,errors='replace').read().splitlines(); s=float(L[1].split()[0])
    lat=np.array([[float(x) for x in L[i].split()[:3]] for i in (2,3,4)])*s
    i=5
    try: [float(x) for x in L[5].split()]
    except ValueError: i=6
    cnt=[int(x) for x in L[i].split()]; i+=1
    if L[i].strip()[:1] in 'sS': i+=1
    i+=1
    co=np.array([[float(x) for x in L[i+j].split()[:3]] for j in range(sum(cnt))])
    return lat,co
def cellpar(lat):
    a,b,c=[np.linalg.norm(v) for v in lat]
    ang=lambda u,v: np.degrees(np.arccos(np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v))))
    return a,b,c,ang(lat[1],lat[2]),ang(lat[0],lat[2]),ang(lat[0],lat[1]),abs(np.linalg.det(lat))
def moments(out):
    txt=open(out,errors='replace').read()
    blocks=re.findall(r'magnetization \(x\)\n\s*\n# of ion.*?\n-+\n(.*?)\n-+',txt,re.S)
    if not blocks: return None
    rows=[l.split() for l in blocks[-1].strip().splitlines()]
    return [float(r[-1]) for r in rows]
W=sys.argv[1]
print(f'{"system":8s} {"pass":>4s} {"a":>7s} {"b":>7s} {"c":>7s} {"alpha":>7s} {"beta":>7s} {"gamma":>7s} {"V":>8s} {"dV%":>7s}  moments')
for n in ['CoS','FeS','FeAs','CrS','CoO_sp']:
    d=os.path.join(W,n)
    start=os.path.join(d,'POSCAR.start')
    if not os.path.exists(start): continue
    l0,_=rdpos(start); p0=cellpar(l0)
    print(f'{n:8s} {"init":>4s} '+' '.join(f'{x:7.3f}' for x in p0[:6])+f' {p0[6]:8.3f} {0.0:7.2f}')
    for k in (1,2):
        f=os.path.join(d,f'CONTCAR.pass{k}'); o=os.path.join(d,f'OUTCAR.pass{k}')
        if not os.path.exists(f): continue
        l,_=rdpos(f); p=cellpar(l)
        m=moments(o) if os.path.exists(o) else None
        ms=' '.join(f'{x:+.3f}' for x in m) if m else '-'
        print(f'{"":8s} {k:>4d} '+' '.join(f'{x:7.3f}' for x in p[:6])+f' {p[6]:8.3f} {100*(p[6]-p0[6])/p0[6]:+7.2f}  {ms}')
    print()
