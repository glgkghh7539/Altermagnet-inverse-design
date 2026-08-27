#!/usr/bin/env python
import sys, os, csv, glob, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0,os.path.expanduser('~'))
exec(open(os.path.expanduser('~/altermag_sym.py')).read().split("if __name__")[0])

zipdir, listfile, out, symprec = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
names=[l.strip() for l in open(listfile) if l.strip()]
rows=[]; bad=0
for i,n in enumerate(names):
    f=os.path.join(zipdir,'POSCAR_'+n)
    if not os.path.exists(f): rows.append(dict(filename=n,sg='',verdict='structure_missing',ops='',n_ops=0,nsym=0)); bad+=1; continue
    try:
        r=classify(f,symprec)
        rows.append(dict(filename=n, sg=r['sg'], verdict=r['verdict'],
                         ops='|'.join(r.get('ops',[])), n_ops=r.get('n_ops',0), nsym=r.get('nsym',0)))
    except Exception as e:
        rows.append(dict(filename=n,sg='',verdict='ERROR:'+str(e)[:40],ops='',n_ops=0,nsym=0)); bad+=1
    if (i+1)%500==0: print(f'  {i+1}/{len(names)}',flush=True)
with open(out,'w',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=['filename','sg','verdict','ops','n_ops','nsym']); w.writeheader(); w.writerows(rows)
import collections
c=collections.Counter(r['verdict'] for r in rows)
print(f'\n{len(rows)} rows total ({bad} failed)  symprec={symprec}')
for k,v in c.most_common(): print(f'  {k:28s} {v:5d}  ({100*v/len(rows):.1f}%)')
