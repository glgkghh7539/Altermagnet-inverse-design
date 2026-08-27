import sys,os,glob,warnings,numpy as np
warnings.filterwarnings('ignore')
from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
MAG=set("Sc Ti V Cr Mn Fe Co Ni Cu Zn Y Zr Nb Mo Ru Rh Pd Ag Cd".split())
print(f'{"structure":22s} {"formula":8s} {"V":>7s} {"CN":>7s} {"planarity":>9s} {"M-X(min-max)":>16s} {"d(M-M)":>7s}  SG(0.01)/SG(0.1)')
for f in sorted(glob.glob(os.path.join(sys.argv[1],'POSCAR_*'))):
    st=Structure.from_file(f); mi=[i for i,s in enumerate(st) if str(s.specie) in MAG][:2]
    cnn=CrystalNN(); cns=[];pl=[];bl=[]
    for m in mi:
        nn=cnn.get_nn_info(st,m); lig=[i for i in nn if str(i['site'].specie) not in MAG]
        cns.append(len(lig)); c=st[m].coords
        v=np.array([i['site'].coords for i in lig])-c
        d=np.linalg.norm(v,axis=1); bl.append((d.min(),d.max()))
        u,s_,vt=np.linalg.svd(v-v.mean(0)); pl.append(float(s_[-1]/np.sqrt(len(v))))
    sg=[]
    for sp in (0.01,0.1):
        try: sg.append(SpacegroupAnalyzer(st,symprec=sp).get_space_group_symbol())
        except Exception: sg.append('?')
    n=os.path.basename(f)[7:]
    print(f'{n:22s} {st.composition.reduced_formula:8s} {st.volume:7.2f} {str(cns):>7s} {pl[0]:9.3f} '
          f'{bl[0][0]:7.2f}-{bl[0][1]:<7.2f} {st.get_distance(mi[0],mi[1]):7.3f}  {sg[0]} / {sg[1]}')
