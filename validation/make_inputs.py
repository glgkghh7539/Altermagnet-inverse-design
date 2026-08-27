#!/usr/bin/env python
"""Generate VASP relax+static inputs for every structure in ~/NiS_hull/structures.

Consistent settings across ALL phases (this is what makes E_form / E_hull meaningful):
  PBE, PAW 54 (Ni_pv, S), ENCUT=520, LASPH, spin-polarised, NO Hubbard U.
  relax : ISIF=3, ISMEAR=1/SIGMA=0.2   (well-behaved forces+stress for metals)
  static: NSW=0,  ISMEAR=-5 (tetrahedron), denser k-mesh, EDIFF=1e-6
"""
import os, glob, json, math, sys

# --- paths --------------------------------------------------------------------
# Tables this script reads are deposited in validation/data/. The raw VASP run trees
# are NOT deposited (tens of GB); they default to ~/NiS_hull and can be pointed
# elsewhere with $NIS_HULL_DIR.
HERE    = os.path.dirname(os.path.abspath(__file__))
DEPOT   = os.path.join(HERE, "data")
RUNROOT = os.environ.get("NIS_HULL_DIR", os.path.expanduser("~/NiS_hull"))
from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Poscar, Potcar

ROOT    = RUNROOT
STRUCTS = os.path.join(ROOT, "structures")
RUNS    = os.path.join(ROOT, "runs")

os.environ.setdefault("PMG_VASP_PSP_DIR", os.path.join(HOME, "pmg_potcars"))

ENCUT      = 520
POT_MAP    = {"Ni": "Ni_pv", "S": "S"}
MAGMOM_MAP = {"Ni": 5.0, "S": 0.6}
KSPACING   = {"relax": 0.25, "static": 0.15}

VASP_BIN = {
    "avx512": "/TGM/Apps/VASP/VASP_BIN/6.4.2/AVX512/vasp.6.4.2.avx512.std.x",
    "avx2":   "/TGM/Apps/VASP/VASP_BIN/6.4.2/AVX2/vasp.6.4.2.avx2.std.x",
}


def kmesh(struct, kspacing):
    """Reproduce VASP's KSPACING -> grid, so we can report/inspect it."""
    b = struct.lattice.reciprocal_lattice.abc  # already includes 2*pi
    return [max(1, int(math.ceil(x / kspacing))) for x in b]


def magmom_string(struct):
    return " ".join(f"{MAGMOM_MAP.get(str(s.specie), 0.6):.1f}" for s in struct)


def incar(struct, stage):
    nk = kmesh(struct, KSPACING[stage])
    nkpts = nk[0] * nk[1] * nk[2]
    lines = [
        f"SYSTEM = {struct.composition.reduced_formula} {stage}",
        "",
        "# --- electronic ---",
        f"  ENCUT  = {ENCUT}",
        "  PREC   = Accurate",
        "  EDIFF  = 1E-06",
        "  ALGO   = Normal",
        "  NELM   = 200",
        "  NELMIN = 6",
        "  LASPH  = .TRUE.",
        "  LREAL  = .FALSE.",
        "  ISYM   = 2",
        "",
        "# --- spin (no Hubbard U: PBE only) ---",
        "  ISPIN  = 2",
        f"  MAGMOM = {magmom_string(struct)}",
        "  LDAU   = .FALSE.",
        "",
        "# --- k-points ---",
        f"  KSPACING = {KSPACING[stage]}",
        "  KGAMMA   = .TRUE.",
        f"# -> {nk[0]}x{nk[1]}x{nk[2]} grid ({nkpts} pts before symmetry)",
        "",
    ]
    if stage == "relax":
        lines += [
            "# --- ionic relaxation (full cell) ---",
            "  IBRION = 2",
            "  ISIF   = 3",
            "  NSW    = 120",
            "  EDIFFG = -0.01",
            "  POTIM  = 0.3",
            "",
            "# --- smearing: MP for metals, safe forces/stress ---",
            "  ISMEAR = 1",
            "  SIGMA  = 0.20",
            "",
            "  LCHARG = .FALSE.",
            "  LWAVE  = .FALSE.",
        ]
    else:
        # tetrahedron needs a real 3D mesh; fall back to Gaussian otherwise
        if min(nk) >= 3:
            smear = ["  ISMEAR = -5"]
        else:
            smear = ["  ISMEAR = 0", "  SIGMA  = 0.05",
                     "# (mesh too coarse for tetrahedron)"]
        lines += [
            "# --- static, final total energy ---",
            "  IBRION = -1",
            "  NSW    = 0",
            "",
        ] + smear + [
            "",
            "  LCHARG = .TRUE.",
            "  LWAVE  = .FALSE.",
            "  LORBIT = 11",
        ]
    lines += ["", "# --- parallel ---", "  NCORE = 4", "  LPLANE = .TRUE.", ""]
    return "\n".join(lines)


def slurm(tag, natoms, arch="avx2"):
    """1 node per job so many phases can run concurrently in queue gaps."""
    if arch == "avx512":
        parts, ntasks = "g6,g5", 32
    else:
        parts, ntasks = "g3,g4,g1,g2", 20
    return f"""#!/bin/bash
#SBATCH -J h.{tag[:12]}
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
#SBATCH -p {parts}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node={ntasks}
#SBATCH --time=168:00:00

module purge
module add compiler/2023.1.0
module add mkl/2023.1.0
module add mpi/2021.12.1
module add VASP/basic_tools

VASP={VASP_BIN[arch]}
cd "$SLURM_SUBMIT_DIR"

set -e

# ---------- stage 1: relax (run twice to kill Pulay stress) ----------
cd relax
for pass in 1 2; do
  if [ -f DONE.$pass ]; then continue; fi
  if [ $pass -eq 2 ]; then cp CONTCAR POSCAR; fi
  mpirun -np $SLURM_NTASKS $VASP > result.pass$pass.txt < /dev/null
  grep -q "reached required accuracy\\|writing wavefunctions" OUTCAR || true
  cp CONTCAR CONTCAR.pass$pass
  cp OUTCAR  OUTCAR.pass$pass
  touch DONE.$pass
done
cd ..

# ---------- stage 2: static on the relaxed cell ----------
cd static
cp ../relax/CONTCAR POSCAR
mpirun -np $SLURM_NTASKS $VASP > result.txt < /dev/null
touch DONE
cd ..

echo "ALL DONE {tag}"
"""


def main():
    files = sorted(glob.glob(os.path.join(STRUCTS, "*.vasp")))
    if not files:
        sys.exit(f"no structures in {STRUCTS} -- run get_structures.py first")

    manifest = []
    for f in files:
        tag = os.path.basename(f)[:-5]
        st = Structure.from_file(f)
        # primitive cell keeps the cost down without changing the energy/atom
        st = st.get_primitive_structure()
        st = st.get_sorted_structure()
        natoms = len(st)
        arch = "avx512" if natoms >= 24 else "avx2"

        d = os.path.join(RUNS, tag)
        for stage in ("relax", "static"):
            sd = os.path.join(d, stage)
            os.makedirs(sd, exist_ok=True)
            Poscar(st).write_file(os.path.join(sd, "POSCAR"))
            with open(os.path.join(sd, "INCAR"), "w") as fh:
                fh.write(incar(st, stage))
            syms = [POT_MAP[str(e)] for e in st.composition.elements]
            # order must follow POSCAR species order
            order, seen = [], set()
            for s in st:
                e = str(s.specie)
                if e not in seen:
                    seen.add(e); order.append(POT_MAP[e])
            Potcar(order, functional="PBE_54").write_file(os.path.join(sd, "POTCAR"))

        with open(os.path.join(d, "run.sh"), "w") as fh:
            fh.write(slurm(tag, natoms, arch))
        os.chmod(os.path.join(d, "run.sh"), 0o755)

        rec = dict(tag=tag, natoms=natoms, arch=arch,
                   formula=st.composition.reduced_formula,
                   k_relax=kmesh(st, KSPACING["relax"]),
                   k_static=kmesh(st, KSPACING["static"]))
        manifest.append(rec)
        print(f"{tag:40s} n={natoms:3d} {arch:7s} "
              f"k_relax={rec['k_relax']} k_static={rec['k_static']}")

    with open(os.path.join(ROOT, "run_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"\n{len(manifest)} runs prepared under {RUNS}")


if __name__ == "__main__":
    main()
