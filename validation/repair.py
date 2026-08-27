#!/usr/bin/env python
"""Repair the U-scan runs that died or would not converge.

Two failure modes showed up, both rooted in the multi-minimum nature of +U
magnetism: CG could not bracket a minimum while the spin state flipped between
ionic steps (ZBRENT fatal error), and SCF sat at the NELM ceiling so the forces
driving the relaxation were meaningless.

The fix per run:
  * seed POSCAR and MAGMOM from the nearest already-converged U of the same
    phase -- U continuation, so the run starts inside the right magnetic basin
  * IBRION=1 (quasi-Newton) instead of 2 (CG), which tolerates a noisy landscape
  * VASP's documented linear-mixing recipe for slow magnetic systems
"""
import os, re, glob, subprocess, sys

# --- paths --------------------------------------------------------------------
# Tables this script reads are deposited in validation/data/. The raw VASP run trees
# are NOT deposited (tens of GB); they default to ~/NiS_hull and can be pointed
# elsewhere with $NIS_HULL_DIR.
HERE    = os.path.dirname(os.path.abspath(__file__))
DEPOT   = os.path.join(HERE, "data")
RUNROOT = os.environ.get("NIS_HULL_DIR", os.path.expanduser("~/NiS_hull"))

ROOT = os.path.join(RUNROOT, "uscan_hull")
US = [0, 2, 4, 6, 7]

TARGETS = [
    (0, "mp-2282_NiS2_Pa-3"),
    (4, "mp-1050_Ni3S4_Fd-3m"),
    (6, "mp-1050_Ni3S4_Fd-3m"),
    (6, "mp-362_Ni3S2_R32"),
    (6, "mp-976920_Ni9S8_C222"),
    (7, "mp-1547_NiS_R3m"),
    (7, "mp-362_Ni3S2_R32"),
    (7, "mp-2282_NiS2_Pa-3"),
    (7, "mp-976920_Ni9S8_C222"),
]

MIX = ["  AMIX     = 0.2", "  BMIX     = 0.00001",
       "  AMIX_MAG = 0.8", "  BMIX_MAG = 0.00001"]


def done(u, ph):
    return os.path.exists(os.path.join(ROOT, "U%d" % u, ph, "static", "DONE"))


def moments(outcar, nions):
    """per-ion total moments from the last magnetisation table"""
    blk, lines = None, open(outcar, errors="ignore").read().splitlines()
    for i, ln in enumerate(lines):
        if "magnetization (x)" in ln:
            blk = i
    if blk is None:
        return None
    out = []
    for ln in lines[blk + 4:]:
        s = ln.split()
        if len(s) == 5 and s[0].isdigit():
            out.append(s[-1])
        else:
            break
    return out if len(out) == nions else None


def nions_of(poscar):
    ls = open(poscar).read().splitlines()
    return sum(int(x) for x in ls[6].split())


for u, ph in TARGETS:
    D = os.path.join(ROOT, "U%d" % u, ph)
    # nearest converged U for this phase, preferring a lower one
    cands = sorted((abs(v - u), 0 if v < u else 1, v) for v in US
                   if v != u and done(v, ph))
    if not cands:
        print("SKIP %-4s %-26s no converged neighbour to seed from" % ("U%d" % u, ph))
        continue
    src_u = cands[0][2]
    S = os.path.join(ROOT, "U%d" % src_u, ph)

    subprocess.run(["cp", os.path.join(S, "relax", "CONTCAR"),
                    os.path.join(D, "relax", "POSCAR")], check=True)
    n = nions_of(os.path.join(D, "relax", "POSCAR"))
    mom = moments(os.path.join(S, "static", "OUTCAR"), n)
    magline = "  MAGMOM = " + " ".join(mom) if mom else None

    for stage in ("relax", "static"):
        p = os.path.join(D, stage, "INCAR")
        txt = open(p).read().splitlines()
        out, seen_mix = [], False
        for ln in txt:
            if ln.startswith("  IBRION = 2"):
                ln = "  IBRION = 1"
            if magline and ln.startswith("  MAGMOM"):
                ln = magline
            if ln.startswith("  AMIX") or ln.startswith("  BMIX"):
                seen_mix = True
                continue
            out.append(ln)
        if not seen_mix:
            out += ["", "# --- mixing: VASP recipe for slowly converging magnetic systems ---"] + MIX
        open(p, "w").write("\n".join(out) + "\n")

    for f in ("relax/DONE.1", "relax/DONE.2", "static/DONE"):
        try:
            os.remove(os.path.join(D, f))
        except OSError:
            pass

    print("U%-3d %-26s  seed=U%-2d  MAGMOM=%s  IBRION=1  +mixing"
          % (u, ph, src_u, "seeded(%d)" % n if mom else "unchanged"))
