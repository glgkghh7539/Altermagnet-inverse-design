#!/usr/bin/env python
"""How much does the S reference energy actually matter?

Shift mu_S by hand (which is exactly what a vdW correction to alpha-S would do)
and watch what moves: formation energies, or energies above hull.
"""
import json, os

# --- paths --------------------------------------------------------------------
# Tables this script reads are deposited in validation/data/. The raw VASP run trees
# are NOT deposited (tens of GB); they default to ~/NiS_hull and can be pointed
# elsewhere with $NIS_HULL_DIR.
HERE    = os.path.dirname(os.path.abspath(__file__))
DEPOT   = os.path.join(HERE, "data")
RUNROOT = os.environ.get("NIS_HULL_DIR", os.path.expanduser("~/NiS_hull"))

rows = json.load(open(os.path.join(DEPOT, "hull_final.json")))
mu_Ni = next(r["e_per_atom"] for r in rows if r["x_S"] == 0.0)
mu_S0 = next(r["e_per_atom"] for r in rows if r["x_S"] == 1.0)


def hull(mu_S):
    out = []
    for r in rows:
        x = r["x_S"]
        out.append(dict(tag=r["tag"], x=x,
                        ef=r["e_per_atom"] - ((1 - x) * mu_Ni + x * mu_S)))
    best = {}
    for r in out:
        k = round(r["x"], 9)
        if k not in best or r["ef"] < best[k]["ef"]:
            best[k] = r
    pts = sorted(best.values(), key=lambda r: r["x"])
    h = []
    for p in pts:
        while len(h) >= 2:
            a, b = h[-2], h[-1]
            if ((b["x"] - a["x"]) * (p["ef"] - a["ef"])
                    - (b["ef"] - a["ef"]) * (p["x"] - a["x"])) <= 0:
                h.pop()
            else:
                break
        h.append(p)

    def hef(x):
        for a, b in zip(h, h[1:]):
            if a["x"] - 1e-9 <= x <= b["x"] + 1e-9:
                t = (x - a["x"]) / (b["x"] - a["x"])
                return a["ef"] + t * (b["ef"] - a["ef"])
        return 0.0
    for r in out:
        r["eh"] = r["ef"] - hef(r["x"])
    return {r["tag"]: r for r in out}, {v["tag"] for v in h}


print("shifting mu_S -- this is exactly what adding a vdW correction to alpha-S does")
print("(a vdW functional binds the S8 crystal more, i.e. pushes mu_S down)\n")
targets = ["local-NiAs_NiS_P6_3mmc", "mp-1547_NiS_R3m", "mp-2282_NiS2_Pa-3"]
print("%9s | %s" % ("d(mu_S)", " | ".join("%-28s" % t.split("_", 1)[1] for t in targets)))
print("%9s | %s" % ("eV/atom", " | ".join("%13s %13s" % ("Ef", "E_hull") for t in targets)))
print("-" * 100)
for d in (-0.30, -0.15, 0.0, +0.15, +0.30):
    res, hs = hull(mu_S0 + d)
    cells = []
    for t in targets:
        r = res[t]
        cells.append("%13.4f %10.1f meV" % (r["ef"], r["eh"] * 1000))
    print("%9.2f | %s" % (d, " | ".join(cells)))
print("-" * 100)
print("Ef moves with the reference. E_hull does not move at all.")

_, h0 = hull(mu_S0)
_, hm = hull(mu_S0 - 0.30)
print("\nstable set at d=0.00 : %s" % ", ".join(sorted(h0)))
print("stable set at d=-0.30: %s" % ", ".join(sorted(hm)))
print("identical: %s" % (h0 == hm))
