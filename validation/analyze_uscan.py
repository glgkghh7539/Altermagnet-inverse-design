#!/usr/bin/env python
"""U dependence of the NiAs-type NiS decomposition energy.

Only Ni sulfides are involved. The quantity of interest is the vertical
distance from NiAs-type NiS to the tie-line joining its neighbours in
(x_S, E/atom) space. Subtracting any linear function of x_S leaves a convex
hull unchanged, so the elemental Ni and S references cancel exactly and never
need a U value -- which is what makes this scan well posed.

At U=0 the number printed here must reproduce the 106.8 meV/atom from the full
nine-phase hull; that is the built-in check.
"""
import os, glob, json, sys

# --- paths --------------------------------------------------------------------
# Tables this script reads are deposited in validation/data/. The raw VASP run trees
# are NOT deposited (tens of GB); they default to ~/NiS_hull and can be pointed
# elsewhere with $NIS_HULL_DIR.
HERE    = os.path.dirname(os.path.abspath(__file__))
DEPOT   = os.path.join(HERE, "data")
RUNROOT = os.environ.get("NIS_HULL_DIR", os.path.expanduser("~/NiS_hull"))

ROOT = os.path.join(RUNROOT, "uscan_hull")
US = [0, 2, 4, 6, 7]
TARGET_FM = "local-NiAs_NiS_P6_3mmc"
TARGET_AFM = "local-NiAs_NiS_P6_3mmc_AFM"


def outcar_read(path):
    e = vol = None
    syms, nions, mom, cur = [], [], [], None
    with open(path, errors="ignore") as f:
        for ln in f:
            if ln.startswith("   VRHFIN ="):
                syms.append(ln.split("=")[1].split(":")[0].strip())
            elif "ions per type" in ln and not nions:
                nions = [int(x) for x in ln.split("=")[1].split()]
            elif "energy(sigma->0)" in ln:
                e = float(ln.split()[-1])
            elif "volume of cell" in ln:
                vol = float(ln.split()[-1])
            elif "magnetization (x)" in ln:
                cur = []
            elif cur is not None:
                s = ln.split()
                if len(s) == 5 and s[0].isdigit():
                    cur.append(float(s[-1]))
                elif s and s[0] == "tot":
                    mom, cur = cur, None
    return e, sum(nions), dict(zip(syms, nions)), vol, mom


def incar_val(path, key):
    try:
        for ln in open(path):
            p = ln.split("=")
            if len(p) == 2 and p[0].strip() == key:
                return p[1].split("#")[0].strip()
    except OSError:
        pass
    return "?"


def lower_hull(pts):
    pts = sorted(pts)
    h = []
    for p in pts:
        while len(h) >= 2:
            a, b = h[-2], h[-1]
            if (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) <= 0:
                h.pop()
            else:
                break
        h.append(p)
    return h


def collect(u):
    rows = {}
    for d in sorted(glob.glob(os.path.join(ROOT, "U%d" % u, "*"))):
        tag = os.path.basename(d)
        if not os.path.exists(os.path.join(d, "static", "DONE")):
            continue
        oc = os.path.join(d, "static", "OUTCAR")
        e, nat, n, vol, mom = outcar_read(oc)
        if e is None or not nat:
            continue
        nNi = n.get("Ni", 0)
        rows[tag] = dict(
            tag=tag, e=e, nat=nat, epa=e / nat, xS=n.get("S", 0) / nat,
            vpa=vol / nat if vol else 0.0,
            mNi=max((abs(x) for x in mom[:nNi]), default=0.0) if mom else 0.0,
            isym=incar_val(os.path.join(d, "static", "INCAR"), "ISYM"))
    return rows


def dE(rows, target):
    """vertical drop from `target` to the tie-line of the other sulfides"""
    if target not in rows:
        return None
    others = [(r["xS"], r["epa"], t) for t, r in rows.items()
              if t not in (TARGET_FM, TARGET_AFM)]
    if len(others) < 2:
        return None
    h = lower_hull(others)
    x = rows[target]["xS"]
    for a, b in zip(h, h[1:]):
        if a[0] - 1e-9 <= x <= b[0] + 1e-9:
            f = (x - a[0]) / (b[0] - a[0])
            ref = a[1] + f * (b[1] - a[1])
            nm = lambda t: t.split("_", 1)[1].split("_")[0]
            if f < 1e-6:
                prod = nm(a[2])
            elif f > 1 - 1e-6:
                prod = nm(b[2])
            else:
                prod = "%.3f %s + %.3f %s" % (1 - f, nm(a[2]), f, nm(b[2]))
            return (rows[target]["epa"] - ref) * 1000, prod
    return None


ALL = {u: collect(u) for u in US}
missing = [(u, 8 - len(ALL[u])) for u in US if len(ALL[u]) < 8]

W = 104
print("=" * W)
print(" NiAs-type NiS: decomposition energy vs Hubbard U   (sulfides only; references cancel)")
print("=" * W)
if missing:
    print(" INCOMPLETE: " + ", ".join("U=%d missing %d" % m for m in missing))
    print("-" * W)
print("%3s | %-24s | %9s %8s %7s | %s"
      % ("U", "variant", "dE_dec", "V/atom", "m(Ni)", "decomposition products"))
print("%3s | %-24s | %9s %8s %7s |" % ("", "", "meV/at", "A^3", "muB"))
print("-" * W)

summary = []
for u in US:
    rows = ALL[u]
    if not rows:
        print("%3d | (nothing finished)" % u)
        continue
    for target, label in ((TARGET_FM, "NiAs FM  (all-FM set)"),
                          (TARGET_AFM, "NiAs AFM (products FM)")):
        r = dE(rows, target)
        if r is None:
            print("%3d | %-24s | %9s" % (u, label, "pending"))
            continue
        de, prod = r
        t = rows[target]
        print("%3d | %-24s | %9.1f %8.2f %7.3f | %s"
              % (u, label, de, t["vpa"], t["mNi"], prod))
        summary.append(dict(U=u, variant=label, dE_meV=de, vol_per_atom=t["vpa"],
                            m_Ni=t["mNi"], products=prod))
    print("-" * W)

print()
print("== per-phase detail ==")
print("%3s | %-30s | %10s %8s %7s %6s %5s"
      % ("U", "phase", "E/atom", "V/atom", "m(Ni)", "x_S", "ISYM"))
print("-" * 84)
isyms = set()
for u in US:
    for t, r in sorted(ALL[u].items(), key=lambda kv: kv[1]["xS"]):
        print("%3d | %-30s | %10.5f %8.2f %7.3f %6.3f %5s"
              % (u, t, r["epa"], r["vpa"], r["mNi"], r["xS"], r["isym"]))
        if not t.endswith("_AFM"):
            isyms.add(r["isym"])
    if ALL[u]:
        print("-" * 84)
if len(isyms) > 1:
    print(" WARNING: the FM runs do not all share one ISYM setting: %s" % sorted(isyms))

json.dump(summary, open(os.path.join(RUNROOT, "uscan_hull_summary.json"), "w"),
          indent=1)

# ---- figure: decomposition energy, moment and volume vs U ----
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))
    for label, mark, col in (("NiAs FM  (all-FM set)", "o-", "#1f77b4"),
                             ("NiAs AFM (products FM)", "s-", "#d62728")):
        pts = [(s["U"], s["dE_meV"], s["m_Ni"], s["vol_per_atom"])
               for s in summary if s["variant"] == label]
        if not pts:
            continue
        u, d, m, v = zip(*sorted(pts))
        short = label.split()[1]
        ax[0].plot(u, d, mark, color=col, label=short)
        ax[1].plot(u, m, mark, color=col, label=short)
        ax[2].plot(u, v, mark, color=col, label=short)

    ax[0].axhline(106.8, ls="--", lw=1, color="0.4")
    ax[0].annotate("no-U hull: 106.8", (0.05, 106.8), fontsize=8, color="0.35",
                   va="bottom")
    ax[0].set_ylabel("decomposition energy  (meV/atom)")
    ax[1].set_ylabel(r"local moment on Ni  ($\mu_B$)")
    ax[2].set_ylabel(r"volume per atom  ($\AA^3$)")
    for a in ax:
        a.set_xlabel("Hubbard $U$ on Ni $d$  (eV)")
        a.grid(alpha=0.25, lw=0.6)
        a.legend(fontsize=8)
    fig.suptitle("NiAs-type NiS vs Hubbard U  (Ni-S sulfides only)", y=1.02)
    out = os.path.join(RUNROOT, "uscan_NiAs_vs_U.png")
    fig.savefig(out, dpi=190, bbox_inches="tight")
    print("\nfigure -> %s" % out)
except Exception as exc:
    print("\nplot skipped: %s" % exc)
