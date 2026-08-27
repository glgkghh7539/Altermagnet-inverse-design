#!/usr/bin/env python
"""Ni-S formation energy + convex hull straight from the run OUTCARs.

Two hulls are reported side by side:
  raw      -- plain PBE total energies, what we actually computed
  MP2020   -- with the Materials Project 2020 sulfide anion correction applied,
              which is what makes E_form directly comparable to MP numbers

A run without static/DONE falls back to its latest relax energy and is flagged
PROV, so the table is usable while the last phase is still relaxing.
"""
import os, glob, sys, json

# --- paths --------------------------------------------------------------------
# Tables this script reads are deposited in validation/data/. The raw VASP run trees
# are NOT deposited (tens of GB); they default to ~/NiS_hull and can be pointed
# elsewhere with $NIS_HULL_DIR.
HERE    = os.path.dirname(os.path.abspath(__file__))
DEPOT   = os.path.join(HERE, "data")
RUNROOT = os.environ.get("NIS_HULL_DIR", os.path.expanduser("~/NiS_hull"))

RUNS = os.path.join(RUNROOT, "runs")
MPREF = os.path.join(DEPOT, "mp_reference.json")


def last_e(outcar):
    e = None
    with open(outcar, errors="ignore") as f:
        for ln in f:
            if "energy(sigma->0)" in ln:
                e = float(ln.split()[-1])
    return e


def counts(outcar):
    """element symbols and per-species ion counts, read off the OUTCAR header"""
    syms, nions = [], []
    with open(outcar, errors="ignore") as f:
        for ln in f:
            if ln.startswith("   VRHFIN ="):
                syms.append(ln.split("=")[1].split(":")[0].strip())
            elif "ions per type" in ln:
                nions = [int(x) for x in ln.split("=")[1].split()]
                break
    return syms, nions


# MP2020 anion correction for S, pulled from pymatgen itself so it tracks the
# installed version rather than a number copied into this file.
def mp2020_s_correction():
    try:
        from pymatgen.core import Composition
        from pymatgen.entries.computed_entries import ComputedEntry
        from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
        e = ComputedEntry(Composition("NiS2"), -10.0,
                          parameters={"run_type": "GGA", "is_hubbard": False,
                                      "hubbards": {},
                                      "potcar_symbols": ["PAW_PBE Ni_pv 54",
                                                         "PAW_PBE S 54"]})
        tot = sum(a.value for a in MaterialsProject2020Compatibility().get_adjustments(e))
        return tot / 2.0
    except Exception as exc:
        print("  (could not read MP2020 correction from pymatgen: %s)" % exc)
        return None


rows = []
for d in sorted(glob.glob(os.path.join(RUNS, "*"))):
    tag = os.path.basename(d)
    st, rl = os.path.join(d, "static"), os.path.join(d, "relax")
    if os.path.exists(os.path.join(st, "DONE")) and os.path.exists(os.path.join(st, "OUTCAR")):
        oc, prov = os.path.join(st, "OUTCAR"), False
    elif os.path.exists(os.path.join(rl, "OUTCAR")):
        oc, prov = os.path.join(rl, "OUTCAR"), True
    else:
        print("  skip %s: no OUTCAR" % tag)
        continue
    e = last_e(oc)
    syms, nions = counts(oc)
    if e is None or not nions:
        print("  skip %s: no energy yet" % tag)
        continue
    n = dict(zip(syms, nions))
    nat = sum(nions)
    rows.append(dict(tag=tag, e=e, nat=nat, nNi=n.get("Ni", 0), nS=n.get("S", 0),
                     epa=e / nat, xS=n.get("S", 0) / nat, prov=prov))

if not rows:
    sys.exit("no runs with energies yet")

DS = mp2020_s_correction()


def hull_for(key):
    """attach ef_<key> to every row and return (hull points, mu_Ni, mu_S)"""
    ref = {}
    for r in rows:
        if r["xS"] in (0.0, 1.0):
            k = "Ni" if r["xS"] == 0.0 else "S"
            if k not in ref or r[key] < ref[k][key]:
                ref[k] = r
    missing = [k for k in ("Ni", "S") if k not in ref]
    if missing:
        sys.exit("cannot form hull: missing elemental reference for %s" % missing)
    mu_Ni, mu_S = ref["Ni"][key], ref["S"][key]
    for r in rows:
        r["ef_" + key] = r[key] - ((1 - r["xS"]) * mu_Ni + r["xS"] * mu_S)

    best = {}
    for r in rows:
        x = round(r["xS"], 9)
        if x not in best or r["ef_" + key] < best[x]["ef_" + key]:
            best[x] = r
    pts = sorted(best.values(), key=lambda r: r["xS"])
    hull = []
    for p in pts:                       # monotone chain, lower hull
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            cross = ((b["xS"] - a["xS"]) * (p["ef_" + key] - a["ef_" + key])
                     - (b["ef_" + key] - a["ef_" + key]) * (p["xS"] - a["xS"]))
            if cross <= 0:              # b sits on/above the chord a-p, drop it
                hull.pop()
            else:
                break
        hull.append(p)

    def hef(x):
        for a, b in zip(hull, hull[1:]):
            if a["xS"] - 1e-9 <= x <= b["xS"] + 1e-9:
                t = (x - a["xS"]) / (b["xS"] - a["xS"])
                return a["ef_" + key] + t * (b["ef_" + key] - a["ef_" + key])
        return 0.0

    for r in rows:
        r["eh_" + key] = r["ef_" + key] - hef(r["xS"])
    return set(h["tag"] for h in hull), mu_Ni, mu_S


# raw energies per atom
for r in rows:
    r["raw"] = r["epa"]
hull_raw, muNi_raw, muS_raw = hull_for("raw")

# MP2020-corrected: the S anion correction applies only where S is the anion,
# i.e. in the Ni-S compounds, never to elemental S or elemental Ni.
if DS is not None:
    for r in rows:
        corr = DS * r["nS"] if (r["nNi"] > 0 and r["nS"] > 0) else 0.0
        r["mp20"] = (r["e"] + corr) / r["nat"]
    hull_mp, muNi_mp, muS_mp = hull_for("mp20")

# MP reference for cross-check
mpref = {}
if os.path.exists(MPREF):
    d = json.load(open(MPREF))
    for e in (d["all"] if isinstance(d, dict) else d):
        mpref[e["material_id"]] = e

prov_any = any(r["prov"] for r in rows)
W = 108
print("=" * W)
print(" Ni-S FORMATION ENERGY / CONVEX HULL   (PBE, no U, ENCUT=520, PAW54 Ni_pv/S)")
if prov_any:
    print(" *** PROVISIONAL: rows flagged PROV are still relaxing ***")
print("=" * W)
print("%-30s %6s | %9s %8s | %9s %8s | %9s %8s" %
      ("phase", "x_S", "Ef_raw", "Eh_raw", "Ef_MP20", "Eh_MP20", "Ef_MP", "Eh_MP"))
print("%-30s %6s | %9s %8s | %9s %8s | %9s %8s" %
      ("", "", "eV/at", "meV/at", "eV/at", "meV/at", "eV/at", "meV/at"))
print("-" * W)
for r in sorted(rows, key=lambda r: (r["xS"], r["ef_raw"])):
    mp = mpref.get(r["tag"].split("_")[0])
    mp_ef = "%9.4f" % mp["e_form_mp"] if mp else "%9s" % "-"
    mp_eh = "%8.1f" % (mp["e_above_hull"] * 1000) if mp else "%8s" % "-"
    if DS is not None:
        c_ef, c_eh = "%9.4f" % r["ef_mp20"], "%8.1f" % (r["eh_mp20"] * 1000)
        star = "*" if r["tag"] in hull_mp else " "
    else:
        c_ef = c_eh = "%9s" % "-"
        star = " "
    flag = "PROV" if r["prov"] else ""
    print("%-30s %6.3f | %9.4f %8.1f | %s %s%s| %s %s %s" %
          (r["tag"], r["xS"], r["ef_raw"], r["eh_raw"] * 1000,
           c_ef, c_eh, star, mp_ef, mp_eh, flag))
print("-" * W)
print(" Ef_raw/Eh_raw : our plain-PBE hull")
if DS is not None:
    print(" Ef_MP20/Eh_MP20: same energies + MP2020 sulfide anion correction "
          "(%+.3f eV per S atom); * = on the MP2020 hull" % DS)
print(" Ef_MP/Eh_MP   : Materials Project values, for cross-check")
print(" references: mu_Ni = %.5f eV/atom, mu_S = %.5f eV/atom (raw)" % (muNi_raw, muS_raw))

# how well do we reproduce MP?
diffs = [(r["tag"], r["ef_mp20"] - mpref[r["tag"].split("_")[0]]["e_form_mp"])
         for r in rows
         if DS is not None and r["tag"].split("_")[0] in mpref and 0 < r["xS"] < 1]
if diffs:
    print("\n Ef_MP20 - Ef_MP  (agreement with Materials Project):")
    for t, dv in sorted(diffs, key=lambda x: x[1]):
        print("   %-30s %+7.1f meV/atom" % (t, dv * 1000))
    a = sum(abs(dv) for _, dv in diffs) / len(diffs)
    print("   mean |deviation| = %.1f meV/atom over %d compounds" % (a * 1000, len(diffs)))


# ---- convex hull figure (MP2020-corrected, the MP-comparable one) ----
if DS is not None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7.2, 5.0))
        hx = sorted([(r["xS"], r["ef_mp20"]) for r in rows if r["tag"] in hull_mp])
        ax.plot([p[0] for p in hx], [p[1] for p in hx], "-", color="0.25",
                lw=1.6, zorder=2, label="convex hull")

        for r in sorted(rows, key=lambda r: r["xS"]):
            on = r["tag"] in hull_mp
            ax.scatter(r["xS"], r["ef_mp20"],
                       s=78 if on else 46,
                       marker="o" if on else "^",
                       facecolor="#1f77b4" if on else "white",
                       edgecolor="#1f77b4" if on else "#d62728",
                       linewidths=1.4, zorder=3)
            # label: formula, offset so the two x_S=0.5 points do not collide
            name = r["tag"].split("_", 1)[1] if "_" in r["tag"] else r["tag"]
            # keep the two elemental labels inside the axes instead of
            # letting them run off the left/right edge
            if r["xS"] < 1e-9:
                ha, off = "left", (7, -13)
            elif r["xS"] > 1 - 1e-9:
                ha, off = "right", (-7, -13)
            else:
                ha, off = "center", (0, -15 if on else 9)
            ax.annotate(name, (r["xS"], r["ef_mp20"]),
                        textcoords="offset points", xytext=off,
                        ha=ha, fontsize=7.2,
                        color="#1f77b4" if on else "#d62728")

        ax.axhline(0, color="0.75", lw=0.8, zorder=1)
        ax.set_xlabel("$x_{\\mathrm{S}}$  (S fraction)")
        ax.set_ylabel("formation energy  (eV/atom)")
        ax.set_title("Ni-S convex hull, PBE + MP2020 sulfide correction")
        ax.set_xlim(-0.05, 1.05)
        lo = min(r["ef_mp20"] for r in rows)
        ax.set_ylim(lo - 0.10, 0.045)      # room for the labels under the minimum
        ax.grid(alpha=0.25, lw=0.6)
        from matplotlib.lines import Line2D
        ax.legend(handles=[
            Line2D([], [], color="0.25", lw=1.6, label="hull"),
            Line2D([], [], marker="o", ls="", mfc="#1f77b4", mec="#1f77b4",
                   label="stable"),
            Line2D([], [], marker="^", ls="", mfc="white", mec="#d62728",
                   label="metastable")], loc="lower left", fontsize=8)
        out = os.path.join(RUNROOT, "hull_NiS_MP2020.png")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print("\n hull figure -> %s" % out)
    except Exception as exc:
        print("\n plot skipped: %s" % exc)

# ---- decomposition of every off-hull phase, on the MP2020 hull ----
if DS is not None:
    print("\n== decomposition of metastable phases (MP2020 hull) ==")
    hl = sorted([r for r in rows if r["tag"] in hull_mp], key=lambda r: r["xS"])
    for r in sorted(rows, key=lambda r: r["eh_mp20"]):
        if r["eh_mp20"] < 1e-4:
            continue
        left = max((h for h in hl if h["xS"] <= r["xS"]), key=lambda h: h["xS"])
        right = min((h for h in hl if h["xS"] >= r["xS"]), key=lambda h: h["xS"])
        lf = left["tag"].split("_", 1)[1].split("_")[0]
        rf = right["tag"].split("_", 1)[1].split("_")[0]
        if abs(right["xS"] - left["xS"]) < 1e-9:
            # a stable polymorph sits at this exact composition, so the
            # "decomposition" is just the transformation into it
            print("  %-30s %7.1f meV/atom  ->  1.000 %s  (same composition: %s)"
                  % (r["tag"], r["eh_mp20"] * 1000, lf, left["tag"]))
            continue
        t = (r["xS"] - left["xS"]) / (right["xS"] - left["xS"])
        print("  %-30s %7.1f meV/atom  ->  %.3f %s + %.3f %s"
              % (r["tag"], r["eh_mp20"] * 1000, 1 - t, lf, t, rf))

# ---- machine-readable dump ----
out = [dict(tag=r["tag"], x_S=r["xS"], natoms=r["nat"],
            e_total=r["e"], e_per_atom=r["epa"],
            ef_raw=r["ef_raw"], e_hull_raw=r["eh_raw"],
            ef_mp2020=r.get("ef_mp20"), e_hull_mp2020=r.get("eh_mp20"),
            on_hull_mp2020=(DS is not None and r["tag"] in hull_mp),
            provisional=r["prov"]) for r in sorted(rows, key=lambda r: r["xS"])]
jp = os.path.join(RUNROOT, "hull_final.json")
json.dump(out, open(jp, "w"), indent=1)
print("\n data -> %s" % jp)
