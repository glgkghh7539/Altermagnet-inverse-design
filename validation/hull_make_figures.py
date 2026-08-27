#!/usr/bin/env python
"""Figures for the Ni-S formation-energy work.

Reads the finished data in ~/tgm-work/hull and writes three PNGs next to it:
  fig1_convex_hull.png    the hull itself
  fig2_mp_validation.png  our numbers against Materials Project
  fig3_U_dependence.png   what happens when a Hubbard U is switched on
"""
import json, os, re

# --- paths --------------------------------------------------------------------
# Tables this script reads are deposited in validation/data/. The raw VASP run trees
# are NOT deposited (tens of GB); they default to ~/NiS_hull and can be pointed
# elsewhere with $NIS_HULL_DIR.
HERE    = os.path.dirname(os.path.abspath(__file__))
DEPOT   = os.path.join(HERE, "data")
RUNROOT = os.environ.get("NIS_HULL_DIR", os.path.expanduser("~/NiS_hull"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


BLUE, RED, GREY = "#1f77b4", "#d62728", "#5a5a5a"
NIAS = "local-NiAs_NiS_P6_3mmc"
# our NiAs-type cell is the same phase MP calls mp-594
ALIAS = {NIAS: "mp-594"}

rows = json.load(open(os.path.join(DEPOT, "hull_final.json")))
mpref = {e["material_id"]: e
         for e in json.load(open(os.path.join(DEPOT, "mp_reference.json")))["all"]}


def label(tag):
    """Ni3S2_R32 -> Ni3S2 (R32); keeps the NiAs type distinguishable"""
    rest = tag.split("_", 1)[1] if "_" in tag else tag
    f, _, spg = rest.partition("_")
    return "%s (%s)" % (f, spg.replace("_", ""))


# ----------------------------------------------------------------- figure 1
fig, ax = plt.subplots(figsize=(7.4, 5.2))
hull = sorted([(r["x_S"], r["ef_mp2020"]) for r in rows if r["on_hull_mp2020"]])
ax.plot([p[0] for p in hull], [p[1] for p in hull], "-", color=GREY, lw=1.7, zorder=2)

for r in sorted(rows, key=lambda r: r["x_S"]):
    on = r["on_hull_mp2020"]
    star = r["tag"] == NIAS
    ax.scatter(r["x_S"], r["ef_mp2020"],
               s=150 if star else (85 if on else 55),
               marker="*" if star else ("o" if on else "^"),
               facecolor=("#f0a202" if star else (BLUE if on else "white")),
               edgecolor=("#8a5b00" if star else (BLUE if on else RED)),
               linewidths=1.5, zorder=4 if star else 3)
    if r["x_S"] < 1e-9:
        ha, off = "left", (8, -13)
    elif r["x_S"] > 1 - 1e-9:
        ha, off = "right", (-8, -13)
    else:
        ha, off = "center", (0, -17 if on else 10)
    txt = label(r["tag"])
    if star:
        txt += "\n107 meV/atom above hull"
        off = (0, 12)
    ax.annotate(txt, (r["x_S"], r["ef_mp2020"]), textcoords="offset points",
                xytext=off, ha=ha, fontsize=7.4,
                color=("#8a5b00" if star else (BLUE if on else RED)),
                fontweight="bold" if star else "normal")

ax.axhline(0, color="0.8", lw=0.8, zorder=1)
lo = min(r["ef_mp2020"] for r in rows)
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(lo - 0.11, 0.055)
ax.set_xlabel(r"$x_{\mathrm{S}}$   (S fraction)")
ax.set_ylabel("formation energy  (eV/atom)")
ax.set_title("Ni–S convex hull   ·   PBE, no $U$, + MP2020 sulfide correction")
ax.grid(alpha=0.22, lw=0.6)
ax.legend(handles=[Line2D([], [], color=GREY, lw=1.7, label="convex hull"),
                   Line2D([], [], marker="o", ls="", mfc=BLUE, mec=BLUE, label="stable"),
                   Line2D([], [], marker="^", ls="", mfc="white", mec=RED, label="metastable"),
                   Line2D([], [], marker="*", ls="", ms=12, mfc="#f0a202",
                          mec="#8a5b00", label="NiAs-type NiS (altermagnet)")],
          loc="lower left", fontsize=8)
fig.savefig(os.path.join(HERE, "fig1_convex_hull.png"), dpi=200, bbox_inches="tight")
print("fig1_convex_hull.png")

# ----------------------------------------------------------------- figure 2
pairs = []
for r in rows:
    mpid = ALIAS.get(r["tag"], r["tag"].split("_")[0])
    m = mpref.get(mpid)
    if m and 0 < r["x_S"] < 1:
        pairs.append((r, m))

fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.4))
xs = [m["e_form_mp"] for _, m in pairs]
ys = [r["ef_mp2020"] for r, _ in pairs]
lim = [min(xs + ys) - 0.025, max(xs + ys) + 0.035]
ax[0].plot(lim, lim, "--", color="0.65", lw=1)
ax[0].scatter(xs, ys, s=64, facecolor=BLUE, edgecolor="white", linewidths=1, zorder=3)
# Ni9S8 and millerite sit within 1 meV of each other, so stagger any labels
# whose points nearly coincide instead of letting the text pile up
span = lim[1] - lim[0]
placed = []
for (r, m), x, y in zip(pairs, xs, ys):
    off = (7, -4)
    for px, py in placed:
        if abs(x - px) < 0.02 * span and abs(y - py) < 0.05 * span:
            off = (7, -16)
            break
    placed.append((x, y))
    ax[0].annotate(label(r["tag"]), (x, y), textcoords="offset points",
                   xytext=off, fontsize=6.8, color=GREY)
ax[0].set_xlim(lim); ax[0].set_ylim(lim)
ax[0].set_xlabel("Materials Project  $E_f$  (eV/atom)")
ax[0].set_ylabel("this work, MP2020-corrected  (eV/atom)")
ax[0].set_title("formation energy, %d phases" % len(pairs))
ax[0].locator_params(axis="x", nbins=5)
ax[0].locator_params(axis="y", nbins=6)
ax[0].grid(alpha=0.22, lw=0.6)

dev = [(r["x_S"], (r["ef_mp2020"] - m["e_form_mp"]) * 1000) for r, m in pairs]
mad = sum(abs(d) for _, d in dev) / len(dev)
ax[1].axhline(0, color="0.65", lw=1, ls="--")
ax[1].axhspan(-mad, mad, color=BLUE, alpha=0.10)
ax[1].scatter(*zip(*dev), s=64, facecolor=BLUE, edgecolor="white", linewidths=1, zorder=3)
for (r, _), (x, d) in zip(pairs, dev):
    ax[1].annotate(label(r["tag"]), (x, d), textcoords="offset points",
                   xytext=(7, -4), fontsize=6.8, color=GREY)
ax[1].set_xlim(0.36, 0.74)
ax[1].set_xlabel(r"$x_{\mathrm{S}}$")
ax[1].set_ylabel("this work $-$ MP   (meV/atom)")
ax[1].set_title("deviation:  mean $|\\Delta|$ = %.1f meV/atom" % mad)
ax[1].grid(alpha=0.22, lw=0.6)
fig.suptitle("Validation against Materials Project   ·   raw PBE differs by exactly "
             "the MP2020 sulfide correction, $-0.503$ eV per S", y=1.02, fontsize=11)
fig.savefig(os.path.join(HERE, "fig2_mp_validation.png"), dpi=200, bbox_inches="tight")
print("fig2_mp_validation.png  (mean |dev| = %.1f meV/atom)" % mad)

# ----------------------------------------------------------------- figure 3
summ = json.load(open(os.path.join(DEPOT, "uscan_hull_summary.json")))
# per-phase energies for the millerite / NiAs ordering, straight from the report
per = {}
for ln in open(os.path.join(DEPOT, "uscan_report.txt")):
    m = re.match(r"\s*(\d+) \| (\S+)\s+\|\s+(-?\d+\.\d+)\s+(\d+\.\d+)", ln)
    if m:
        per[(int(m.group(1)), m.group(2))] = (float(m.group(3)), float(m.group(4)))

US = sorted({s["U"] for s in summ})
fig, ax = plt.subplots(1, 3, figsize=(14.4, 4.2))

for key, mark, col, name in ((
        "NiAs FM  (all-FM set)", "o-", BLUE, "NiAs FM"),
        ("NiAs AFM (products FM)", "s-", RED, "NiAs AFM")):
    pts = sorted((s["U"], s["dE_meV"]) for s in summ if s["variant"] == key)
    ax[0].plot(*zip(*pts), mark, color=col, label=name, lw=1.6, ms=6)
ax[0].axhline(0, color="0.55", lw=1)
ax[0].axhspan(-260, 0, color=RED, alpha=0.07)
ax[0].annotate("below 0 = NiAs type predicted stable\nagainst decomposition (unphysical)",
               (3.4, -150), fontsize=7.4, color=RED, ha="center")
ax[0].scatter([0], [106.8], s=150, marker="*", facecolor="#f0a202",
              edgecolor="#8a5b00", zorder=5)
ax[0].annotate("107", (0, 106.8), textcoords="offset points", xytext=(12, 2),
               fontsize=8, fontweight="bold", color="#8a5b00")
ax[0].set_ylabel("decomposition energy  (meV/atom)")
ax[0].set_title("(a)  distance above the tie-line")

dd = [(u, (per[(u, "local-NiAs_NiS_P6_3mmc_AFM")][0]
           - per[(u, "mp-1547_NiS_R3m")][0]) * 1000) for u in US
      if (u, "mp-1547_NiS_R3m") in per]
ax[1].axhline(0, color="0.55", lw=1)
ax[1].axhspan(-260, 0, color=RED, alpha=0.07)
ax[1].plot(*zip(*dd), "D-", color="#2b7a4b", lw=1.6, ms=6, label="NiAs AFM $-$ millerite")
ax[1].annotate("millerite lower\n(matches experiment)", (3.5, 55), fontsize=7.6,
               color="#2b7a4b", ha="center")
ax[1].annotate("NiAs type lower\n(contradicts experiment)", (3.5, -170),
               fontsize=7.6, color=RED, ha="center")
ax[1].set_ylabel(r"$\Delta E$   (meV/atom)")
ax[1].set_title("(b)  polymorph ordering")

for key, mark, col, name in ((
        "NiAs FM  (all-FM set)", "o-", BLUE, "NiAs FM"),
        ("NiAs AFM (products FM)", "s-", RED, "NiAs AFM")):
    pts = sorted((s["U"], s["vol_per_atom"]) for s in summ if s["variant"] == key)
    ax[2].plot(*zip(*pts), mark, color=col, label=name, lw=1.6, ms=6)
ax[2].axhline(13.71, ls="--", lw=1.2, color="#2b7a4b")
ax[2].annotate(r"experiment, $\alpha$-NiS  13.7 $\AA^3$", (7.0, 13.71),
               textcoords="offset points", xytext=(-4, 7), fontsize=7.6,
               ha="right", color="#2b7a4b")
ax[2].set_ylabel(r"volume per atom  ($\AA^3$)")
ax[2].set_title("(c)  cell volume")

for a in ax:
    a.set_xlabel("Hubbard $U$ on Ni $d$   (eV)")
    a.grid(alpha=0.22, lw=0.6)
    a.legend(fontsize=8, loc="best")
fig.subplots_adjust(wspace=0.30)
fig.suptitle("Switching on a Hubbard $U$ breaks the thermodynamics   ·   "
             "only $U=0$ reproduces experiment", y=1.03, fontsize=11)
fig.savefig(os.path.join(HERE, "fig3_U_dependence.png"), dpi=200, bbox_inches="tight")
print("fig3_U_dependence.png")
