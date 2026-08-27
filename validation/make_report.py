#!/usr/bin/env python
"""Build NiS_formation_energy.md from the finished data, so every number in the
report comes from the data files rather than being retyped."""
import json, os, re

# --- paths --------------------------------------------------------------------
# Tables this script reads are deposited in validation/data/. The raw VASP run trees
# are NOT deposited (tens of GB); they default to ~/NiS_hull and can be pointed
# elsewhere with $NIS_HULL_DIR.
HERE    = os.path.dirname(os.path.abspath(__file__))
DEPOT   = os.path.join(HERE, "data")
RUNROOT = os.environ.get("NIS_HULL_DIR", os.path.expanduser("~/NiS_hull"))

rows = json.load(open(os.path.join(DEPOT, "hull_final.json")))
mpref = {e["material_id"]: e
         for e in json.load(open(os.path.join(DEPOT, "mp_reference.json")))["all"]}
summ = json.load(open(os.path.join(DEPOT, "uscan_hull_summary.json")))
ALIAS = {"local-NiAs_NiS_P6_3mmc": "mp-594"}
NIAS = "local-NiAs_NiS_P6_3mmc"

NAME = {
    "mp-23_Ni_Fm-3m": ("Ni", "Fm-3m", "fcc nickel"),
    "mp-362_Ni3S2_R32": ("Ni₃S₂", "R32", "heazlewoodite"),
    "mp-976920_Ni9S8_C222": ("Ni₉S₈", "C222", "godlevskite"),
    "mp-1547_NiS_R3m": ("NiS", "R3m", "millerite (beta-NiS, low-temperature phase)"),
    NIAS: ("**NiS**", "**P6_3/mmc**", "**NiAs-type alpha-NiS - the altermagnet**"),
    "mp-1050_Ni3S4_Fd-3m": ("Ni₃S₄", "Fd-3m", "polydymite"),
    "mp-1180046_NiS2_P2_1c": ("NiS₂", "P2₁/c", "—"),
    "mp-2282_NiS2_Pa-3": ("NiS₂", "Pa-3", "pyrite (vaesite)"),
    "mp-77_S_Fddd": ("S", "Fddd", "alpha-sulfur"),
}


def hull_table():
    out = ["| phase | space group | mineral | atoms | x_S | Ef_raw | Ef (MP2020) | E_hull | MP: Ef | MP: E_hull |",
           "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(rows, key=lambda r: (r["x_S"], r["ef_mp2020"])):
        f, spg, mineral = NAME[r["tag"]]
        m = mpref.get(ALIAS.get(r["tag"], r["tag"].split("_")[0]))
        star = " ★" if r["on_hull_mp2020"] else ""
        out.append("| %s | %s | %s | %d | %.3f | %.4f | **%.4f** | **%.1f**%s | %s | %s |" % (
            f, spg, mineral, r["natoms"], r["x_S"], r["ef_raw"], r["ef_mp2020"],
            r["e_hull_mp2020"] * 1000, star,
            "%.4f" % m["e_form_mp"] if m else "—",
            "%.1f" % (m["e_above_hull"] * 1000) if m else "—"))
    return "\n".join(out)


def dev_stats():
    d = []
    for r in rows:
        m = mpref.get(ALIAS.get(r["tag"], r["tag"].split("_")[0]))
        if m and 0 < r["x_S"] < 1:
            d.append((r["ef_mp2020"] - m["e_form_mp"]) * 1000)
    return len(d), sum(abs(x) for x in d) / len(d), min(d), max(d)


def u_table():
    out = ["| U (eV) | FM-consistent set | AFM (FM products) | decomposition products | m(Ni) AFM | V/atom AFM |",
           "|---:|---:|---:|---|---:|---:|"]
    for u in sorted({s["U"] for s in summ}):
        fm = next(s for s in summ if s["U"] == u and s["variant"].startswith("NiAs FM"))
        af = next(s for s in summ if s["U"] == u and s["variant"].startswith("NiAs AFM"))
        b = "**" if u == 0 else ""
        out.append("| %s%d%s | %s%+.1f%s | %s%+.1f%s | %s | %.3f | %.2f |" % (
            b, u, b, b, fm["dE_meV"], b, b, af["dE_meV"], b,
            af["products"].replace("Ni9S8", "Ni₉S₈").replace("Ni3S4", "Ni₃S₄")
                          .replace("NiS2", "NiS₂"),
            af["m_Ni"], af["vol_per_atom"]))
    return "\n".join(out)


def order_table():
    per = {}
    for ln in open(os.path.join(DEPOT, "uscan_report.txt")):
        m = re.match(r"\s*(\d+) \| (\S+)\s+\|\s+(-?\d+\.\d+)\s+(\d+\.\d+)", ln)
        if m:
            per[(int(m.group(1)), m.group(2))] = float(m.group(3))
    out = ["| U (eV) | millerite | NiAs type (AFM) | lower of the two | matches experiment |",
           "|---:|---:|---:|---|:---:|"]
    for u in sorted({s["U"] for s in summ}):
        mil = per.get((u, "mp-1547_NiS_R3m"))
        nia = per.get((u, "local-NiAs_NiS_P6_3mmc_AFM"))
        if mil is None or nia is None:
            continue
        low = "millerite" if mil < nia else "NiAs type"
        ok = "○" if mil < nia else "✗"
        out.append("| %d | %.5f | %.5f | %s | %s |" % (u, mil, nia, low, ok))
    return "\n".join(out)


n, mad, lo, hi = dev_stats()
nias = next(r for r in rows if r["tag"] == NIAS)
mil = next(r for r in rows if r["tag"] == "mp-1547_NiS_R3m")

md = open(os.path.join(DEPOT, "report_template.md")).read()
md = md.format(
    hull_table=hull_table(), u_table=u_table(), order_table=order_table(),
    n_dev=n, mad=mad, dev_lo=lo, dev_hi=hi,
    nias_ef=nias["ef_mp2020"], nias_hull=nias["e_hull_mp2020"] * 1000,
    nias_raw=nias["ef_raw"],
    mil_ef=mil["ef_mp2020"], mil_hull=mil["e_hull_mp2020"] * 1000,
    poly_gap=(nias["ef_mp2020"] - mil["ef_mp2020"]) * 1000,
)
p = os.path.join(HERE, "NiS_formation_energy.md")
open(p, "w").write(md)
print("wrote", p, "(%d chars)" % len(md))
