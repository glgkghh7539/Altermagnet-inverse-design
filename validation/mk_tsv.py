#!/usr/bin/env python
"""Flatten hull_final.json (+ the MP reference) into one tab-separated table."""
import json, os

# --- paths --------------------------------------------------------------------
# Tables this script reads are deposited in validation/data/. The raw VASP run trees
# are NOT deposited (tens of GB); they default to ~/NiS_hull and can be pointed
# elsewhere with $NIS_HULL_DIR.
HERE    = os.path.dirname(os.path.abspath(__file__))
DEPOT   = os.path.join(HERE, "data")
RUNROOT = os.environ.get("NIS_HULL_DIR", os.path.expanduser("~/NiS_hull"))

rows = json.load(open(os.path.join(DEPOT, "hull_final.json")))
d = json.load(open(os.path.join(DEPOT, "mp_reference.json")))
mp = {e["material_id"]: e for e in (d["all"] if isinstance(d, dict) else d)}

# the NiAs-type NiS is our own structure, but MP has the same phase as mp-594
ALIAS = {"local-NiAs_NiS_P6_3mmc": "mp-594"}

cols = ["tag", "formula", "spacegroup", "x_S", "natoms",
        "E_total_eV", "E_per_atom_eV",
        "Ef_raw_eV_atom", "Ehull_raw_meV_atom",
        "Ef_MP2020_eV_atom", "Ehull_MP2020_meV_atom", "on_hull_MP2020",
        "Ef_MP_eV_atom", "Ehull_MP_meV_atom", "dEf_vs_MP_meV_atom"]

out = [cols]
for r in rows:
    tag = r["tag"]
    mpid = ALIAS.get(tag, tag.split("_")[0])
    m = mp.get(mpid)
    # split at the FIRST underscore after the id: space groups themselves
    # contain underscores (P6_3mmc, P2_1c), so rsplit would mangle them
    rest = tag.split("_", 1)[1] if "_" in tag else tag
    parts = rest.split("_", 1)
    formula, spg = (parts + [""])[:2]
    ef_mp = "%.4f" % m["e_form_mp"] if m else ""
    eh_mp = "%.1f" % (m["e_above_hull"] * 1000) if m else ""
    dd = ("%+.1f" % ((r["ef_mp2020"] - m["e_form_mp"]) * 1000)
          if m and 0 < r["x_S"] < 1 else "")
    out.append([tag, formula, spg, "%.4f" % r["x_S"], str(r["natoms"]),
                "%.6f" % r["e_total"], "%.6f" % r["e_per_atom"],
                "%.4f" % r["ef_raw"], "%.1f" % (r["e_hull_raw"] * 1000),
                "%.4f" % r["ef_mp2020"], "%.1f" % (r["e_hull_mp2020"] * 1000),
                "yes" if r["on_hull_mp2020"] else "no",
                ef_mp, eh_mp, dd])

p = os.path.join(HERE, "hull_final.tsv")
with open(p, "w") as f:
    for row in out:
        f.write("\t".join(row) + "\n")
print("wrote", p)
