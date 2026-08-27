#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_strains.py  (updated 2025-08-01)

Scans axis_strain/POSCAR_*/[xyz][0-9]{3} directories,
keeps only those carrying a DONE flag,
parses OUTCAR and EIGENVAL,
and writes the spin-splitting results to CSV.

- structure: "material_strain" form, e.g. POSCAR_Ag2F6_3_x950
- skipped when the magnetization of both ion1 and ion2 is below 0.001
"""

import csv
from pathlib import Path
from parse_eigenval import check_magnetization, parse_eigenval

ROOT = Path("axis_strain")
CSV_OUT = ROOT / "spin_splitting_summary.csv"

HEADER = [
    "structure",
    "fermi_level", "gamma_avg_meV", "max_split_eV",
    "band_number", "kx", "ky", "kz",
    "ion1_tot", "ion2_tot", "overall_tot"
]

rows = []

valid_prefixes = ("x", "y", "z")

for mat_dir in sorted(ROOT.glob("POSCAR_*")):
    mat_name = mat_dir.name
    for strain_dir in sorted(mat_dir.iterdir()):
        name = strain_dir.name
        # directory pattern and DONE flag check
        if not strain_dir.is_dir(): 
            continue
        if not (name.startswith(valid_prefixes) and len(name) == 4 and name[1:].isdigit()):
            continue
        if not (strain_dir / "DONE").exists():
            continue

        structure_id = f"{mat_name}_{name}"
        outcar = strain_dir / "OUTCAR"
        eig    = strain_dir / "EIGENVAL"

        if not (outcar.is_file() and eig.is_file()):
            print(f"[WARN] Missing OUTCAR/EIGENVAL in {structure_id}, skipping.")
            continue

        # 1) read the magnetization
        mag_ok, overall_tot, ion1_tot, ion2_tot = check_magnetization(str(outcar))

        # 2) ion1/ion2 small skip
        if ion1_tot is not None and ion2_tot is not None:
            if abs(ion1_tot) < 0.001 and abs(ion2_tot) < 0.001:
                print(f"[INFO] Skipping {structure_id}: Both ion1 and ion2 tot < 0.001")
                continue

        if not mag_ok:
            print(f"[INFO] Skipping {structure_id}: magnetization condition not met (overall_tot={overall_tot})")
            continue

        # 3) parse EIGENVAL
        try:
            gamma_avg, max_split, max_kpt, max_band, fermi = parse_eigenval(str(eig))
        except Exception as e:
            print(f"[ERROR] parse_eigenval failed at {structure_id}: {e}")
            continue

        if gamma_avg is None:
            print(f"[INFO] Skipping {structure_id}: no Γ-point data")
            continue

        # k-point unpack
        if max_kpt:
            kx, ky, kz = max_kpt
        else:
            kx = ky = kz = ""

        # unit conversion
        gamma_meV = gamma_avg * 1000.0

        rows.append([
            structure_id,
            f"{fermi:.4f}",
            f"{gamma_meV:.3f}",
            f"{max_split:.6f}",
            str(max_band) if max_band is not None else "",
            f"{kx:.6f}" if kx != "" else "",
            f"{ky:.6f}" if ky != "" else "",
            f"{kz:.6f}" if kz != "" else "",
            f"{ion1_tot:.3f}" if ion1_tot is not None else "",
            f"{ion2_tot:.3f}" if ion2_tot is not None else "",
            f"{overall_tot:.3f}" if overall_tot is not None else "",
        ])
        print(f"[PROCESSED] {structure_id}")

# write the CSV
with CSV_OUT.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(HEADER)
    writer.writerows(rows)

print(f"Analysis complete. Results saved to: {CSV_OUT}")

