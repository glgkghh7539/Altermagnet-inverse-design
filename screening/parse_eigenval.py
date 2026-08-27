#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parse_eigenval.py EIGENVAL

Overview:
1. From the OUTCAR in the same directory:
   (a) read the E-fermi value of the last SCF cycle, and
   (b) locate the last magnetization (x) block and check that the tot values of
       ions 1 and 2 are present and that |overall tot| is at most 0.001.
2. Only if that condition holds, parse the EIGENVAL file and:
   - compute the spin splitting at Gamma (kx, ky, kz all near 0), taken here as the mean
     of |E_up - E_down| over all bands, and record it;
   - find the maximum spin splitting in the window -2 eV to 0 eV relative to the Fermi
     level (E in [E_F-2, E_F]) together with its location (k-point and band index).
3. If the condition is not met, print a skip message.
"""

import sys
import os
import re

# threshold constants
TOL_GAMMA = 1e-6         # tolerance for deciding that k is 0
ENERGY_WINDOW = 2.0      # -2 eV to 0 eV relative to the Fermi level
MAG_TOT_THRESHOLD = 0.01  # threshold on |overall tot| (in VASP magnetic units, not eV)

def find_fermi_from_outcar(outcar_path):
    """
    Search backwards from the end of the OUTCAR for the last 'E-fermi : x.xxxx' value.
    """
    if not os.path.isfile(outcar_path):
        return None

    with open(outcar_path, 'r', errors='ignore') as f:
        lines = f.readlines()

    for line in reversed(lines):
        if "E-fermi" in line:
            match = re.search(r"E-fermi\s*:\s*([-\d\.]+)", line)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    pass
    return None

def check_magnetization(outcar_path):
    """
    From the last magnetization (x) block of the OUTCAR:
    - read the tot values of ions 1 and 2, and
    - check that the absolute value of the overall tot (on the final "tot" line) is at most 0.001.
    
    Returns:
      (condition_met, overall_tot, ion1_tot, ion2_tot)
    """
    if not os.path.isfile(outcar_path):
        return False, None, None, None

    with open(outcar_path, 'r', errors='ignore') as f:
        content = f.read()

    # locate the last magnetization (x) block
    pos = content.rfind("magnetization (x)")
    if pos == -1:
        return False, None, None, None
    block = content[pos:]
    lines = block.splitlines()

    overall_tot = None
    ion1_tot = None
    ion2_tot = None

    # the overall tot is on the line beginning with "tot" (last column)
    for line in lines:
        if line.strip().lower().startswith("tot"):
            tokens = line.split()
            if len(tokens) >= 2:
                try:
                    overall_tot = float(tokens[-1])
                except ValueError:
                    overall_tot = None
            break

    # ions 1 and 2 are read from the lines whose first field is the ion index
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        parts = line.split()
        try:
            ion_idx = int(parts[0])
        except ValueError:
            continue
        if ion_idx == 1 and len(parts) >= 5:
            try:
                ion1_tot = float(parts[-1])
            except ValueError:
                ion1_tot = None
        elif ion_idx == 2 and len(parts) >= 5:
            try:
                ion2_tot = float(parts[-1])
            except ValueError:
                ion2_tot = None
        if ion1_tot is not None and ion2_tot is not None:
            break

    if overall_tot is None or ion1_tot is None or ion2_tot is None:
        return False, overall_tot, ion1_tot, ion2_tot

    # condition: |overall tot| <= 0.001
    if abs(overall_tot) <= MAG_TOT_THRESHOLD:
        return True, overall_tot, ion1_tot, ion2_tot
    else:
        return False, overall_tot, ion1_tot, ion2_tot

def parse_eigenval(file_path):
    """
    Parse the EIGENVAL file and compute:
      1. the spin splitting at Gamma (k near 0): the mean of |E_up - E_down| over all bands
      2. the maximum spin splitting in the window -2 eV to 0 eV relative to the Fermi level
         (E in [E_F-2, E_F]) together with its location (k-point, band index)
         
    In the header, (nelect, nkpoints, nbands) is found as the line holding three integers.
    The band data are assumed to be one k-point coordinate line followed by nbands lines.
    """
    dir_of_eigenval = os.path.dirname(os.path.abspath(file_path))
    outcar_path = os.path.join(dir_of_eigenval, "OUTCAR")
    
    fermi = find_fermi_from_outcar(outcar_path)
    if fermi is None:
        fermi = 0.0

    # energy window: [E_F - 2, E_F]
    E_LOWER = fermi - ENERGY_WINDOW
    E_UPPER = fermi

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"EIGENVAL not found: {file_path}")

    with open(file_path, "r", errors='ignore') as f:
        raw_lines = f.readlines()

    # drop blank lines and surrounding whitespace
    lines = [line.strip() for line in raw_lines if line.strip()]
    if len(lines) < 2:
        raise ValueError("EIGENVAL file too short or empty.")

    # 1) find (nelect, nkpoints, nbands) in the header: the first line holding three integers
    nelect = nkpoints = nbands = None
    header_index = -1
    for i, line in enumerate(lines):
        parts = line.split()
        try:
            ints = list(map(int, parts))
        except:
            continue
        if len(ints) == 3:
            nelect, nkpoints, nbands = ints
            header_index = i
            break

    if nkpoints is None or nbands is None:
        raise ValueError("Could not find header line with 3 integers (nelect, nkpoints, nbands).")

    # 2) parse the k-point and band data, starting after header_index
    idx = header_index + 1
    data_kpoints = []  # per k-point: (kx, ky, kz, [ (band_index, E_up, E_down), ... ])
    for k in range(nkpoints):
        if idx >= len(lines):
            break
        # the k-point coordinate line (usually one line; the weight is ignored)
        k_line = lines[idx].split()
        idx += 1
        try:
            kx, ky, kz = map(float, k_line[:3])
        except:
            kx = ky = kz = 0.0
        band_list = []
        for b in range(nbands):
            if idx >= len(lines):
                break
            band_line = lines[idx].split()
            idx += 1
            if len(band_line) < 5:
                continue
            try:
                band_index = int(band_line[0])
                E_up = float(band_line[1])
                E_down = float(band_line[2])
            except:
                continue
            band_list.append((band_index, E_up, E_down))
        data_kpoints.append((kx, ky, kz, band_list))

    # 3) splitting at Gamma (k near 0): the mean of |E_up - E_down| over all bands
    gamma_diffs = []
    for (kx, ky, kz, bands) in data_kpoints:
        if abs(kx) < TOL_GAMMA and abs(ky) < TOL_GAMMA and abs(kz) < TOL_GAMMA:
            for (bidx, E_up, E_down) in bands:
                gamma_diffs.append(abs(E_up - E_down))
            break  # if several Gamma points are present, use only the first

    if gamma_diffs:
        gamma_avg = sum(gamma_diffs) / len(gamma_diffs)
    else:
        gamma_avg = None

    # 4) maximum spin splitting in [-2, 0] relative to the Fermi level, and its location
    max_window_diff = 0.0
    max_kpoint = None
    max_band_idx = None
    for (kx, ky, kz, bands) in data_kpoints:
        for (bidx, E_up, E_down) in bands:
            # both energies must lie within -2 eV to 0 eV of the Fermi level
            if (E_LOWER <= E_up <= E_UPPER) and (E_LOWER <= E_down <= E_UPPER):
                diff = abs(E_up - E_down)
                if diff > max_window_diff:
                    max_window_diff = diff
                    max_kpoint = (kx, ky, kz)
                    max_band_idx = bidx

    return gamma_avg, max_window_diff, max_kpoint, max_band_idx, fermi

def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_eigenval.py EIGENVAL")
        sys.exit(1)

    eigenval_path = sys.argv[1]
    dir_of_file = os.path.dirname(os.path.abspath(eigenval_path))
    outcar_path = os.path.join(dir_of_file, "OUTCAR")
    
    # check the magnetization condition first
    mag_ok, overall_tot, ion1_tot, ion2_tot = check_magnetization(outcar_path)
    if not mag_ok:
        print("Magnetization condition not satisfied:")
        print(f"  overall tot = {overall_tot}, ion1 tot = {ion1_tot}, ion2 tot = {ion2_tot}")
        sys.exit(0)

    try:
        gamma_avg, max_window_diff, max_kpt, max_band_idx, fermi = parse_eigenval(eigenval_path)
    except Exception as e:
        print(f"Error while parsing EIGENVAL: {e}")
        sys.exit(1)

    # report:
    #  - the splitting at Gamma (mean over bands)
    #  - the maximum spin splitting in [-2, 0] relative to the Fermi level, and its location
    print(f"Fermi level = {fermi:.4f} eV")
    if gamma_avg is not None:
        print(f"Gamma point average spin splitting = {gamma_avg*1000:.3f} meV")
    else:
        print("No Gamma point data found.")

    if max_window_diff > 0.0 and max_kpt is not None:
        print(f"Max spin splitting in [E_F-2, E_F] = {max_window_diff:.6f} eV")
        print(f"Found at k-point = {max_kpt}, band index = {max_band_idx}")
    else:
        print("No spin splitting data found in the energy window [E_F-2, E_F].")

    # optionally write the result to a file in the current directory
    output_file = os.path.join(dir_of_file, "spin_splitting_summary.txt")
    with open(output_file, "w") as f_out:
        f_out.write(f"Fermi level = {fermi:.4f} eV\n")
        if gamma_avg is not None:
            f_out.write(f"Gamma point average spin splitting = {gamma_avg*1000:.3f} meV\n")
        else:
            f_out.write("No Gamma point data found.\n")
        if max_window_diff > 0.0 and max_kpt is not None:
            f_out.write(f"Max spin splitting in [E_F-2, E_F] = {max_window_diff:.6f} eV\n")
            f_out.write(f"Found at k-point = {max_kpt}, band index = {max_band_idx}\n")
        else:
            f_out.write("No spin splitting data found in the energy window [E_F-2, E_F].\n")
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()

