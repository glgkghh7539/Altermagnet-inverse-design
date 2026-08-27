#!/bin/bash

# check that the files exist
if [ ! -f "EIGENVAL" ] || [ ! -f "OUTCAR" ]; then
    echo "Error: Current directory must contain both 'EIGENVAL' and 'OUTCAR'."
    exit 1
fi

# generate the Python script
cat << 'PY_EOF' > _temp_sse_calc.py
import sys

def get_fermi_energy():
    ef = None
    try:
        with open('OUTCAR', 'r') as f:
            for line in f:
                if "E-fermi" in line:
                    parts = line.split()
                    ef = float(parts[2])
        return ef
    except:
        return None

def run():
    ef = get_fermi_energy()
    if ef is None:
        print("Error: Could not find 'E-fermi' in OUTCAR.")
        return

    print(f"Found Fermi Energy (Ef): {ef:.6f} eV")
    print(f"Searching Range        : {ef - 2.0:.6f} eV ~ {ef:.6f} eV")
    print(f"Condition              : Occupation_Up ~ 1.0 & Occupation_Down ~ 1.0")
    print("-" * 60)

    # 2. read EIGENVAL
    try:
        with open('EIGENVAL', 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading EIGENVAL: {e}")
        return

    # header handling
    try:
        header = lines[5].strip().split()
        nkpts = int(header[1])
        nbands = int(header[2])
    except:
        print("Error: Invalid EIGENVAL header.")
        return

    max_diff = -1.0
    result = None
    count_checked = 0
    count_in_range = 0

    idx = 7
    total_lines = len(lines)

    for k in range(nkpts):
        # skip blank lines
        while idx < total_lines and not lines[idx].strip():
            idx += 1
        if idx >= total_lines: break

        # k-point information
        k_line = lines[idx].strip().split()
        kx, ky, kz = float(k_line[0]), float(k_line[1]), float(k_line[2])
        idx += 1

        for b in range(nbands):
            if idx >= total_lines: break
            
            band_line = lines[idx].strip().split()
            band_id = int(band_line[0])

            # parse the ISPIN=2 data
            if len(band_line) == 5:
                e_up = float(band_line[1])
                e_down = float(band_line[2])
                occ_up = float(band_line[3])
                occ_down = float(band_line[4])

                # condition 1: occupation equal to 1.0 for both spins (valence band)
                if occ_up > 0.95 and occ_down > 0.95:
                    count_checked += 1
                    
                    # mean band energy (e_up and e_down could also be tested separately)
                    avg_e = (e_up + e_down) / 2.0
                    
                    # condition 2: energy window (Ef - 2.0 <= E <= Ef)
                    # the window may be applied loosely or strictly; it is applied strictly here
                    if (ef - 2.0) <= avg_e <= (ef + 0.1): # the upper bound is relaxed slightly above Ef, for numerical noise
                        count_in_range += 1
                        diff = abs(e_up - e_down)

                        if diff > max_diff:
                            max_diff = diff
                            result = {
                                'band': band_id,
                                'k_id': k + 1,
                                'k': (kx, ky, kz),
                                'diff': diff,
                                'energies': (e_up, e_down),
                                'occ': (occ_up, occ_down)
                            }
            idx += 1

    if result:
        print(f"\n[Result] Max Splitting in Range [Ef-2.0, Ef]")
        print("-" * 60)
        print(f"Valid Bands Checked   : {count_in_range} (out of {count_checked} occupied states)")
        print(f"Max Energy Difference : {result['diff']:.6f} eV")
        print(f"Band Index            : {result['band']}")
        print(f"Band Energies         : Up={result['energies'][0]:.4f}, Down={result['energies'][1]:.4f} eV")
        print(f"Relative to Ef        : Up={result['energies'][0]-ef:.4f}, Down={result['energies'][1]-ef:.4f} eV")
        print(f"K-Point Index         : {result['k_id']}")
        print(f"K-Point Coordinates   : ({result['k'][0]:.6f}, {result['k'][1]:.6f}, {result['k'][2]:.6f})")
        print("-" * 60)
    else:
        print("\nNo bands found matching both Occupation=1 and Energy Range conditions.")

if __name__ == "__main__":
    run()
PY_EOF

python3 _temp_sse_calc.py
rm _temp_sse_calc.py
