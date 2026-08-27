#!/usr/bin/env bash
# =====================================================================
# Create INCAR next to every POSCAR file (same directory).
# - Top-level folders must be two element symbols concatenated: MX (e.g., CrSe, NiO, YBi)
# - If M in {Ag, Cd}: comment out the whole "## GGA+U and LDA+U" block.
# - Else: set LDAUU = <mapped_value> 0.00   (only the first number changes).
# =====================================================================

set -euo pipefail
shopt -s nullglob

# --- U value mapping for magnetic element M (first number of LDAUU) ---
declare -A Umap=(
  [Sc]=3 [Ti]=3 [V]=4 [Cr]=4 [Mn]=4 [Fe]=4 [Co]=3 [Ni]=7 [Cu]=7 [Zn]=7
  [Y]=3 [Zr]=3 [Nb]=3 [Mo]=4 [Ru]=4 [Rh]=4 [Pd]=4
)

# --- Elements for which GGA+U block must be fully commented ---
NOU_SET=("Ag" "Cd")

is_in_array() {
  local x="$1"; shift
  local e
  for e in "$@"; do [[ "$e" == "$x" ]] && return 0; done
  return 1
}

# --- Write the base INCAR template (exactly as provided) to a file path ---
write_incar_template() {
  local out="$1"
  cat > "$out" <<'INCAR_EOF'
 SYSTEM =
 Starting parameters for this run:
   NWRITE = 2          write-flag
   ISTART = 0          job   : 0-new, 1-cont, 2-samecut
   ICHARG = 2          charge: 0-wave, 1-file, 2-atom, >10-const
   INIWAV = 1          electr: 0-lowe 1-rand  2-diag

 Electronic Relaxation:
   PREC = high         low | medium | high
   ENCUT = 500         kinetic energy cutoff (eV)
#   NBANDS = 64 
   ISPIN = 2
   MAGMOM = 3 -3 99*0
#   LSORBIT  .TRUE.

   NELMDL = -9         number of delayed ELM steps
   NELM = 200          number of ELM steps
   EDIFF = 1E-06       energy stopping-criterion for ELM
   LREAL = Auto        real-space projection (.FALSE., .TRUE., On, Auto)
   IALGO = 38          algorithm (38=CG for small, 48=RMM for big systems)

## Van der Waals
   IVDW = 0
   LCHARG = .TRUE.
   LAECHG = .TRUE.
   LWAVE = .TRUE.
   LVTOT = .TRUE.

   NSW = 200           max number of geometry steps
   NELMIN = 6
   IBRION = 2          ionic relax: 0-MD, 1-quasi-Newton, 2-CG, 3-Damped MD
   EDIFFG = -0.01      force (eV/A) stopping-criterion for geometry steps
   ISIF = 3            (1:force=y stress=trace only ions=y shape=n volume=n)
   ISYM = 0            (1-use symmetry, 0-no symmetry)
   POTIM = 0.15        initial time step for geo-opt (increase for soft sys)

#  IDIPOL = 3
#  DIPOL = 0.5 0.5 0.5
#  LDIPOL = .TRUE.

## GGA+U and LDA+U
  LDAU = .TRUE.
  LDAUTYPE = 2
  LDAUL = 2 -1
  LDAUU = 4.00 0.00
  LDAUJ = 0.00 0.00
  LDAUPRINT = 1
  LMAXMIX = 4

#  DOS related values:
  ISMEAR = 0         (-5:tet_b,-4:tet,-3:scan,-1:Fermi,0:gaus,1:MP)
  SIGMA = 0.01       broadening in eV
#  NEDOS = 3000
  LORBIT = 11
#  EMIN = -20
#  EMAX = 10

#
#   NPAR = 4
   NCORE = 4
   LPLANE = .TRUE.
INCAR_EOF
}

# --- Comment out the entire GGA+U block (header through LMAXMIX line) ---
comment_u_block_inplace() {
  local file="$1"
  awk '
    BEGIN{inblock=0}
    /^[[:space:]]*##[[:space:]]*GGA\+U[[:space:]]*and[[:space:]]*LDA\+U[[:space:]]*$/ {inblock=1; print "# "$0; next}
    inblock && /^[[:space:]]*LMAXMIX[[:space:]]*=/ {print "# "$0; inblock=0; next}
    inblock {print "# "$0; next}
    {print}
  ' "$file" > "${file}.tmp" && mv "${file}.tmp" "$file"
}

# --- Update LDAUU first value to given float (e.g., 7.00) ---
update_LDAUU_inplace() {
  local file="$1"
  local uval="$2"   # e.g., 7.00
  # replace only the first number on the LDAUU line, keep trailing " 0.00"
  sed -E "s/^([[:space:]]*LDAUU[[:space:]]*=[[:space:]]*)[0-9.]+([[:space:]]+0\.00)/\1${uval}\2/" "$file" > "${file}.tmp" \
    && mv "${file}.tmp" "$file"
}

echo "==> Scanning MX folders in: $(pwd)"
echo

# Loop top-level directories like CrSe/, NiO/, YBi/, ...
for folder in */ ; do
  name="${folder%/}"

  # Match exactly two element symbols concatenated: M + X
  if [[ "$name" =~ ^([A-Z][a-z]?)([A-Z][a-z]?)$ ]]; then
    M="${BASH_REMATCH[1]}"
    X="${BASH_REMATCH[2]}"
  else
    continue
  fi

  echo "Processing ${name}  (M=${M}, X=${X})"

  # Find every POSCAR under this MX folder and create INCAR alongside it
  found_any=0
  while IFS= read -r -d $'\0' pos; do
    dir="$(dirname "$pos")"
    out="${dir}/INCAR"

    # 1) write template
    write_incar_template "$out"

    # 2) apply U rules
    if is_in_array "$M" "${NOU_SET[@]}"; then
      # Ag or Cd: comment out whole GGA+U block
      comment_u_block_inplace "$out"
      echo "  [+] ${out} (GGA+U block commented for M=${M})"
    else
      # others: set LDAUU first value if mapped; leave default if not mapped
      if [[ -n "${Umap[$M]+x}" ]]; then
        printf -v Ufmt "%.2f" "${Umap[$M]}"
        update_LDAUU_inplace "$out" "$Ufmt"
        echo "  [+] ${out} (LDAUU set to ${Ufmt} 0.00 for M=${M})"
      else
        echo "  [!] ${out} (no U map for M=${M}; left as template default 4.00 0.00)"
      fi
    fi
    found_any=1
  done < <(find "$name" -type f -name "POSCAR" -print0)

  if [[ "$found_any" -eq 0 ]]; then
    echo "  (no POSCAR files found under ${name}/)"
  fi

  echo
done

echo "==> Done. All INCARs were created next to their POSCARs."

