#!/usr/bin/env bash
# Create POTCAR right next to every POSCAR file (same directory).

set -euo pipefail

# VASP PBE PSP library. POTCAR files are licensed and are not deposited; point
# $VASP_PBE at your own copy, or edit the fallback below.
ROOT_PSP="${VASP_PBE:-/scratch/e1348a04/VASP_PBE}"

declare -A element_map=(
  [B]="B" [C]="C" [N]="N" [O]="O" [F]="F"
  [Al]="Al" [Si]="Si" [P]="P" [S]="S" [Cl]="Cl"
  [Sc]="Sc_sv" [Ti]="Ti_sv" [V]="V_sv" [Cr]="Cr_pv" [Mn]="Mn_pv"
  [Fe]="Fe" [Co]="Co" [Ni]="Ni" [Cu]="Cu" [Zn]="Zn"
  [Ga]="Ga_d" [Ge]="Ge_d" [As]="As" [Se]="Se" [Br]="Br"
  [Y]="Y_sv" [Zr]="Zr_sv" [Nb]="Nb_sv" [Mo]="Mo_sv"
  [Ru]="Ru_pv" [Rh]="Rh_pv" [Pd]="Pd" [Ag]="Ag" [Cd]="Cd"
  [In]="In_d" [Sn]="Sn_d" [Sb]="Sb" [Te]="Te" [I]="I"
  [La]="La" [Ce]="Ce" [Pr]="Pr_3" [Nd]="Nd_3" [Sm]="Sm_3"
  [Eu]="Eu_2" [Gd]="Gd_3" [Tb]="Tb_3" [Dy]="Dy_3" [Ho]="Ho_3"
  [Er]="Er_3" [Tm]="Tm_3" [Yb]="Yb_2" [Lu]="Lu_3"
  [Hf]="Hf_pv" [Ta]="Ta_pv" [W]="W_sv" [Re]="Re" [Os]="Os"
  [Ir]="Ir" [Pt]="Pt" [Au]="Au" [Hg]="Hg"
  [Tl]="Tl_d" [Pb]="Pb_d" [Bi]="Bi_d"
)

potcar_path() {
  local elem="$1"
  local mapped="${element_map[$elem]-}"
  [[ -n "$mapped" ]] && echo "${ROOT_PSP}/${mapped}/POTCAR" || echo ""
}

echo "==> Scanning MX folders in: $(pwd)"
echo

for folder in */ ; do
  name="${folder%/}"

  # MX (two element symbols stuck together)
  if [[ "$name" =~ ^([A-Z][a-z]?)([A-Z][a-z]?)$ ]]; then
    M="${BASH_REMATCH[1]}"
    X="${BASH_REMATCH[2]}"
  else
    continue
  fi

  echo "Processing ${name}  (M=${M}, X=${X})"

  M_path="$(potcar_path "$M")"
  X_path="$(potcar_path "$X")"

  if [[ -z "$M_path" || -z "$X_path" ]]; then
    echo "  [SKIP] mapping not found: M='${M}' -> '${element_map[$M]-}', X='${X}' -> '${element_map[$X]-}'"
    echo
    continue
  fi
  if [[ ! -s "$M_path" ]]; then
    echo "  [ERROR] missing POTCAR for M=${M}: $M_path"
    echo
    continue
  fi
  if [[ ! -s "$X_path" ]]; then
    echo "  [ERROR] missing POTCAR for X=${X}: $X_path"
    echo
    continue
  fi

  # Find every POSCAR under this MX folder and create POTCAR alongside it
  found_any=0
  while IFS= read -r -d '' pos; do
    dir="$(dirname "$pos")"
    out="${dir}/POTCAR"
    cat "$M_path" "$X_path" > "$out"
    echo "  [+] ${out}  (from ${M_path} + ${X_path})"
    found_any=1
  done < <(find "$name" -type f -name "POSCAR" -print0)

  if [[ "$found_any" -eq 0 ]]; then
    echo "  (no POSCAR files found under ${name}/)"
  fi

  echo
done

echo "==> Done. All POTCARs were created next to their POSCARs."

