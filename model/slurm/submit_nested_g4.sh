#!/bin/bash
#SBATCH --job-name=nested_f
#SBATCH --partition=g3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --output=nested_hp_final_%j.log

WORKDIR=$HOME/altermagnet_revision/nested_hp_final
CSV=$WORKDIR/fin_data.csv

mkdir -p "$WORKDIR/out"
cd "$WORKDIR"


export OMP_NUM_THREADS=4

python3 nested_hp_validation.py \
    --csv "$CSV" \
    --outdir "$WORKDIR/out" \
    --n-procs 5 \
    --xgb-jobs 4
