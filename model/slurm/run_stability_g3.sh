#!/bin/bash
#SBATCH --job-name=stabsel_f
#SBATCH --partition=g3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=02:00:00
#SBATCH --output=stabsel_final_%j.log

# ---------------------------------------------------------------------------
# Nested grouped CV + stability selection, 20 seeds x 5 outer folds = 100 runs.
#
# Needs ONLY fin_data.csv as data. Python deps: numpy pandas xgboost scikit-learn.
#
# Setup once (login node MASTER):
#     conda create -n mlrev python=3.11 -y
#     conda activate mlrev
#     conda install -c conda-forge numpy pandas scikit-learn xgboost -y
#
# Then:
#     chmod +x run_stability_g3.sh
#     sbatch run_stability_g3.sh
#
# g3 = n007/n008, 20 cores, ~120 GB free. Expect ~6-10 min wall on 20 cores.
# (g5/g6 are currently allocated to another user; g4 n011 has only ~46 GB free.)
# ---------------------------------------------------------------------------

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

# XGBoost and BLAS MUST stay single-threaded inside each worker process,
# otherwise 20 workers x 20 threads oversubscribe the node and it runs SLOWER
# than serial. The Python script sets these too, but belt and braces.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mlrev

NCORE=${SLURM_CPUS_PER_TASK:-20}

echo "[$(date)] host=$(hostname) cores=$NCORE"
python -c "import xgboost,sklearn,pandas,numpy;print('xgboost',xgboost.__version__,'sklearn',sklearn.__version__,'pandas',pandas.__version__,'numpy',numpy.__version__)"

if [ ! -f fin_data.csv ]; then
    echo "ERROR: fin_data.csv not found in $SLURM_SUBMIT_DIR" >&2
    exit 1
fi

python stability_selection_100_parallel.py fin_data.csv --workers "$NCORE" </dev/null

echo "[$(date)] done"
ls -la stability_100.csv nested_folds_100.csv
