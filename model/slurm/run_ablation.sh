#!/bin/bash
#SBATCH --job-name=ablation
#SBATCH --partition=g3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=04:00:00
#SBATCH --output=ablation_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mlrev
NCORE=${SLURM_CPUS_PER_TASK:-20}
echo "[$(date)] host=$(hostname) cores=$NCORE"
python -c "import xgboost,sklearn,pandas,numpy;print('xgboost',xgboost.__version__,'sklearn',sklearn.__version__)"
md5sum fin_data.csv
python ablation_grouped.py fin_data.csv --outdir . --workers "$NCORE" </dev/null
echo "[$(date)] done"
