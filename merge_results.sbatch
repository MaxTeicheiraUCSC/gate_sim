#!/bin/bash
#SBATCH --job-name=gate_merge
#SBATCH --partition=128x24
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00

# Merge results from GATE CZT simulation array jobs

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================================"
echo "GATE CZT Merge Results"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Script dir: ${SCRIPT_DIR}"

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gate_env

echo "Python: $(which python)"
echo "Working dir: $(pwd)"

cd "${SCRIPT_DIR}"

echo "Starting merge..."
python "${SCRIPT_DIR}/merge_results.py" --output-dir "${SCRIPT_DIR}/output"

EXIT_CODE=$?
echo "Merge finished with exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
