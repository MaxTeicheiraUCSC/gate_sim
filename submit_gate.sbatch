#!/bin/bash
#SBATCH --job-name=gate_czt
#SBATCH --partition=128x24
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --array=1-10
#SBATCH --output=logs/gate_%A_%a.out
#SBATCH --error=logs/gate_%A_%a.err

# GATE 10 CZT Slit Simulation

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GATE_SIM_DIR="${GATE_SIM_DIR:-$SCRIPT_DIR}"
TOTAL_PRIMARIES=${TOTAL_PRIMARIES:-100000000}
NUM_JOBS=${SLURM_ARRAY_TASK_COUNT:-10}
PRIMARIES_PER_JOB=$((TOTAL_PRIMARIES / NUM_JOBS))

echo "============================================================"
echo "GATE CZT Slit Simulation - Job ${SLURM_ARRAY_TASK_ID}/${NUM_JOBS}"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Primaries per job: ${PRIMARIES_PER_JOB}"

mkdir -p "${GATE_SIM_DIR}/logs"
cd ${GATE_SIM_DIR}

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gate_env

echo "Python: $(which python)"
echo "Working dir: $(pwd)"

# Build Slack flags if webhook is available
SLACK_FLAGS=""
if [[ -n "${AUTOPILOT_SLACK_WEBHOOK:-}" ]]; then
    SLACK_FLAGS="--slack-webhook ${AUTOPILOT_SLACK_WEBHOOK} --slack-interval 30"
fi

echo "Starting simulation..."
python ${GATE_SIM_DIR}/czt_slit_simulation_cluster.py \
    --seed ${SLURM_ARRAY_TASK_ID} \
    --primaries ${PRIMARIES_PER_JOB} \
    --job-id ${SLURM_ARRAY_TASK_ID} \
    ${SLACK_FLAGS}

EXIT_CODE=$?
echo "Job ${SLURM_ARRAY_TASK_ID} finished with exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
