#!/bin/bash
# =============================================================================
# GATE Simulation Job Submission Wrapper
# =============================================================================
# This script configures and submits the GATE simulation array job to SLURM.
# It handles calculation of primaries per job and sets up the environment.
#
# Usage:
#   ./submit_jobs.sh                    # Default: 10M primaries, 10 jobs
#   ./submit_jobs.sh -n 20000000 -j 20  # 20M primaries, 20 jobs
#   ./submit_jobs.sh --test             # Test mode: 100K primaries, 2 jobs
# =============================================================================

set -e

# Default parameters
TOTAL_PRIMARIES=10000000    # 10 million primaries
NUM_JOBS=10                 # Number of parallel jobs
PARTITION="96x24gpu4"             # SLURM partition (adjust based on sinfo)
TIME_LIMIT="04:00:00"       # 4 hours per job
MEMORY="16G"                # Memory per job
CPUS=8                      # CPUs per job
TEST_MODE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--primaries)
            TOTAL_PRIMARIES="$2"
            shift 2
            ;;
        -j|--jobs)
            NUM_JOBS="$2"
            shift 2
            ;;
        -p|--partition)
            PARTITION="$2"
            shift 2
            ;;
        -t|--time)
            TIME_LIMIT="$2"
            shift 2
            ;;
        -m|--memory)
            MEMORY="$2"
            shift 2
            ;;
        -c|--cpus)
            CPUS="$2"
            shift 2
            ;;
        --test)
            TEST_MODE=true
            TOTAL_PRIMARIES=100000
            NUM_JOBS=2
            TIME_LIMIT="00:30:00"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -n, --primaries NUM    Total primaries to simulate (default: 10000000)"
            echo "  -j, --jobs NUM         Number of parallel jobs (default: 10)"
            echo "  -p, --partition NAME   SLURM partition (default: gpu)"
            echo "  -t, --time HH:MM:SS    Time limit per job (default: 04:00:00)"
            echo "  -m, --memory SIZE      Memory per job (default: 16G)"
            echo "  -c, --cpus NUM         CPUs per job (default: 8)"
            echo "  --test                 Test mode: 100K primaries, 2 jobs, 30 min"
            echo "  -h, --help             Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Calculate primaries per job
PRIMARIES_PER_JOB=$((TOTAL_PRIMARIES / NUM_JOBS))

echo "============================================================"
echo "GATE CZT Simulation - Job Submission"
echo "============================================================"
if [ "$TEST_MODE" = true ]; then
    echo "MODE: TEST (reduced parameters for verification)"
fi
echo ""
echo "Configuration:"
echo "  Total primaries:    ${TOTAL_PRIMARIES}"
echo "  Number of jobs:     ${NUM_JOBS}"
echo "  Primaries per job:  ${PRIMARIES_PER_JOB}"
echo "  Partition:          ${PARTITION}"
echo "  Time limit:         ${TIME_LIMIT}"
echo "  Memory:             ${MEMORY}"
echo "  CPUs per job:       ${CPUS}"
echo "============================================================"

# Create logs directory
mkdir -p logs

# Check if simulation script exists
if [ ! -f "czt_slit_simulation_cluster.py" ]; then
    echo "ERROR: czt_slit_simulation_cluster.py not found!"
    echo "Make sure you're running this script from the simulation directory."
    exit 1
fi

# Check if sbatch file exists
if [ ! -f "submit_gate.sbatch" ]; then
    echo "ERROR: submit_gate.sbatch not found!"
    exit 1
fi

# Confirm submission
echo ""
read -p "Submit jobs? [y/N] " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Submit the array job
echo "Submitting array job..."
JOB_ID=$(sbatch \
    --partition=${PARTITION} \
    --array=1-${NUM_JOBS} \
    --time=${TIME_LIMIT} \
    --mem=${MEMORY} \
    --cpus-per-task=${CPUS} \
    --export=ALL,TOTAL_PRIMARIES=${TOTAL_PRIMARIES} \
    submit_gate.sbatch | awk '{print $4}')

echo ""
echo "============================================================"
echo "Job submitted successfully!"
echo "============================================================"
echo "Job ID: ${JOB_ID}"
echo "Array tasks: ${NUM_JOBS} jobs (IDs: 1-${NUM_JOBS})"
echo ""
echo "Useful commands:"
echo "  squeue -u \$USER                  # Check job status"
echo "  squeue -j ${JOB_ID}               # Check this job array"
echo "  scancel ${JOB_ID}                 # Cancel all jobs in array"
echo "  scancel ${JOB_ID}_5               # Cancel specific array task"
echo "  tail -f logs/gate_${JOB_ID}_1.out # Monitor job 1 output"
echo ""
echo "After completion:"
echo "  python merge_results.py           # Merge all results"
echo "============================================================"
