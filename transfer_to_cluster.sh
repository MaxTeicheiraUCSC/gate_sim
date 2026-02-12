#!/bin/bash
# =============================================================================
# Transfer simulation files to UCSC Hummingbird Cluster
# =============================================================================
# Usage: ./transfer_to_cluster.sh
# =============================================================================

set -e

# Configuration
CLUSTER_USER="mteichei"
CLUSTER_HOST="hb.ucsc.edu"
REMOTE_DIR="~/gate_sim"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo "Transfer GATE Simulation Files to Hummingbird"
echo "============================================================"
echo ""
echo "Local directory:  $SCRIPT_DIR"
echo "Remote:           ${CLUSTER_USER}@${CLUSTER_HOST}:${REMOTE_DIR}"
echo ""

# Files to transfer
FILES=(
    "czt_slit_simulation_cluster.py"
    "submit_gate.sbatch"
    "submit_jobs.sh"
    "merge_results.py"
    "setup_cluster.sh"
    "GateMaterials.db"
)

echo "Files to transfer:"
for f in "${FILES[@]}"; do
    if [ -f "$SCRIPT_DIR/$f" ]; then
        echo "  [OK] $f"
    else
        echo "  [MISSING] $f"
    fi
done

echo ""
read -p "Proceed with transfer? [y/N] " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Create remote directory
echo ""
echo "Creating remote directory..."
ssh ${CLUSTER_USER}@${CLUSTER_HOST} "mkdir -p ${REMOTE_DIR}/logs ${REMOTE_DIR}/output"

# Transfer files
echo "Transferring files..."
for f in "${FILES[@]}"; do
    if [ -f "$SCRIPT_DIR/$f" ]; then
        echo "  Transferring $f..."
        scp "$SCRIPT_DIR/$f" ${CLUSTER_USER}@${CLUSTER_HOST}:${REMOTE_DIR}/
    fi
done

echo ""
echo "============================================================"
echo "Transfer complete!"
echo "============================================================"
echo ""
echo "Next steps on the cluster:"
echo "  1. SSH to cluster:  ssh ${CLUSTER_USER}@${CLUSTER_HOST}"
echo "  2. Navigate:        cd ${REMOTE_DIR}"
echo "  3. Setup env:       bash setup_cluster.sh"
echo "  4. Test job:        ./submit_jobs.sh --test"
echo "  5. Full run:        ./submit_jobs.sh"
echo ""
