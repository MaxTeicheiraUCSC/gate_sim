#!/bin/bash
# =============================================================================
# GATE Simulation Environment Setup for UCSC Hummingbird Cluster
# =============================================================================
# This script sets up the OpenGATE environment on the cluster.
# Run this ONCE after logging into the cluster for the first time.
#
# Usage:
#   bash setup_cluster.sh
#
# Prerequisites:
#   - SSH access to hb.ucsc.edu
#   - Sufficient storage quota in home directory
# =============================================================================

set -e

echo "============================================================"
echo "GATE Environment Setup for Hummingbird Cluster"
echo "============================================================"
echo ""

# Check if we're on the cluster
if [[ ! $(hostname) =~ hb|hummingbird ]]; then
    echo "Warning: This script is intended to run on the Hummingbird cluster."
    echo "Current host: $(hostname)"
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# =============================================================================
# Step 1: Check available modules
# =============================================================================
echo ""
echo "Step 1: Checking available modules..."
echo "--------------------------------------------------------------"

# Check for GATE/Geant4/CUDA modules
echo "Searching for GATE modules..."
module avail 2>&1 | grep -i gate || echo "  No GATE modules found"

echo ""
echo "Searching for Geant4 modules..."
module avail 2>&1 | grep -i geant || echo "  No Geant4 modules found"

echo ""
echo "Searching for CUDA modules..."
module avail 2>&1 | grep -i cuda || echo "  No CUDA modules found"

echo ""
echo "Searching for Python modules..."
module avail 2>&1 | grep -i python | head -5 || echo "  No Python modules found"

# =============================================================================
# Step 2: Check GPU availability
# =============================================================================
echo ""
echo "Step 2: Checking cluster partitions and GPUs..."
echo "--------------------------------------------------------------"

echo "Available partitions:"
sinfo -s 2>/dev/null || echo "  sinfo not available (might need to be on login node)"

echo ""
echo "GPU partitions:"
sinfo -p gpu -O "NodeList,Gres,CPUsState,Memory" 2>/dev/null || \
sinfo 2>/dev/null | grep -i gpu || \
echo "  Could not determine GPU partitions"

# =============================================================================
# Step 3: Set up Conda environment
# =============================================================================
echo ""
echo "Step 3: Setting up Conda environment..."
echo "--------------------------------------------------------------"

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "Conda found: $(which conda)"
elif [ -f "$HOME/miniconda3/bin/conda" ]; then
    echo "Found Miniconda at $HOME/miniconda3"
    export PATH="$HOME/miniconda3/bin:$PATH"
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/bin/conda" ]; then
    echo "Found Anaconda at $HOME/anaconda3"
    export PATH="$HOME/anaconda3/bin:$PATH"
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "Conda not found. Installing Miniconda..."

    # Download and install Miniconda
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    MINICONDA_SCRIPT="$HOME/miniconda_installer.sh"

    wget -q "$MINICONDA_URL" -O "$MINICONDA_SCRIPT"
    bash "$MINICONDA_SCRIPT" -b -p "$HOME/miniconda3"
    rm "$MINICONDA_SCRIPT"

    export PATH="$HOME/miniconda3/bin:$PATH"
    source "$HOME/miniconda3/etc/profile.d/conda.sh"

    # Initialize conda for bash
    conda init bash

    echo "Miniconda installed successfully!"
fi

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
# Create GATE environment
echo ""
echo "Creating gate_env conda environment..."

if conda env list | grep -q "gate_env"; then
    echo "Environment 'gate_env' already exists."
    read -p "Recreate it? [y/N] " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n gate_env -y
        CREATE_ENV=true
    else
        CREATE_ENV=false
    fi
else
    CREATE_ENV=true
fi

if [ "$CREATE_ENV" = true ]; then
    echo "Creating new environment..."
    conda create -n gate_env python=3.10 -y

    echo "Activating environment and installing packages..."
    conda activate gate_env

    # Install OpenGATE and dependencies
    pip install --upgrade pip
    pip install opengate
    pip install numpy scipy matplotlib
    pip install uproot awkward  # For ROOT file merging

    # Try to install SimpleITK for dose map handling
    pip install SimpleITK || echo "SimpleITK installation failed (optional)"

    echo ""
    echo "Environment setup complete!"
fi

# =============================================================================
# Step 4: Create simulation directory
# =============================================================================
echo ""
echo "Step 4: Setting up simulation directory..."
echo "--------------------------------------------------------------"

GATE_SIM_DIR="$HOME/gate_sim"
mkdir -p "$GATE_SIM_DIR"
mkdir -p "$GATE_SIM_DIR/logs"
mkdir -p "$GATE_SIM_DIR/output"

echo "Simulation directory: $GATE_SIM_DIR"

# =============================================================================
# Step 5: Verify installation
# =============================================================================
echo ""
echo "Step 5: Verifying OpenGATE installation..."
echo "--------------------------------------------------------------"

conda activate gate_env

echo "Python: $(which python)"
echo "Python version: $(python --version)"

echo ""
echo "Testing OpenGATE import..."
python -c "
import opengate as gate
print(f'OpenGATE version: {gate.__version__}')
print('OpenGATE imported successfully!')

# Test basic functionality
sim = gate.Simulation()
print('Simulation object created successfully!')
" || echo "OpenGATE verification failed!"

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "============================================================"
echo "Setup Summary"
echo "============================================================"
echo ""
echo "1. Conda environment: gate_env"
echo "   Activate with: conda activate gate_env"
echo ""
echo "2. Simulation directory: $GATE_SIM_DIR"
echo ""
echo "3. Next steps:"
echo "   a. Transfer simulation files:"
echo "      scp -r /path/to/local/files mteichei@hb.ucsc.edu:~/gate_sim/"
echo ""
echo "   b. Test single job:"
echo "      cd ~/gate_sim"
echo "      conda activate gate_env"
echo "      python czt_slit_simulation_cluster.py --seed 1 --primaries 100000 --job-id 1"
echo ""
echo "   c. Submit array job:"
echo "      ./submit_jobs.sh --test  # Test first"
echo "      ./submit_jobs.sh         # Full run"
echo ""
echo "============================================================"

# Add to bashrc for convenience
echo ""
read -p "Add conda activation to .bashrc? [y/N] " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "" >> ~/.bashrc
    echo "# GATE simulation environment" >> ~/.bashrc
    echo "alias gate_env='conda activate gate_env'" >> ~/.bashrc
    echo "export GATE_SIM_DIR=$HOME/gate_sim" >> ~/.bashrc
    echo "Added aliases to .bashrc"
fi

echo ""
echo "Setup complete!"
