# GATE CZT Slit Collimator Simulation

Monte Carlo simulation of Cs-137 gamma rays passing through a tungsten slit collimator onto a CdZnTe (CZT) semiconductor detector, built with [OpenGATE 10](https://github.com/OpenGATE/opengate) (GEANT4).

Designed for distributed execution on SLURM clusters with real-time Slack progress reporting.

## Geometry

```
Source (Cs-137, 662 keV)          Tungsten Collimator         CZT Detector
       *  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┤  ║  ├ ─ ─ ─ ─ ─ ─ ─ ─  ┌──────────┐
    (0,0,0)                       └──╨──┘                     │ 5×40×40  │
                                  50mm W                      │   mm     │
                                  0.2mm slit                  └──────────┘
                                  z = 50mm                    z = 120mm
```

| Component | Material | Dimensions |
|-----------|----------|------------|
| Source | Cs-137 (662 keV gammas, isotropic) | Point at origin |
| Collimator | Tungsten (19.3 g/cm³) | 50 × 50 × 50 mm |
| Slit | Air gap in collimator | 0.2 × 40 mm |
| Detector | CdZnTe (5.78 g/cm³) | 5 × 40 × 40 mm |

## Digitizer Chain

The simulation models realistic detector response:

1. **Hits Collection** — All energy deposits (photoelectric + Compton interactions)
2. **Adder** — Sums energy per event (full or partial absorption)
3. **Energy Blurring** — Gaussian blur, 2% FWHM at 662 keV (typical CZT resolution)
4. **Phase Space** — Records photon positions/directions at detector entrance

Physics list: `G4EmLivermorePhysics` with production cuts of 0.1 mm in detector/collimator regions.

## Quick Start

### Local (single job)

```bash
conda activate gate_env
python czt_slit_simulation_cluster.py --seed 1 --primaries 100000 --job-id 1
```

### Cluster (distributed)

```bash
# Test run: 100K primaries, 2 jobs
./submit_jobs.sh --test

# Production run: 100M primaries, 10 jobs
./submit_jobs.sh -n 100000000 -j 10
```

### Post-processing

```bash
python merge_results.py --output-dir output
python analyze_phsp.py --input output/merged/phsp_detector_merged.root
python diagnostic_3d_plot.py --input output/merged/phsp_detector_merged.root
```

## Project Structure

```
├── czt_slit_simulation_cluster.py   # Main GATE simulation (OpenGATE Python API)
├── merge_results.py                 # Merge ROOT files from distributed jobs
├── analyze_phsp.py                  # Energy spectra, hit maps, FWHM analysis
├── diagnostic_3d_plot.py            # 3D geometry + particle trajectory visualization
├── submit_jobs.sh                   # SLURM array job submission wrapper
├── submit_gate.sbatch               # Per-job SLURM batch script
├── merge_results.sbatch             # Post-processing pipeline (merge → analyze → Slack)
├── setup_cluster.sh                 # One-time cluster environment setup
├── transfer_to_cluster.sh           # SCP files to cluster
├── cluster.yaml                     # Autopilot pipeline configuration
├── GateMaterials.db                 # Custom material definitions (CdZnTe, W)
└── docs/                            # GitHub Pages results viewer
```

## Output Files

Each simulation job produces:

| File | Description |
|------|-------------|
| `hits.root` | All energy deposit steps in detector |
| `singles.root` | Summed energy per event |
| `blurred.root` | Energy-blurred singles (2% FWHM) |
| `phsp_detector.root` | Phase space at detector entrance |
| `simulation_stats.txt` | GATE run statistics |
| `job_metadata.txt` | Seed, primaries, threads, timing |

After merging and analysis:

| File | Description |
|------|-------------|
| `output/merged/phsp_detector_merged.root` | Combined phase space from all jobs |
| `output/analysis/energy_spectrum.png` | Energy histogram with 662 keV reference |
| `output/analysis/hit_map.png` | X-Y and X-Z detector surface maps |
| `output/analysis/x_distribution.png` | Slit projection with FWHM measurement |
| `output/analysis/analysis_summary.png` | 6-panel summary figure |
| `output/diagnostics/3d_full_geometry.png` | 3D wireframe geometry visualization |

## Pipeline

```
submit_jobs.sh
  └─ submit_gate.sbatch × N (parallel SLURM array)
       └─ czt_slit_simulation_cluster.py (per job)
            └─ output/job_XXXX/*.root

merge_results.sbatch (runs after all jobs complete)
  ├─ merge_results.py         → output/merged/
  ├─ diagnostic_3d_plot.py    → output/diagnostics/
  ├─ analyze_phsp.py          → output/analysis/
  └─ Slack notification with plots
```

## Cluster Setup (UCSC Hummingbird)

```bash
# One-time setup
bash setup_cluster.sh

# Transfer simulation files
bash transfer_to_cluster.sh
```

Creates a `gate_env` conda environment with: `opengate`, `numpy`, `scipy`, `matplotlib`, `uproot`, `awkward`, `SimpleITK`.

## Dependencies

- Python 3.10+
- [OpenGATE 10](https://github.com/OpenGATE/opengate) (bundles GEANT4)
- numpy, scipy, matplotlib
- uproot, awkward (ROOT file I/O)
- SimpleITK (optional, for dose maps)

## Results

View simulation results and analysis plots at: https://maxteicheiraucsc.github.io/gate_sim/
