#!/usr/bin/env python3
"""
GATE 10 Simulation: CZT Detector with Tungsten Slit Collimator
Cluster-adapted version with realistic detector response

Features:
- Digitizer chain for energy deposition
- Energy blurring (2% FWHM at 662 keV for CZT)
- Captures Compton scattering events
- Full energy spectrum including Compton continuum
"""

import argparse
import os
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run CZT slit collimator simulation on cluster"
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--primaries", type=int, required=True)
    parser.add_argument("--job-id", type=int, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--threads", type=int, default=None)
    return parser.parse_args()


def get_num_threads(args):
    if args.threads is not None:
        return args.threads
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        return int(slurm_cpus)
    return os.cpu_count()


def run_simulation(args):
    import opengate as gate

    # Setup output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_output = args.output_dir if args.output_dir else os.path.join(script_dir, "output")
    job_output_dir = os.path.join(base_output, f"job_{args.job_id:04d}")
    os.makedirs(job_output_dir, exist_ok=True)

    print("=" * 60)
    print(f"GATE 10 Cluster Simulation - Job {args.job_id}")
    print("With Digitizer Chain & Energy Blurring")
    print("=" * 60)
    print(f"Seed: {args.seed}")
    print(f"Primaries: {args.primaries:,}")
    print(f"Output: {job_output_dir}")

    # Create simulation
    sim = gate.Simulation()
    sim.visu = False
    sim.random_seed = args.seed
    sim.number_of_threads = get_num_threads(args)
    sim.progress_bar = True

    print(f"Threads: {sim.number_of_threads}")

    # Units
    mm = gate.g4_units.mm
    cm = gate.g4_units.cm
    keV = gate.g4_units.keV

    # World volume
    world = sim.world
    world.size = [50 * cm, 50 * cm, 50 * cm]
    world.material = "G4_AIR"

    # Materials database
    materials_db = os.path.join(script_dir, "GateMaterials.db")
    if os.path.exists(materials_db):
        sim.volume_manager.add_material_database(materials_db)

    # CZT Detector
    detector_x = 5 * mm
    detector_y = 40 * mm
    detector_z = 40 * mm
    source_to_detector = 10 * cm

    detector = sim.add_volume("Box", name="czt_detector")
    detector.size = [detector_x, detector_y, detector_z]
    detector.translation = [0, 0, source_to_detector + detector_z / 2]
    detector.material = "CdZnTe"
    detector.color = [0.2, 0.6, 1.0, 0.8]

    # Tungsten Slit Collimator
    collimator_size = 50 * mm
    collimator_depth = 50 * mm
    slit_width = 0.2 * mm
    slit_length = 40 * mm
    collimator_z_center = 5 * cm

    collimator = sim.add_volume("Box", name="collimator_body")
    collimator.size = [collimator_size, collimator_size, collimator_depth]
    collimator.translation = [0, 0, collimator_z_center]
    collimator.material = "G4_W"
    collimator.color = [0.5, 0.5, 0.5, 1.0]

    slit = sim.add_volume("Box", name="slit")
    slit.mother = "collimator_body"
    slit.size = [slit_width, slit_length, collimator_depth]
    slit.translation = [0, 0, 0]
    slit.material = "G4_AIR"
    slit.color = [1.0, 1.0, 0.0, 1.0]

    # Physics - include all EM processes for Compton scattering
    sim.physics_manager.physics_list_name = "G4EmStandardPhysics_option4"
    sim.physics_manager.set_production_cut("world", "all", 1 * mm)
    sim.physics_manager.set_production_cut("czt_detector", "all", 0.1 * mm)
    sim.physics_manager.set_production_cut("collimator_body", "all", 0.1 * mm)
    sim.physics_manager.set_production_cut("slit", "all", 0.1 * mm)

    # Cs-137 Point Source
    source = sim.add_source("GenericSource", name="cs137_source")
    source.particle = "gamma"
    source.energy.type = "mono"
    source.energy.mono = 662 * keV
    source.position.type = "point"
    source.position.translation = [0, 0, 0]
    source.direction.type = "iso"
    source.n = args.primaries

    # =========================================================================
    # DIGITIZER CHAIN - Realistic detector response
    # =========================================================================
    
    # 1. Hits Collection: capture ALL energy deposits (steps) in detector
    #    This captures both photoelectric and Compton interactions
    hits = sim.add_actor("DigitizerHitsCollectionActor", name="hits")
    hits.attached_to = "czt_detector"
    hits.output_filename = os.path.join(job_output_dir, "hits.root")
    hits.attributes = [
        "TotalEnergyDeposit",
        "PostPosition",
        "PrePosition",
        "GlobalTime",
        "EventID",
        "TrackID",
        "PDGCode",
        "TrackCreatorProcess",
        "PreStepUniqueVolumeID",
    ]

    # 2. Adder: sum all energy deposits per event
    #    This gives total deposited energy (full absorption or partial)
    adder = sim.add_actor("DigitizerAdderActor", name="singles")
    adder.attached_to = "czt_detector"
    adder.input_digi_collection = "hits"
    adder.policy = "EnergyWinnerPosition"
    adder.output_filename = os.path.join(job_output_dir, "singles.root")

    # 3. Energy Blurring: simulate detector energy resolution
    #    CZT typically has ~2% FWHM at 662 keV
    blur = sim.add_actor("DigitizerBlurringActor", name="blurred")
    blur.attached_to = "czt_detector"
    blur.input_digi_collection = "singles"
    blur.blur_attribute = "TotalEnergyDeposit"
    blur.blur_method = "Gaussian"
    blur.blur_fwhm = 0.02  # 2% energy resolution
    blur.blur_reference_value = 662 * keV
    blur.output_filename = os.path.join(job_output_dir, "blurred.root")

    # Also keep phase space for position analysis
    phsp = sim.add_actor("PhaseSpaceActor", name="phsp_detector")
    phsp.attached_to = "czt_detector"
    phsp.output_filename = os.path.join(job_output_dir, "phsp_detector.root")
    phsp.attributes = [
        "KineticEnergy",
        "PrePosition",
        "PreDirection",
        "GlobalTime",
        "EventID",
        "TrackID",
    ]

    # Statistics
    stats = sim.add_actor("SimulationStatisticsActor", name="stats")
    stats.track_types_flag = True
    stats.output_filename = os.path.join(job_output_dir, "simulation_stats.txt")

    # Print geometry summary
    print(f"\nGeometry:")
    print(f"  - Detector: {detector_x/mm:.0f} x {detector_y/mm:.0f} x {detector_z/mm:.0f} mm CZT")
    print(f"  - Collimator: {collimator_size/mm:.0f} mm tungsten, {slit_width/mm:.1f} mm slit")
    print(f"  - Energy blurring: 2% FWHM at 662 keV")

    # Run simulation
    print(f"\nStarting simulation...")
    start_time = time.time()
    sim.run()
    elapsed = time.time() - start_time

    print(f"\n" + "=" * 60)
    print(f"Simulation Complete!")
    print(f"=" * 60)
    print(f"Job ID: {args.job_id}")
    print(f"Elapsed time: {elapsed:.1f} seconds")
    print(f"Primaries: {args.primaries:,}")
    print(f"Rate: {args.primaries/elapsed:.0f} primaries/second")
    print(f"Output: {job_output_dir}")

    # Write metadata
    with open(os.path.join(job_output_dir, "job_metadata.txt"), "w") as f:
        f.write(f"job_id={args.job_id}\n")
        f.write(f"seed={args.seed}\n")
        f.write(f"primaries={args.primaries}\n")
        f.write(f"threads={sim.number_of_threads}\n")
        f.write(f"elapsed_seconds={elapsed:.2f}\n")
        f.write(f"energy_blur_fwhm=0.02\n")

    return 0


if __name__ == "__main__":
    args = parse_args()
    sys.exit(run_simulation(args))
