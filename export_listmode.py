#!/usr/bin/env python3
"""
Export per-hit list-mode energy data from the CZT detector ROOT output.

The simulation attaches all digitizer actors to "czt_detector" only, so these
ROOT files already contain detector hits exclusively (no collimator data).

By default this reads the blurred singles file, whose TotalEnergyDeposit branch
is the per-event detected energy WITH realistic CZT energy resolution
(2% FWHM @ 662 keV). One CSV row = one detected single.

Usage:
    # Run on the merged-data artifact downloaded from a GitHub Actions run:
    python export_listmode.py --input blurred_merged.root --output listmode.csv

    # Single-job file works too:
    python export_listmode.py --input output/job_0001/blurred.root

    # Include time + position columns as well as energy:
    python export_listmode.py --input blurred_merged.root --full
"""

import argparse
import csv
import os
import sys

try:
    import numpy as np
except ImportError:
    print("Error: numpy not installed. Run: pip install numpy", file=sys.stderr)
    sys.exit(1)

try:
    import uproot
except ImportError:
    print("Error: uproot not installed. Run: pip install uproot", file=sys.stderr)
    sys.exit(1)


def load_branches(filename):
    """Load all branches from the first tree as flat numpy arrays."""
    print(f"Loading: {filename}")
    with uproot.open(filename) as f:
        keys = f.keys()
        if not keys:
            print(f"Error: no trees found in {filename}", file=sys.stderr)
            sys.exit(1)
        tree_name = keys[0].split(";")[0]
        tree = f[tree_name]
        print(f"Tree: {tree_name}  ({tree.num_entries} entries)")
        print(f"Branches: {tree.keys()}")

        data = {}
        for key in tree.keys():
            try:
                data[key] = np.asarray(tree[key].array(library="np")).flatten()
            except Exception:
                print(f"  Skipping non-numeric branch: {key}")
        return data


def to_keV(energy, unit):
    """Convert deposited-energy array to keV."""
    if unit == "keV":
        return energy
    if unit == "MeV":
        return energy * 1000.0
    # auto: GATE stores energy in MeV, so a 662 keV photopeak shows up as ~0.662.
    if len(energy) > 0 and energy.max() < 1.0:
        print("  Detected MeV units (max < 1) -> converting to keV")
        return energy * 1000.0
    return energy


def main():
    parser = argparse.ArgumentParser(
        description="Export detector list-mode energy data from GATE ROOT output"
    )
    parser.add_argument(
        "--input", default="output/merged/blurred_merged.root",
        help="Input ROOT file (default: output/merged/blurred_merged.root)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output CSV path (default: <input basename>_listmode.csv)",
    )
    parser.add_argument(
        "--energy-unit", choices=["auto", "MeV", "keV"], default="auto",
        help="Units of TotalEnergyDeposit in the file (default: auto-detect)",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Also export GlobalTime and position columns when available",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        print("Download the 'merged-data' artifact from the GitHub Actions run "
              "and point --input at blurred_merged.root.", file=sys.stderr)
        sys.exit(1)

    data = load_branches(args.input)

    if "TotalEnergyDeposit" not in data:
        print("Error: no 'TotalEnergyDeposit' branch found. This file may be a "
              "phase-space file (use blurred_merged.root or singles.root).",
              file=sys.stderr)
        sys.exit(1)

    energy = to_keV(data["TotalEnergyDeposit"].astype(float), args.energy_unit)
    n = len(energy)

    # Assemble columns: energy is always present; EventID if available.
    columns = []
    if "EventID" in data and len(data["EventID"]) == n:
        columns.append(("EventID", data["EventID"]))
    columns.append(("energy_keV", energy))

    if args.full:
        for name in ("GlobalTime", "PostPosition_X", "PostPosition_Y",
                     "PostPosition_Z", "PrePosition_X", "PrePosition_Y",
                     "PrePosition_Z"):
            if name in data and len(data[name]) == n:
                columns.append((name, data[name]))

    out_path = args.output
    if out_path is None:
        base = os.path.splitext(os.path.basename(args.input))[0]
        out_path = f"{base}_listmode.csv"

    header = [name for name, _ in columns]
    arrays = [arr for _, arr in columns]
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for row in zip(*arrays):
            writer.writerow(row)

    print(f"\nWrote {n:,} detector hits -> {out_path}")
    print(f"Columns: {', '.join(header)}")
    if n > 0:
        print(f"Energy: mean={energy.mean():.1f} keV, "
              f"min={energy.min():.1f}, max={energy.max():.1f} keV")


if __name__ == "__main__":
    main()
