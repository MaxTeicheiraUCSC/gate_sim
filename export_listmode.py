#!/usr/bin/env python3
"""
Export per-hit list-mode data from the CZT detector ROOT output.

GATE's digitizer output is already list-mode: each entry in the ROOT tree is
one detected single (energy, time, position, ...). This script is a format
converter -- it flattens that tree into a CSV for programs that don't read
ROOT. By default it keeps EVERY branch (nothing thrown away) and ADDS two
derived convenience columns: energy in keV and position in the CZT detector
frame.

By default it reads the blurred singles file, whose TotalEnergyDeposit branch
is the per-event detected energy WITH realistic CZT energy resolution
(2% FWHM @ 662 keV). One CSV row = one detected single.

Derived columns appended to the raw branches:
  - energy_keV       : TotalEnergyDeposit converted MeV -> keV.
  - x_mm/y_mm/z_mm   : detected position RELATIVE TO THE CZT DETECTOR CENTER.

Coordinate transform
---------------------
GATE writes positions in global/world coordinates (mm). The CZT detector is a
Box placed in the world with no rotation at translation [0, 0, 120] mm
(= source_to_detector + detector_z/2, from czt_slit_simulation_cluster.py).
Detector-frame coords are therefore global - detector_center. If the geometry
changes, pass the new center with --detector-center X Y Z (mm).

Usage
-----
    # Run on the merged-data artifact downloaded from a GitHub Actions run:
    python export_listmode.py --input blurred_merged.root --output listmode.csv

    # Single-job file works too:
    python export_listmode.py --input output/job_0001/blurred.root

    # Raw branches only, skip the derived energy_keV / x_mm columns:
    python export_listmode.py --input blurred_merged.root --no-derived

    # Override the detector center (mm) if the geometry changed:
    python export_listmode.py --input blurred_merged.root --detector-center 0 0 120
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


# Default CZT detector center in world coordinates (mm).
# From czt_slit_simulation_cluster.py: detector.translation =
# [0, 0, source_to_detector + detector_z/2] = [0, 0, 100 + 20] = [0, 0, 120].
DEFAULT_DETECTOR_CENTER = (0.0, 0.0, 120.0)


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


def find_position_prefix(data, n):
    """Pick the position branch family present in the file.

    Singles/blurred carry the winning hit's position (EnergyWinnerPosition
    policy); prefer that. Fall back to pre-step or a generic Position.
    """
    for prefix in ("PostPosition", "Position", "PrePosition"):
        comps = [f"{prefix}_{ax}" for ax in ("X", "Y", "Z")]
        if all(c in data and len(data[c]) == n for c in comps):
            return prefix, comps
    return None, None


def main():
    parser = argparse.ArgumentParser(
        description="Export full detector list-mode data from GATE ROOT output"
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
        "--detector-center", nargs=3, type=float, metavar=("X", "Y", "Z"),
        default=list(DEFAULT_DETECTOR_CENTER),
        help="CZT detector center in world mm; derived x/y/z_mm columns are "
             "reported relative to it (default: 0 0 120)",
    )
    parser.add_argument(
        "--no-derived", action="store_true",
        help="Export raw branches only; skip derived energy_keV and x/y/z_mm",
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

    n = len(data["TotalEnergyDeposit"])

    # 1. Keep ALL raw branches that line up with the entry count (one row per
    #    detected single). Branches of a different length (e.g. flattened
    #    vectors) are skipped so the CSV stays rectangular.
    columns = []
    for name, arr in data.items():
        if len(arr) == n:
            columns.append((name, arr))
        else:
            print(f"  Skipping branch (length {len(arr)} != {n}): {name}")

    # 2. Append derived convenience columns alongside the raw data.
    if not args.no_derived:
        energy_keV = to_keV(data["TotalEnergyDeposit"].astype(float),
                            args.energy_unit)
        columns.append(("energy_keV", energy_keV))

        cx, cy, cz = args.detector_center
        prefix, comps = find_position_prefix(data, n)
        if prefix is not None:
            print(f"  Derived x/y/z_mm from {prefix}_{{X,Y,Z}}; subtracting "
                  f"detector center ({cx}, {cy}, {cz}) mm")
            columns.append(("x_mm", data[comps[0]].astype(float) - cx))
            columns.append(("y_mm", data[comps[1]].astype(float) - cy))
            columns.append(("z_mm", data[comps[2]].astype(float) - cz))
        else:
            print("  Warning: no position branch (PostPosition/Position/"
                  "PrePosition) -> derived x/y/z_mm omitted")

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
    print(f"Columns ({len(header)}): {', '.join(header)}")
    if n > 0 and not args.no_derived:
        e = dict(columns)["energy_keV"]
        print(f"Energy: mean={e.mean():.1f} keV, "
              f"min={e.min():.1f}, max={e.max():.1f} keV")
        if "z_mm" in header:
            z = dict(columns)["z_mm"]
            print(f"z_mm (detector frame): min={z.min():.2f}, "
                  f"max={z.max():.2f}  (expect ~±2.5 mm, half-thickness)")


if __name__ == "__main__":
    main()
