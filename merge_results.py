#!/usr/bin/env python3
"""
Merge results from distributed GATE simulation jobs.

This script combines:
- ROOT phase space files from all jobs
- Dose maps (MHD/RAW format)
- Statistics from all jobs

Usage:
    python merge_results.py                          # Auto-detect jobs in ./output
    python merge_results.py --output-dir /path/to/output
    python merge_results.py --jobs 1,2,3,5          # Specific jobs only
"""

import argparse
import glob
import os
import re
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Merge distributed GATE simulation results")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Base output directory containing job_XXXX folders",
    )
    parser.add_argument(
        "--jobs",
        type=str,
        default=None,
        help="Comma-separated list of job IDs to merge (default: all)",
    )
    parser.add_argument(
        "--merged-dir",
        type=str,
        default=None,
        help="Directory for merged output (default: output/merged)",
    )
    parser.add_argument(
        "--skip-root",
        action="store_true",
        help="Skip ROOT file merging",
    )
    parser.add_argument(
        "--skip-dose",
        action="store_true",
        help="Skip dose map merging",
    )
    return parser.parse_args()


def find_job_directories(output_dir, job_ids=None):
    """Find all job output directories."""
    job_dirs = []
    pattern = os.path.join(output_dir, "job_*")

    for path in sorted(glob.glob(pattern)):
        if os.path.isdir(path):
            # Extract job ID from directory name
            match = re.search(r"job_(\d+)", os.path.basename(path))
            if match:
                job_id = int(match.group(1))
                if job_ids is None or job_id in job_ids:
                    job_dirs.append((job_id, path))

    return job_dirs


def merge_root_files(job_dirs, merged_dir):
    """Merge ROOT phase space files using hadd."""
    print("\n" + "=" * 60)
    print("Merging ROOT Phase Space Files")
    print("=" * 60)

    root_files = []
    for job_id, job_dir in job_dirs:
        phsp_file = os.path.join(job_dir, "phsp_detector.root")
        if os.path.exists(phsp_file):
            root_files.append(phsp_file)
            print(f"  Found: {phsp_file}")
        else:
            print(f"  Warning: Missing {phsp_file}")

    if not root_files:
        print("  No ROOT files found to merge!")
        return False

    merged_root = os.path.join(merged_dir, "phsp_detector_merged.root")

    # Try using hadd (ROOT's file merger)
    try:
        import subprocess
        cmd = ["hadd", "-f", merged_root] + root_files
        print(f"\n  Running: hadd -f {merged_root} ...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  Successfully merged {len(root_files)} files -> {merged_root}")
            return True
        else:
            print(f"  hadd failed: {result.stderr}")
    except FileNotFoundError:
        print("  'hadd' not found. Trying Python-based merge...")

    # Fallback: Python-based merge using uproot
    try:
        import uproot
        import awkward as ak

        print(f"\n  Using uproot to merge {len(root_files)} files...")

        # Read all files and concatenate
        all_data = {}
        tree_name = None

        for i, rf in enumerate(root_files):
            print(f"    Reading {i+1}/{len(root_files)}: {os.path.basename(rf)}")
            with uproot.open(rf) as f:
                # Get the tree name (usually "phsp_detector" or similar)
                if tree_name is None:
                    tree_name = list(f.keys())[0].split(";")[0]

                tree = f[tree_name]
                arrays = tree.arrays()

                for key in arrays.fields:
                    if key not in all_data:
                        all_data[key] = []
                    all_data[key].append(arrays[key])

        # Concatenate all arrays
        merged_data = {key: ak.concatenate(vals) for key, vals in all_data.items()}

        # Write merged file
        print(f"  Writing merged file: {merged_root}")
        with uproot.recreate(merged_root) as f:
            f[tree_name] = merged_data

        print(f"  Successfully merged {len(root_files)} files -> {merged_root}")
        return True

    except ImportError:
        print("  Error: Neither 'hadd' nor 'uproot' available for ROOT merging.")
        print("  Install uproot: pip install uproot awkward")
        return False
    except Exception as e:
        print(f"  Error merging ROOT files: {e}")
        return False


def merge_dose_maps(job_dirs, merged_dir):
    """Merge dose maps by summing voxel values."""
    print("\n" + "=" * 60)
    print("Merging Dose Maps")
    print("=" * 60)

    try:
        import numpy as np
    except ImportError:
        print("  Error: numpy required for dose map merging")
        return False

    # Look for dose files
    dose_files = []
    for job_id, job_dir in job_dirs:
        # GATE outputs dose as .mhd (header) + .raw (data)
        mhd_file = os.path.join(job_dir, "dose_detector-edep.mhd")
        if not os.path.exists(mhd_file):
            mhd_file = os.path.join(job_dir, "dose_detector.mhd")

        if os.path.exists(mhd_file):
            dose_files.append(mhd_file)
            print(f"  Found: {mhd_file}")
        else:
            print(f"  Warning: No dose file in {job_dir}")

    if not dose_files:
        print("  No dose files found — skipping dose merge (simulation may not produce dose maps)")
        return True

    # Try using SimpleITK for MHD files
    try:
        import SimpleITK as sitk

        print(f"\n  Using SimpleITK to merge {len(dose_files)} dose maps...")

        # Read first image for reference
        ref_image = sitk.ReadImage(dose_files[0])
        merged_array = sitk.GetArrayFromImage(ref_image).astype(np.float64)

        # Sum all dose maps
        for i, df in enumerate(dose_files[1:], 2):
            print(f"    Adding {i}/{len(dose_files)}: {os.path.basename(df)}")
            img = sitk.ReadImage(df)
            merged_array += sitk.GetArrayFromImage(img).astype(np.float64)

        # Create merged image
        merged_image = sitk.GetImageFromArray(merged_array.astype(np.float32))
        merged_image.CopyInformation(ref_image)

        # Write merged dose map
        merged_mhd = os.path.join(merged_dir, "dose_detector_merged.mhd")
        sitk.WriteImage(merged_image, merged_mhd)

        print(f"  Successfully merged {len(dose_files)} dose maps -> {merged_mhd}")
        print(f"  Total dose sum: {merged_array.sum():.6e}")
        return True

    except ImportError:
        print("  SimpleITK not available, trying manual MHD parsing...")

    # Fallback: Manual MHD/RAW parsing
    try:
        merged_array = None
        header_info = None

        for i, mhd_file in enumerate(dose_files):
            print(f"    Reading {i+1}/{len(dose_files)}: {os.path.basename(mhd_file)}")

            # Parse MHD header
            header = {}
            with open(mhd_file, 'r') as f:
                for line in f:
                    if '=' in line:
                        key, value = line.strip().split('=', 1)
                        header[key.strip()] = value.strip()

            if header_info is None:
                header_info = header

            # Read RAW data
            raw_file = os.path.join(os.path.dirname(mhd_file), header.get('ElementDataFile', ''))
            if not os.path.exists(raw_file):
                # Try same name with .raw extension
                raw_file = mhd_file.replace('.mhd', '.raw')

            # Determine data type
            dtype_map = {
                'MET_FLOAT': np.float32,
                'MET_DOUBLE': np.float64,
                'MET_SHORT': np.int16,
                'MET_INT': np.int32,
            }
            dtype = dtype_map.get(header.get('ElementType', 'MET_FLOAT'), np.float32)

            # Read and reshape
            data = np.fromfile(raw_file, dtype=dtype)
            dims = [int(x) for x in header.get('DimSize', '1 1 1').split()]
            data = data.reshape(dims[::-1])  # MHD uses x,y,z but numpy uses z,y,x

            if merged_array is None:
                merged_array = data.astype(np.float64)
            else:
                merged_array += data.astype(np.float64)

        # Write merged output
        merged_mhd = os.path.join(merged_dir, "dose_detector_merged.mhd")
        merged_raw = os.path.join(merged_dir, "dose_detector_merged.raw")

        # Write RAW data
        merged_array.astype(np.float32).tofile(merged_raw)

        # Write MHD header
        with open(merged_mhd, 'w') as f:
            f.write("ObjectType = Image\n")
            f.write(f"NDims = {header_info.get('NDims', '3')}\n")
            f.write(f"DimSize = {header_info.get('DimSize', '5 40 40')}\n")
            f.write(f"ElementSpacing = {header_info.get('ElementSpacing', '1 1 1')}\n")
            f.write(f"Offset = {header_info.get('Offset', '0 0 0')}\n")
            f.write("ElementType = MET_FLOAT\n")
            f.write(f"ElementDataFile = {os.path.basename(merged_raw)}\n")

        print(f"  Successfully merged {len(dose_files)} dose maps -> {merged_mhd}")
        print(f"  Total dose sum: {merged_array.sum():.6e}")
        return True

    except Exception as e:
        print(f"  Error merging dose maps: {e}")
        return False


def aggregate_statistics(job_dirs, merged_dir):
    """Aggregate simulation statistics from all jobs."""
    print("\n" + "=" * 60)
    print("Aggregating Statistics")
    print("=" * 60)

    total_primaries = 0
    total_tracks = 0
    total_steps = 0
    total_time = 0.0
    job_count = 0

    for job_id, job_dir in job_dirs:
        # Check for stats file
        stats_file = os.path.join(job_dir, "simulation_stats.txt")
        metadata_file = os.path.join(job_dir, "job_metadata.txt")

        if os.path.exists(stats_file):
            print(f"  Reading: {stats_file}")
            with open(stats_file, 'r') as f:
                content = f.read()
                # Parse GATE stats format (varies by version)
                for line in content.split('\n'):
                    if 'NumberOfEvents' in line or 'primaries' in line.lower():
                        try:
                            num = int(re.search(r'(\d+)', line).group(1))
                            total_primaries += num
                        except:
                            pass
                    if 'Tracks' in line:
                        try:
                            num = int(re.search(r'(\d+)', line).group(1))
                            total_tracks += num
                        except:
                            pass
            job_count += 1

        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                for line in f:
                    if line.startswith('elapsed_seconds='):
                        try:
                            total_time += float(line.split('=')[1])
                        except:
                            pass
                    if line.startswith('primaries='):
                        try:
                            # Use metadata primaries if stats parsing failed
                            if total_primaries == 0:
                                total_primaries += int(line.split('=')[1])
                        except:
                            pass

    # Write merged statistics
    merged_stats = os.path.join(merged_dir, "merged_statistics.txt")
    with open(merged_stats, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Merged Simulation Statistics\n")
        f.write("=" * 60 + "\n")
        f.write(f"Jobs merged: {job_count}\n")
        f.write(f"Total primaries: {total_primaries:,}\n")
        f.write(f"Total tracks: {total_tracks:,}\n")
        f.write(f"Total CPU time: {total_time:.1f} seconds ({total_time/3600:.2f} hours)\n")
        if total_time > 0:
            f.write(f"Average rate: {total_primaries/total_time:.0f} primaries/second\n")
        f.write("=" * 60 + "\n")

    print(f"\n  Summary:")
    print(f"    Jobs merged: {job_count}")
    print(f"    Total primaries: {total_primaries:,}")
    print(f"    Total CPU time: {total_time:.1f} seconds")
    print(f"  Written: {merged_stats}")

    return True


def main():
    args = parse_args()

    print("=" * 60)
    print("GATE Simulation Results Merger")
    print("=" * 60)

    # Parse job IDs if specified
    job_ids = None
    if args.jobs:
        job_ids = set(int(x) for x in args.jobs.split(','))
        print(f"Merging specific jobs: {sorted(job_ids)}")

    # Find job directories
    job_dirs = find_job_directories(args.output_dir, job_ids)

    if not job_dirs:
        print(f"No job directories found in {args.output_dir}")
        print("Expected format: output/job_0001/, output/job_0002/, etc.")
        return 1

    print(f"\nFound {len(job_dirs)} job directories:")
    for job_id, path in job_dirs:
        print(f"  Job {job_id}: {path}")

    # Create merged output directory
    merged_dir = args.merged_dir or os.path.join(args.output_dir, "merged")
    os.makedirs(merged_dir, exist_ok=True)
    print(f"\nMerged output directory: {merged_dir}")

    success = True

    # Merge ROOT files
    if not args.skip_root:
        if not merge_root_files(job_dirs, merged_dir):
            success = False

    # Merge dose maps
    if not args.skip_dose:
        if not merge_dose_maps(job_dirs, merged_dir):
            success = False

    # Aggregate statistics
    aggregate_statistics(job_dirs, merged_dir)

    print("\n" + "=" * 60)
    if success:
        print("Merge completed successfully!")
    else:
        print("Merge completed with some errors (see above)")
    print("=" * 60)
    print(f"Merged files are in: {merged_dir}")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
