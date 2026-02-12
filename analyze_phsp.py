#!/usr/bin/env python3
"""
Analysis script for CZT Slit Collimator Phase Space Data

Generates visualizations:
1. Energy spectrum
2. X-Y position distribution (hit map)
3. X position histogram with FWHM
4. Z depth distribution
5. Angular distribution of incoming particles

Usage:
    python analyze_phsp.py                          # Use default merged file
    python analyze_phsp.py --input path/to/file.root
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for cluster
import matplotlib.pyplot as plt
import numpy as np

try:
    import uproot
except ImportError:
    print("Error: uproot not installed. Run: pip install uproot")
    sys.exit(1)

try:
    from scipy.stats import gaussian_kde
    from scipy.optimize import curve_fit
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available, some analyses will be limited")


def gaussian(x, amp, mu, sigma):
    """Gaussian function for fitting."""
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def calc_fwhm(data, bins=100):
    """Calculate FWHM from histogram data."""
    hist, edges = np.histogram(data, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2

    max_val = hist.max()
    half_max = max_val / 2
    max_idx = np.argmax(hist)

    # Find left and right crossings
    left_idx = 0
    for i in range(max_idx, -1, -1):
        if hist[i] < half_max:
            left_idx = i
            break

    right_idx = len(hist) - 1
    for i in range(max_idx, len(hist)):
        if hist[i] < half_max:
            right_idx = i
            break

    fwhm = centers[right_idx] - centers[left_idx]
    return fwhm, centers[left_idx], centers[right_idx]


def load_data(filename):
    """Load phase space data from ROOT file."""
    print(f"Loading: {filename}")

    with uproot.open(filename) as f:
        # Get the first tree
        tree_name = [k for k in f.keys() if not k.endswith(';1') or ';' not in k][0]
        tree = f[tree_name]

        print(f"Tree: {tree_name}")
        print(f"Available branches: {tree.keys()}")

        data = {}
        for key in tree.keys():
            try:
                data[key] = tree[key].array(library="np")
            except:
                pass

        return data


def plot_energy_spectrum(data, output_dir):
    """Plot energy spectrum."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Energy in keV
    if 'KineticEnergy' in data:
        energy = data['KineticEnergy'] * 1000  # Convert to keV if in MeV
        if energy.max() > 1000:  # Already in keV
            energy = data['KineticEnergy']
    else:
        print("Warning: No energy data found")
        return None

    # Histogram
    bins = np.linspace(0, 700, 141)
    counts, edges, _ = ax.hist(energy, bins=bins, color='steelblue',
                                edgecolor='black', linewidth=0.3, alpha=0.7)

    ax.set_xlabel('Energy (keV)', fontsize=12)
    ax.set_ylabel('Counts', fontsize=12)
    ax.set_title('Energy Spectrum at CZT Detector\n(Cs-137: 662 keV)', fontsize=14)
    ax.set_xlim(0, 700)
    ax.grid(True, alpha=0.3)

    # Add photopeak annotation
    peak_idx = np.argmax(counts[bins[:-1] > 600])
    peak_energy = bins[:-1][bins[:-1] > 600][peak_idx]
    ax.axvline(662, color='red', linestyle='--', linewidth=1.5, label='662 keV (Cs-137)')
    ax.legend(fontsize=10)

    # Statistics
    stats_text = f'Total hits: {len(energy):,}\nMean: {energy.mean():.1f} keV\nStd: {energy.std():.1f} keV'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    outfile = os.path.join(output_dir, 'energy_spectrum.png')
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"Saved: {outfile}")
    plt.close()
    return outfile


def plot_hit_map(data, output_dir):
    """Plot 2D hit position map."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Get positions
    x = data.get('PrePosition_X', data.get('PostPosition_X', None))
    y = data.get('PrePosition_Y', data.get('PostPosition_Y', None))
    z = data.get('PrePosition_Z', data.get('PostPosition_Z', None))

    if x is None or y is None:
        print("Warning: No position data found")
        return None

    # X-Y map
    ax = axes[0]
    h = ax.hist2d(x, y, bins=[100, 100], cmap='hot')
    plt.colorbar(h[3], ax=ax, label='Counts')
    ax.set_xlabel('X Position (mm)', fontsize=12)
    ax.set_ylabel('Y Position (mm)', fontsize=12)
    ax.set_title('Hit Map (X-Y)', fontsize=14)
    ax.set_aspect('equal')

    # X-Z map
    ax = axes[1]
    if z is not None:
        h = ax.hist2d(x, z, bins=[100, 100], cmap='hot')
        plt.colorbar(h[3], ax=ax, label='Counts')
        ax.set_xlabel('X Position (mm)', fontsize=12)
        ax.set_ylabel('Z Position (mm)', fontsize=12)
        ax.set_title('Hit Map (X-Z)', fontsize=14)

    plt.suptitle('Detector Hit Maps - CZT Slit Collimator', fontsize=14, y=1.02)
    plt.tight_layout()
    outfile = os.path.join(output_dir, 'hit_map.png')
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"Saved: {outfile}")
    plt.close()
    return outfile


def plot_x_distribution(data, output_dir):
    """Plot X position distribution with FWHM calculation."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = data.get('PrePosition_X', data.get('PostPosition_X', None))
    if x is None:
        print("Warning: No X position data found")
        return None

    # Histogram
    bins = np.linspace(-3, 3, 121)
    counts, edges, _ = ax.hist(x, bins=bins, color='steelblue',
                                edgecolor='black', linewidth=0.3, alpha=0.7)
    centers = (edges[:-1] + edges[1:]) / 2

    # Calculate FWHM
    fwhm, x_left, x_right = calc_fwhm(x, bins=121)

    # Draw FWHM lines
    max_count = counts.max()
    ax.axvline(x_left, color='red', linestyle='-', linewidth=2, label=f'FWHM = {fwhm:.3f} mm')
    ax.axvline(x_right, color='red', linestyle='-', linewidth=2)
    ax.hlines(max_count/2, x_left, x_right, color='red', linestyle='--', linewidth=1.5)

    # Slit width reference
    ax.axvline(-0.1, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.axvline(0.1, color='green', linestyle=':', linewidth=1.5, alpha=0.7, label='Slit width (0.2 mm)')

    ax.set_xlabel('X Position (mm)', fontsize=12)
    ax.set_ylabel('Counts', fontsize=12)
    ax.set_title('X Position Distribution at Detector\nSlit Projection Analysis', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Statistics
    stats_text = f'Total hits: {len(x):,}\nMean: {x.mean():.4f} mm\nStd: {x.std():.4f} mm\nFWHM: {fwhm:.4f} mm'
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    outfile = os.path.join(output_dir, 'x_distribution.png')
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"Saved: {outfile}")
    plt.close()
    return outfile


def plot_z_depth(data, output_dir):
    """Plot Z depth distribution in detector."""
    fig, ax = plt.subplots(figsize=(10, 6))

    z = data.get('PrePosition_Z', data.get('PostPosition_Z', None))
    if z is None:
        print("Warning: No Z position data found")
        return None

    # Histogram
    counts, edges, _ = ax.hist(z, bins=80, color='steelblue',
                                edgecolor='black', linewidth=0.3, alpha=0.7)

    ax.set_xlabel('Z Position (mm)', fontsize=12)
    ax.set_ylabel('Counts', fontsize=12)
    ax.set_title('Interaction Depth in CZT Detector', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Mark detector boundaries
    z_min, z_max = z.min(), z.max()
    ax.axvline(z_min, color='red', linestyle='--', linewidth=1.5, label=f'Front face: {z_min:.1f} mm')
    ax.axvline(z_max, color='orange', linestyle='--', linewidth=1.5, label=f'Back face: {z_max:.1f} mm')
    ax.legend(fontsize=10)

    plt.tight_layout()
    outfile = os.path.join(output_dir, 'z_depth.png')
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"Saved: {outfile}")
    plt.close()
    return outfile


def plot_summary(data, output_dir):
    """Create a summary figure with multiple panels."""
    fig = plt.figure(figsize=(16, 12))

    # Get data
    energy = data.get('KineticEnergy', np.array([]))
    if len(energy) > 0 and energy.max() < 1:
        energy = energy * 1000  # Convert MeV to keV

    x = data.get('PrePosition_X', data.get('PostPosition_X', np.array([])))
    y = data.get('PrePosition_Y', data.get('PostPosition_Y', np.array([])))
    z = data.get('PrePosition_Z', data.get('PostPosition_Z', np.array([])))

    # 1. Energy spectrum
    ax1 = fig.add_subplot(2, 3, 1)
    if len(energy) > 0:
        ax1.hist(energy, bins=100, color='steelblue', edgecolor='black', linewidth=0.3, alpha=0.7)
        ax1.axvline(662, color='red', linestyle='--', label='662 keV')
        ax1.set_xlabel('Energy (keV)')
        ax1.set_ylabel('Counts')
        ax1.set_title('Energy Spectrum')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # 2. X-Y hit map
    ax2 = fig.add_subplot(2, 3, 2)
    if len(x) > 0 and len(y) > 0:
        h = ax2.hist2d(x, y, bins=50, cmap='hot')
        plt.colorbar(h[3], ax=ax2)
        ax2.set_xlabel('X (mm)')
        ax2.set_ylabel('Y (mm)')
        ax2.set_title('Hit Map (X-Y)')

    # 3. X distribution
    ax3 = fig.add_subplot(2, 3, 3)
    if len(x) > 0:
        ax3.hist(x, bins=100, color='steelblue', edgecolor='black', linewidth=0.3, alpha=0.7)
        fwhm, x_left, x_right = calc_fwhm(x)
        ax3.axvline(x_left, color='red', linestyle='-', linewidth=2)
        ax3.axvline(x_right, color='red', linestyle='-', linewidth=2)
        ax3.set_xlabel('X Position (mm)')
        ax3.set_ylabel('Counts')
        ax3.set_title(f'X Distribution (FWHM={fwhm:.3f} mm)')
        ax3.grid(True, alpha=0.3)

    # 4. Y distribution
    ax4 = fig.add_subplot(2, 3, 4)
    if len(y) > 0:
        ax4.hist(y, bins=100, color='green', edgecolor='black', linewidth=0.3, alpha=0.7)
        ax4.set_xlabel('Y Position (mm)')
        ax4.set_ylabel('Counts')
        ax4.set_title('Y Distribution')
        ax4.grid(True, alpha=0.3)

    # 5. Z depth
    ax5 = fig.add_subplot(2, 3, 5)
    if len(z) > 0:
        ax5.hist(z, bins=80, color='orange', edgecolor='black', linewidth=0.3, alpha=0.7)
        ax5.set_xlabel('Z Position (mm)')
        ax5.set_ylabel('Counts')
        ax5.set_title('Z Depth Distribution')
        ax5.grid(True, alpha=0.3)

    # 6. Statistics text
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    stats_text = "Simulation Statistics\n" + "="*30 + "\n\n"
    stats_text += f"Total hits: {len(x):,}\n\n"
    if len(energy) > 0:
        stats_text += f"Energy:\n  Mean: {energy.mean():.1f} keV\n  Std: {energy.std():.1f} keV\n\n"
    if len(x) > 0:
        fwhm, _, _ = calc_fwhm(x)
        stats_text += f"X Position:\n  Mean: {x.mean():.4f} mm\n  Std: {x.std():.4f} mm\n  FWHM: {fwhm:.4f} mm\n\n"
    if len(y) > 0:
        stats_text += f"Y Position:\n  Mean: {y.mean():.4f} mm\n  Std: {y.std():.4f} mm\n\n"

    stats_text += "Geometry:\n"
    stats_text += "  Slit width: 0.2 mm\n"
    stats_text += "  Source: Cs-137 (662 keV)\n"
    stats_text += "  Detector: 5x40x40 mm CZT"

    ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('CZT Slit Collimator Simulation - Summary', fontsize=16, y=1.02)
    plt.tight_layout()

    outfile = os.path.join(output_dir, 'analysis_summary.png')
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"Saved: {outfile}")
    plt.close()
    return outfile


def main():
    parser = argparse.ArgumentParser(description='Analyze GATE phase space data')
    parser.add_argument('--input', type=str, default='output/merged/phsp_detector_merged.root',
                        help='Input ROOT file')
    parser.add_argument('--output-dir', type=str, default='output/analysis',
                        help='Output directory for plots')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("CZT Slit Collimator Phase Space Analysis")
    print("=" * 60)

    # Load data
    data = load_data(args.input)
    print(f"\nLoaded {len(data.get('KineticEnergy', []))} entries")

    # Generate plots
    print("\nGenerating visualizations...")

    plot_energy_spectrum(data, args.output_dir)
    plot_hit_map(data, args.output_dir)
    plot_x_distribution(data, args.output_dir)
    plot_z_depth(data, args.output_dir)
    plot_summary(data, args.output_dir)

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Output files in: {args.output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
