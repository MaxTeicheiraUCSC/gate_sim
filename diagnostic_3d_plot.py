#!/usr/bin/env python3
"""
3D Diagnostic Plot of Photon Phase Space at CZT Detector
Reads merged phase space data and produces:
1. 3D scatter plot of arrival positions colored by kinetic energy
2. 2D hit map (X-Y) at detector surface
3. Energy spectrum + arrival statistics
4. 3D view with geometry overlays (detector + collimator outlines)
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

try:
    import uproot
except ImportError:
    print("Error: uproot not installed")
    sys.exit(1)


def load_phsp(filename):
    """Load phase space data from merged ROOT file."""
    print(f"Loading: {filename}")
    with uproot.open(filename) as f:
        tree_name = list(f.keys())[0].split(";")[0]
        tree = f[tree_name]
        print(f"Tree: {tree_name}, Entries: {tree.num_entries}")
        print(f"Branches: {tree.keys()}")

        data = {}
        for key in tree.keys():
            try:
                arr = tree[key].array(library="np")
                data[key] = np.asarray(arr).flatten()
            except Exception:
                try:
                    import awkward as ak
                    arr = tree[key].array()
                    data[key] = ak.to_numpy(arr).flatten()
                except Exception:
                    print(f"  Skipping branch: {key}")
        return data


def plot_3d_energy(data, output_dir):
    """3D scatter of arrival positions colored by kinetic energy."""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    x = data['PrePosition_X']
    y = data['PrePosition_Y']
    z = data['PrePosition_Z']
    energy = data['KineticEnergy']  # MeV

    # Convert to keV for display
    energy_keV = energy * 1000

    sc = ax.scatter(x, y, z, c=energy_keV, cmap='plasma', s=20, alpha=0.7,
                    vmin=0, vmax=min(700, energy_keV.max()))

    fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1, label='Kinetic Energy (keV)')

    ax.set_xlabel('X (mm)', fontsize=11)
    ax.set_ylabel('Y (mm)', fontsize=11)
    ax.set_zlabel('Z (mm)', fontsize=11)
    ax.set_title('Photon Arrival at CZT Detector\nColored by Kinetic Energy',
                 fontsize=14, pad=20)

    stats = (f"Total photons: {len(x):,}\n"
             f"Mean energy: {energy_keV.mean():.1f} keV\n"
             f"Max energy: {energy_keV.max():.1f} keV")
    fig.text(0.02, 0.02, stats, fontsize=10, fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    ax.view_init(elev=25, azim=135)
    plt.tight_layout()
    outfile = os.path.join(output_dir, '3d_arrivals_energy.png')
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"Saved: {outfile}")
    plt.close()
    return outfile


def plot_hitmap(data, output_dir):
    """2D hit map of photon arrival positions at detector surface."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    x = data['PrePosition_X']
    y = data['PrePosition_Y']

    # Scatter plot
    ax1 = axes[0]
    ax1.scatter(x, y, s=15, alpha=0.6, c='steelblue', edgecolors='navy', linewidths=0.3)
    ax1.set_xlabel('X (mm)', fontsize=11)
    ax1.set_ylabel('Y (mm)', fontsize=11)
    ax1.set_title('Photon Arrival Positions (scatter)', fontsize=13)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # 2D histogram
    ax2 = axes[1]
    if len(x) > 1:
        h = ax2.hist2d(x, y, bins=[40, 40], cmap='hot')
        plt.colorbar(h[3], ax=ax2, label='Counts')
    else:
        ax2.scatter(x, y, s=30, c='red')
    ax2.set_xlabel('X (mm)', fontsize=11)
    ax2.set_ylabel('Y (mm)', fontsize=11)
    ax2.set_title('Hit Map (X-Y projection)', fontsize=13)
    ax2.set_aspect('equal')

    plt.suptitle(f'Detector Surface Hit Map — {len(x):,} photons', fontsize=15, y=1.02)
    plt.tight_layout()
    outfile = os.path.join(output_dir, 'hitmap_detector.png')
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"Saved: {outfile}")
    plt.close()
    return outfile


def plot_3d_with_geometry(data, output_dir):
    """3D scatter with detector and collimator geometry outlines."""
    fig = plt.figure(figsize=(16, 11))
    ax = fig.add_subplot(111, projection='3d')

    x = data['PrePosition_X']
    y = data['PrePosition_Y']
    z = data['PrePosition_Z']
    energy = data['KineticEnergy'] * 1000  # keV

    # Plot arrivals
    sc = ax.scatter(x, y, z, c=energy, cmap='hot', s=15, alpha=0.6,
                    vmin=0, vmax=min(700, energy.max()))
    fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.08, label='Energy (keV)')

    # Draw detector outline (5mm x 40mm x 40mm at z=100+20=120mm center)
    det_x, det_y, det_z = 5.0, 40.0, 40.0
    det_center_z = 100.0 + det_z / 2

    def draw_box(ax, cx, cy, cz, sx, sy, sz, color, label):
        """Draw wireframe box."""
        hx, hy, hz = sx/2, sy/2, sz/2
        corners = np.array([
            [cx-hx, cy-hy, cz-hz], [cx+hx, cy-hy, cz-hz],
            [cx+hx, cy+hy, cz-hz], [cx-hx, cy+hy, cz-hz],
            [cx-hx, cy-hy, cz+hz], [cx+hx, cy-hy, cz+hz],
            [cx+hx, cy+hy, cz+hz], [cx-hx, cy+hy, cz+hz],
        ])
        edges = [
            [0,1],[1,2],[2,3],[3,0],
            [4,5],[5,6],[6,7],[7,4],
            [0,4],[1,5],[2,6],[3,7],
        ]
        for i, (a, b) in enumerate(edges):
            ax.plot3D(*zip(corners[a], corners[b]),
                      color=color, linewidth=1.5, alpha=0.7,
                      label=label if i == 0 else None)

    # Detector box
    draw_box(ax, 0, 0, det_center_z, det_x, det_y, det_z, 'cyan', 'CZT Detector')

    # Collimator box (50mm x 50mm x 50mm at z=50mm)
    draw_box(ax, 0, 0, 50.0, 50.0, 50.0, 50.0, 'gray', 'W Collimator')

    # Source point
    ax.scatter([0], [0], [0], c='red', s=100, marker='*', zorder=10, label='Cs-137 Source')

    # Slit opening line
    ax.plot3D([-0.1, 0.1], [0, 0], [50, 50], color='yellow', linewidth=3,
              alpha=0.8, label='Slit (0.2 mm)')

    ax.set_xlabel('X (mm)', fontsize=11)
    ax.set_ylabel('Y (mm)', fontsize=11)
    ax.set_zlabel('Z (mm)', fontsize=11)
    ax.set_title('CZT Slit Collimator Simulation — Full Geometry\n'
                 'Cs-137 source → W collimator → CZT detector',
                 fontsize=14, pad=20)
    ax.legend(loc='upper right', fontsize=9)

    ax.view_init(elev=20, azim=150)
    plt.tight_layout()
    outfile = os.path.join(output_dir, '3d_full_geometry.png')
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"Saved: {outfile}")
    plt.close()
    return outfile


def plot_diagnostic_summary(data, output_dir):
    """Multi-panel diagnostic: energy spectrum, spatial projections, stats."""
    fig = plt.figure(figsize=(18, 12))

    energy = data['KineticEnergy'] * 1000  # keV
    x = data['PrePosition_X']
    y = data['PrePosition_Y']
    z = data['PrePosition_Z']
    event_id = data['EventID']

    # 1. Energy spectrum
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.hist(energy, bins=np.linspace(0, 700, 141), color='steelblue',
             edgecolor='black', linewidth=0.3, alpha=0.7)
    ax1.axvline(662, color='red', linestyle='--', linewidth=1.5, label='662 keV')
    ax1.set_xlabel('Kinetic Energy (keV)')
    ax1.set_ylabel('Counts')
    ax1.set_title('Energy Spectrum (Arriving Photons)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. X-Y hitmap
    ax2 = fig.add_subplot(2, 3, 2)
    if len(x) > 1:
        h = ax2.hist2d(x, y, bins=[40, 40], cmap='hot')
        plt.colorbar(h[3], ax=ax2, label='Counts')
    else:
        ax2.scatter(x, y, s=30, c='red')
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_title('Hit Map (X-Y projection)')
    ax2.set_aspect('equal')

    # 3. Direction cosine distribution
    ax3 = fig.add_subplot(2, 3, 3)
    if 'PreDirection_X' in data and 'PreDirection_Y' in data:
        dx = data['PreDirection_X']
        dy = data['PreDirection_Y']
        ax3.scatter(dx, dy, s=10, alpha=0.5, c='steelblue')
        ax3.set_xlabel('Direction X')
        ax3.set_ylabel('Direction Y')
        ax3.set_title('Angular Distribution')
        ax3.set_aspect('equal')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No direction data', transform=ax3.transAxes,
                 ha='center', va='center')

    # 4. 3D scatter (smaller version)
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    n = min(5000, len(x))
    if n > 0:
        idx = np.random.choice(len(x), n, replace=False) if len(x) > n else np.arange(len(x))
        sc = ax4.scatter(x[idx], y[idx], z[idx], c=energy[idx],
                         cmap='plasma', s=10, alpha=0.6, vmin=0, vmax=700)
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    ax4.set_title('3D Arrival Positions')
    ax4.view_init(elev=25, azim=135)

    # 5. Y-position distribution (slit profile)
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.hist(y, bins=50, color='steelblue', edgecolor='black', linewidth=0.3, alpha=0.7)
    ax5.set_xlabel('Y position (mm)')
    ax5.set_ylabel('Counts')
    ax5.set_title('Y-Profile (Slit Projection)')
    ax5.grid(True, alpha=0.3)

    # 6. Diagnostics text
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    n_unique_events = len(np.unique(event_id))
    n_unique_tracks = len(np.unique(data['TrackID']))

    diag = (
        "DIAGNOSTIC SUMMARY\n"
        "=" * 35 + "\n\n"
        f"Total photons:     {len(energy):,}\n"
        f"Unique events:     {n_unique_events:,}\n"
        f"Unique tracks:     {n_unique_tracks:,}\n\n"
        f"Energy:\n"
        f"  Mean:            {energy.mean():.1f} keV\n"
        f"  Max:             {energy.max():.1f} keV\n"
        f"  Median:          {np.median(energy):.1f} keV\n\n"
        f"Spatial extent:\n"
        f"  X: [{x.min():.2f}, {x.max():.2f}] mm\n"
        f"  Y: [{y.min():.2f}, {y.max():.2f}] mm\n"
        f"  Z: [{z.min():.2f}, {z.max():.2f}] mm\n\n"
        f"Geometry:\n"
        f"  Source: Cs-137 (662 keV)\n"
        f"  Slit: 0.2 mm tungsten\n"
        f"  Detector: 5x40x40 mm CZT\n"
    )

    ax6.text(0.05, 0.95, diag, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='honeydew', alpha=0.9))

    plt.suptitle('CZT Slit Collimator — Full Diagnostic Report', fontsize=16, y=1.01)
    plt.tight_layout()
    outfile = os.path.join(output_dir, 'diagnostic_summary.png')
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"Saved: {outfile}")
    plt.close()
    return outfile


def main():
    import argparse
    parser = argparse.ArgumentParser(description='3D diagnostic plots for CZT simulation')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--input', type=str,
                        default=os.path.join(script_dir, 'output', 'merged', 'phsp_detector_merged.root'),
                        help='Input merged ROOT file')
    parser.add_argument('--output-dir', type=str,
                        default=os.path.join(script_dir, 'output', 'diagnostics'),
                        help='Output directory for diagnostic plots')
    args = parser.parse_args()

    input_file = args.input
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("3D Diagnostic Analysis — CZT Photon Phase Space")
    print("=" * 60)

    if not os.path.isfile(input_file):
        print(f"Input file not found: {input_file}")
        print("Skipping diagnostics (merged data not yet available)")
        sys.exit(0)

    data = load_phsp(input_file)
    n_photons = len(data['KineticEnergy'])
    print(f"\nTotal photons: {n_photons:,}")

    if n_photons == 0:
        print("No photons in file — skipping plots")
        sys.exit(0)

    print("\nGenerating diagnostic plots...")
    plot_3d_energy(data, output_dir)
    plot_hitmap(data, output_dir)
    plot_3d_with_geometry(data, output_dir)
    plot_diagnostic_summary(data, output_dir)

    print("\n" + "=" * 60)
    print(f"All diagnostics saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
