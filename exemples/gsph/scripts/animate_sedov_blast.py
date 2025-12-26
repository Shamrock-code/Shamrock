#!/usr/bin/env python3
"""
GSPH Sedov-Taylor Blast Wave Animation with Analytical Solution Overlay

Creates animated comparison between GSPH simulation results and
the self-similar Sedov-Taylor solution.

The GSPH method originated from:
- Inutsuka, S. (2002) "Reformulation of Smoothed Particle Hydrodynamics
  with Riemann Solver"

Usage:
    python3 animate_sedov_blast.py <data_dir> [output_file]
"""

import glob
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Import from shared analytical module
exemples_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, exemples_dir)
from common.analytical.sedov import SedovAnalytical

# Try to import animation tools
try:
    from matplotlib.animation import FuncAnimation, PillowWriter

    HAS_ANIMATION = True
except ImportError:
    HAS_ANIMATION = False
    print("Warning: Animation requires pillow. Install with: pip install pillow")

# Try to import tqdm for progress bar
try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Configuration
data_dir = sys.argv[1] if len(sys.argv) > 1 else "results/gsph_sedov"
output_file = sys.argv[2] if len(sys.argv) > 2 else "gsph_sedov_animation.gif"

print("=" * 70)
print("GSPH Sedov-Taylor Blast Wave Animation")
print("=" * 70)
print(f"Data directory: {data_dir}")
print(f"Output file:    {output_file}")
print()


def load_snapshot(filename):
    """Load a single snapshot CSV file."""
    data = {}
    metadata = {}

    with open(filename, "r") as f:
        # Read metadata lines (start with #)
        for line in f:
            if line.startswith("#"):
                if ":" in line:
                    key, value = line[1:].strip().split(":", 1)
                    metadata[key.strip()] = value.strip()
            else:
                break

        # Read header and data
        f.seek(0)
        lines = [l for l in f.readlines() if not l.startswith("#")]

        if len(lines) < 2:
            return None

        header = lines[0].strip().split(",")

        for col_name in header:
            data[col_name] = []

        for line in lines[1:]:
            values = line.strip().split(",")
            for i, col_name in enumerate(header):
                try:
                    data[col_name].append(float(values[i]))
                except (ValueError, IndexError):
                    pass

    # Convert to numpy arrays
    for key in data:
        data[key] = np.array(data[key])

    data["metadata"] = metadata
    return data


def find_snapshots(data_dir):
    """Find all snapshot files in the data directory."""
    files = sorted(glob.glob(f"{data_dir}/snapshot_*.csv"))
    return files


def compute_radial_profiles(data, n_bins=100):
    """Compute radially averaged profiles from 3D particle data."""
    # Get positions
    x = data.get("pos_x", np.zeros(1))
    y = data.get("pos_y", np.zeros(1))
    z = data.get("pos_z", np.zeros(1))

    # Compute radial distance
    r = np.sqrt(x**2 + y**2 + z**2)

    # Get fields
    rho = data.get("dens", np.ones_like(r))
    vel_x = data.get("vel_x", np.zeros_like(r))
    vel_y = data.get("vel_y", np.zeros_like(r))
    vel_z = data.get("vel_z", np.zeros_like(r))
    pres = data.get("pres", np.ones_like(r))

    # Compute radial velocity
    vel_r = np.where(r > 0, (x * vel_x + y * vel_y + z * vel_z) / r, 0)

    # Create bins
    r_max = r.max() if len(r) > 0 else 1.0
    bin_edges = np.linspace(0, r_max, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Bin the data
    rho_profile = np.zeros(n_bins)
    vel_profile = np.zeros(n_bins)
    pres_profile = np.zeros(n_bins)
    counts = np.zeros(n_bins)

    indices = np.digitize(r, bin_edges) - 1
    indices = np.clip(indices, 0, n_bins - 1)

    for i in range(len(r)):
        idx = indices[i]
        rho_profile[idx] += rho[i]
        vel_profile[idx] += vel_r[i]
        pres_profile[idx] += pres[i]
        counts[idx] += 1

    # Average
    mask = counts > 0
    rho_profile[mask] /= counts[mask]
    vel_profile[mask] /= counts[mask]
    pres_profile[mask] /= counts[mask]

    return bin_centers, rho_profile, vel_profile, pres_profile, mask


# Find snapshot files
print("Scanning for snapshot files...")
files = find_snapshots(data_dir)

if len(files) == 0:
    print(f"ERROR: No snapshot files found in {data_dir}")
    print("Looking for: snapshot_*.csv")
    sys.exit(1)

print(f"Found {len(files)} snapshot files")
print()

# Load first snapshot to get parameters
first_data = load_snapshot(files[0])
if first_data is None:
    print("ERROR: Could not load first snapshot")
    sys.exit(1)

# Extract parameters from metadata
gamma = float(first_data["metadata"].get("gamma", "1.666667"))
E_blast = float(first_data["metadata"].get("E_blast", "1.0"))
rho_0 = float(first_data["metadata"].get("rho_0", "1.0"))

print("Simulation parameters:")
print(f"  gamma = {gamma}")
print(f"  E_blast = {E_blast}")
print(f"  rho_0 = {rho_0}")
print()

# Create analytical solution object
sedov_analytical = SedovAnalytical(gamma=gamma, E_blast=E_blast, rho_0=rho_0)

# Determine frame skip for reasonable animation size
n_frames = len(files)
max_frames = 50
frame_skip = max(1, n_frames // max_frames)
frame_indices = list(range(0, n_frames, frame_skip))

print(f"Animation: {len(frame_indices)} frames (every {frame_skip} snapshots)")
print()

# Pre-load all frame data
print("Loading snapshot data...")
frame_data = []

pbar = tqdm(total=len(frame_indices), desc="Loading") if HAS_TQDM else None

for idx in frame_indices:
    data = load_snapshot(files[idx])
    if data is not None:
        frame_data.append(data)
    if pbar:
        pbar.update(1)

if pbar:
    pbar.close()

print(f"Loaded {len(frame_data)} frames")
print()

# Create animation
print("Creating animation...")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle(
    "GSPH Sedov-Taylor Blast Wave - Comparison with Analytical Solution",
    fontsize=16,
    fontweight="bold",
)

# Colors
sim_color = "#0173B2"  # Blue for simulation
ana_color = "#D55E00"  # Red-orange for analytical

pbar_anim = tqdm(total=len(frame_data), desc="Rendering") if HAS_TQDM else None


def update(frame_num):
    """Update function for animation."""
    data = frame_data[frame_num]

    # Get time from metadata
    time_str = data["metadata"].get("time", "0.0")
    try:
        time = float(time_str.split()[0])
    except:
        time = 0.0

    # Compute radial profiles from simulation
    r_sim, rho_sim, vel_sim, pres_sim, mask = compute_radial_profiles(data)

    # Get analytical solution
    r_ana, rho_ana, vel_ana, pres_ana = sedov_analytical.solution_at_time(
        time, r_max=r_sim.max() * 1.2, n_points=500
    )

    # Clear axes
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()

    # Density
    ax1.plot(r_ana, rho_ana, color=ana_color, linewidth=2.5, label="Analytical", zorder=1)
    ax1.scatter(
        r_sim[mask], rho_sim[mask], color=sim_color, s=15, alpha=0.6, label="GSPH", zorder=2
    )
    ax1.set_ylabel(r"Density $\rho$", fontsize=12, fontweight="bold")
    ax1.set_title("Density Profile", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, r_sim.max() * 1.1)
    ax1.axvline(
        sedov_analytical.shock_radius(time),
        color="gray",
        linestyle="--",
        alpha=0.5,
        label=f"$R_s$ = {sedov_analytical.shock_radius(time):.3f}",
    )

    # Velocity
    ax2.plot(r_ana, vel_ana, color=ana_color, linewidth=2.5, label="Analytical", zorder=1)
    ax2.scatter(
        r_sim[mask], vel_sim[mask], color=sim_color, s=15, alpha=0.6, label="GSPH", zorder=2
    )
    ax2.set_ylabel(r"Radial Velocity $v_r$", fontsize=12, fontweight="bold")
    ax2.set_title("Velocity Profile", fontsize=13, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, r_sim.max() * 1.1)

    # Pressure
    ax3.plot(r_ana, pres_ana, color=ana_color, linewidth=2.5, label="Analytical", zorder=1)
    ax3.scatter(
        r_sim[mask], pres_sim[mask], color=sim_color, s=15, alpha=0.6, label="GSPH", zorder=2
    )
    ax3.set_ylabel(r"Pressure $P$", fontsize=12, fontweight="bold")
    ax3.set_xlabel(r"Radius $r$", fontsize=12, fontweight="bold")
    ax3.set_title("Pressure Profile", fontsize=13, fontweight="bold")
    ax3.legend(loc="upper right", fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, r_sim.max() * 1.1)

    # Shock radius vs time (accumulated data)
    ax4.text(
        0.5,
        0.5,
        f"Shock Radius: $R_s$ = {sedov_analytical.shock_radius(time):.4f}\n\n"
        f"Post-shock density: {sedov_analytical.post_shock_density():.2f}\n\n"
        f"Density ratio: {sedov_analytical.post_shock_density() / rho_0:.1f}",
        transform=ax4.transAxes,
        fontsize=14,
        verticalalignment="center",
        horizontalalignment="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    ax4.set_title("Sedov-Taylor Parameters", fontsize=13, fontweight="bold")
    ax4.axis("off")

    # Add time label
    fig.suptitle(
        f"GSPH Sedov-Taylor Blast Wave - t = {time:.4f}\n"
        f"Comparison with Self-Similar Solution (Inutsuka 2002)",
        fontsize=14,
        fontweight="bold",
    )

    if pbar_anim:
        pbar_anim.update(1)

    return ax1, ax2, ax3, ax4


if HAS_ANIMATION and len(frame_data) > 0:
    anim = FuncAnimation(fig, update, frames=len(frame_data), interval=150, blit=False, repeat=True)

    # Save animation
    os.makedirs(
        os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True
    )
    writer = PillowWriter(fps=8)
    anim.save(output_file, writer=writer, dpi=150)

    if pbar_anim:
        pbar_anim.close()

    plt.close()

    print()
    print("=" * 70)
    print("Animation Complete!")
    print("=" * 70)
    print(f"Saved: {output_file}")
else:
    print("ERROR: Cannot create animation (no data or missing dependencies)")

# Also create final state comparison plot
if len(frame_data) > 0:
    print()
    print("Creating final state comparison plot...")

    data = frame_data[-1]
    time_str = data["metadata"].get("time", "0.0")
    try:
        time = float(time_str.split()[0])
    except:
        time = 0.0

    r_sim, rho_sim, vel_sim, pres_sim, mask = compute_radial_profiles(data)
    r_ana, rho_ana, vel_ana, pres_ana = sedov_analytical.solution_at_time(
        time, r_max=r_sim.max() * 1.2, n_points=500
    )

    fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))
    fig2.suptitle(
        f"GSPH Sedov-Taylor Blast Wave - Final State (t = {time:.4f})\n"
        f"GSPH Method (Inutsuka 2002) vs Analytical Solution",
        fontsize=14,
        fontweight="bold",
    )

    # Density
    ax1.plot(r_ana, rho_ana, color=ana_color, linewidth=2.5, label="Analytical")
    ax1.scatter(r_sim[mask], rho_sim[mask], color=sim_color, s=20, alpha=0.6, label="GSPH")
    ax1.set_ylabel(r"Density $\rho$", fontsize=12, fontweight="bold")
    ax1.set_title("Density Profile", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axvline(sedov_analytical.shock_radius(time), color="gray", linestyle="--", alpha=0.5)

    # Velocity
    ax2.plot(r_ana, vel_ana, color=ana_color, linewidth=2.5, label="Analytical")
    ax2.scatter(r_sim[mask], vel_sim[mask], color=sim_color, s=20, alpha=0.6, label="GSPH")
    ax2.set_ylabel(r"Radial Velocity $v_r$", fontsize=12, fontweight="bold")
    ax2.set_title("Velocity Profile", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Pressure
    ax3.plot(r_ana, pres_ana, color=ana_color, linewidth=2.5, label="Analytical")
    ax3.scatter(r_sim[mask], pres_sim[mask], color=sim_color, s=20, alpha=0.6, label="GSPH")
    ax3.set_ylabel(r"Pressure $P$", fontsize=12, fontweight="bold")
    ax3.set_xlabel(r"Radius $r$", fontsize=12, fontweight="bold")
    ax3.set_title("Pressure Profile", fontsize=13, fontweight="bold")
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Info panel
    info_text = (
        f"Sedov-Taylor Parameters:\n\n"
        f"$\\gamma$ = {gamma:.4f}\n"
        f"$E_{{blast}}$ = {E_blast:.2f}\n"
        f"$\\rho_0$ = {rho_0:.2f}\n\n"
        f"At t = {time:.4f}:\n"
        f"$R_s$ = {sedov_analytical.shock_radius(time):.4f}\n"
        f"$\\rho_s$ = {sedov_analytical.post_shock_density():.2f}"
    )
    ax4.text(
        0.5,
        0.5,
        info_text,
        transform=ax4.transAxes,
        fontsize=12,
        verticalalignment="center",
        horizontalalignment="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        family="monospace",
    )
    ax4.set_title("Simulation Parameters", fontsize=13, fontweight="bold")
    ax4.axis("off")

    plt.tight_layout()

    final_plot = output_file.replace(".gif", "_final.png")
    plt.savefig(final_plot, dpi=150, bbox_inches="tight")
    print(f"Saved: {final_plot}")
    plt.close()

print("=" * 70)
