#!/usr/bin/env python3
"""
GSPH Sod Shock Tube Animation with Analytical Solution Overlay

Creates animated comparison between GSPH simulation results and
the exact Riemann solution.

The GSPH method originated from:
- Inutsuka, S. (2002) "Reformulation of Smoothed Particle Hydrodynamics
  with Riemann Solver"

Usage:
    python3 animate_sod_tube.py <data_dir> [output_file]
"""

import glob
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Use shamrock's built-in SodTube analytical solution
import shamrock

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
data_dir = sys.argv[1] if len(sys.argv) > 1 else "results/gsph_sod"
output_file = sys.argv[2] if len(sys.argv) > 2 else "gsph_sod_animation.gif"

print("=" * 70)
print("GSPH Sod Shock Tube Animation")
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
gamma = float(first_data["metadata"].get("gamma", "1.4"))
rho_L = float(first_data["metadata"].get("rho_L", "1.0"))
rho_R = float(first_data["metadata"].get("rho_R", "0.125"))
p_L = float(first_data["metadata"].get("p_L", "1.0"))
p_R = float(first_data["metadata"].get("p_R", "0.1"))

print("Simulation parameters:")
print(f"  gamma = {gamma}")
print(f"  Left:  rho = {rho_L}, P = {p_L}")
print(f"  Right: rho = {rho_R}, P = {p_R}")
print()

# Create analytical solution object using shamrock's built-in SodTube
# Note: SodTube uses rho_1/P_1 for left state and rho_5/P_5 for right state
sod_analytical = shamrock.phys.SodTube(
    gamma=gamma, rho_1=rho_L, P_1=p_L, rho_5=rho_R, P_5=p_R
)


def get_analytical_solution(sod, t, x_array):
    """Get analytical solution at multiple x positions."""
    rho = np.zeros(len(x_array))
    vel = np.zeros(len(x_array))
    pres = np.zeros(len(x_array))
    for i, x in enumerate(x_array):
        rho[i], vel[i], pres[i] = sod.get_value(t, x)
    ene = pres / ((gamma - 1) * np.maximum(rho, 1e-10))
    return rho, vel, pres, ene

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
    "GSPH Sod Shock Tube - Comparison with Analytical Solution", fontsize=16, fontweight="bold"
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

    # Get simulation data
    x_sim = data["pos_x"]
    sort_idx = np.argsort(x_sim)
    x_sim = x_sim[sort_idx]

    rho_sim = data["dens"][sort_idx]
    vel_sim = data["vel_x"][sort_idx]
    pres_sim = data["pres"][sort_idx]
    ene_sim = data["ene"][sort_idx]

    # Get analytical solution
    x_ana = np.linspace(x_sim.min(), x_sim.max(), 500)
    rho_ana, vel_ana, pres_ana, ene_ana = get_analytical_solution(sod_analytical, time, x_ana)

    # Clear axes
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()

    # Density
    ax1.plot(x_ana, rho_ana, color=ana_color, linewidth=2.5, label="Analytical", zorder=1)
    ax1.scatter(x_sim, rho_sim, color=sim_color, s=10, alpha=0.6, label="GSPH", zorder=2)
    ax1.set_ylabel("Density ρ", fontsize=12, fontweight="bold")
    ax1.set_title("Density Profile", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(x_sim.min(), x_sim.max())

    # Velocity
    ax2.plot(x_ana, vel_ana, color=ana_color, linewidth=2.5, label="Analytical", zorder=1)
    ax2.scatter(x_sim, vel_sim, color=sim_color, s=10, alpha=0.6, label="GSPH", zorder=2)
    ax2.set_ylabel("Velocity u", fontsize=12, fontweight="bold")
    ax2.set_title("Velocity Profile", fontsize=13, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(x_sim.min(), x_sim.max())

    # Pressure
    ax3.plot(x_ana, pres_ana, color=ana_color, linewidth=2.5, label="Analytical", zorder=1)
    ax3.scatter(x_sim, pres_sim, color=sim_color, s=10, alpha=0.6, label="GSPH", zorder=2)
    ax3.set_ylabel("Pressure P", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Position x", fontsize=12, fontweight="bold")
    ax3.set_title("Pressure Profile", fontsize=13, fontweight="bold")
    ax3.legend(loc="upper right", fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(x_sim.min(), x_sim.max())

    # Internal Energy
    ax4.plot(x_ana, ene_ana, color=ana_color, linewidth=2.5, label="Analytical", zorder=1)
    ax4.scatter(x_sim, ene_sim, color=sim_color, s=10, alpha=0.6, label="GSPH", zorder=2)
    ax4.set_ylabel("Internal Energy e", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Position x", fontsize=12, fontweight="bold")
    ax4.set_title("Internal Energy Profile", fontsize=13, fontweight="bold")
    ax4.legend(loc="upper right", fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(x_sim.min(), x_sim.max())

    # Add time label
    fig.suptitle(
        f"GSPH Sod Shock Tube - t = {time:.4f}\n"
        f"Comparison with Analytical Solution (Inutsuka 2002)",
        fontsize=14,
        fontweight="bold",
    )

    if pbar_anim:
        pbar_anim.update(1)

    return ax1, ax2, ax3, ax4


if HAS_ANIMATION and len(frame_data) > 0:
    anim = FuncAnimation(fig, update, frames=len(frame_data), interval=100, blit=False, repeat=True)

    # Save animation
    os.makedirs(
        os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True
    )
    writer = PillowWriter(fps=10)
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

    x_sim = data["pos_x"]
    sort_idx = np.argsort(x_sim)
    x_sim = x_sim[sort_idx]

    rho_sim = data["dens"][sort_idx]
    vel_sim = data["vel_x"][sort_idx]
    pres_sim = data["pres"][sort_idx]
    ene_sim = data["ene"][sort_idx]

    x_ana = np.linspace(x_sim.min(), x_sim.max(), 500)
    rho_ana, vel_ana, pres_ana, ene_ana = get_analytical_solution(sod_analytical, time, x_ana)

    fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))
    fig2.suptitle(
        f"GSPH Sod Shock Tube - Final State (t = {time:.4f})\n"
        f"GSPH Method (Inutsuka 2002) vs Analytical Solution",
        fontsize=14,
        fontweight="bold",
    )

    ax1.plot(x_ana, rho_ana, color=ana_color, linewidth=2.5, label="Analytical")
    ax1.scatter(x_sim, rho_sim, color=sim_color, s=15, alpha=0.6, label="GSPH")
    ax1.set_ylabel("Density ρ", fontsize=12, fontweight="bold")
    ax1.set_title("Density Profile", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.plot(x_ana, vel_ana, color=ana_color, linewidth=2.5, label="Analytical")
    ax2.scatter(x_sim, vel_sim, color=sim_color, s=15, alpha=0.6, label="GSPH")
    ax2.set_ylabel("Velocity u", fontsize=12, fontweight="bold")
    ax2.set_title("Velocity Profile", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    ax3.plot(x_ana, pres_ana, color=ana_color, linewidth=2.5, label="Analytical")
    ax3.scatter(x_sim, pres_sim, color=sim_color, s=15, alpha=0.6, label="GSPH")
    ax3.set_ylabel("Pressure P", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Position x", fontsize=12, fontweight="bold")
    ax3.set_title("Pressure Profile", fontsize=13, fontweight="bold")
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    ax4.plot(x_ana, ene_ana, color=ana_color, linewidth=2.5, label="Analytical")
    ax4.scatter(x_sim, ene_sim, color=sim_color, s=15, alpha=0.6, label="GSPH")
    ax4.set_ylabel("Internal Energy e", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Position x", fontsize=12, fontweight="bold")
    ax4.set_title("Internal Energy Profile", fontsize=13, fontweight="bold")
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    final_plot = output_file.replace(".gif", "_final.png")
    plt.savefig(final_plot, dpi=150, bbox_inches="tight")
    print(f"Saved: {final_plot}")
    plt.close()

print("=" * 70)
