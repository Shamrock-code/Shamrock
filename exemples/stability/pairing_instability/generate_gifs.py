"""
Generate annotated GIF animations for Pairing Instability Test.

Reads VTK output files and creates annotated animations showing the
pairing instability evolution for both Square and FCC lattice configurations.

Usage:
    python generate_gifs.py                    # Generate all gifs
    python generate_gifs.py --lattice square   # Only square lattice
    python generate_gifs.py --lattice fcc      # Only FCC lattice
    python generate_gifs.py --annotated        # Only annotated versions
    python generate_gifs.py --simple           # Only simple versions

Output directory: ../../../../../../simulations_data/pairing_instability/gifs/
"""

import argparse
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

try:
    import pyvista as pv
except ImportError:
    print("Error: pyvista is required. Install with: pip install pyvista")
    sys.exit(1)

try:
    import imageio
except ImportError:
    print("Error: imageio is required. Install with: pip install imageio")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("Error: Pillow is required. Install with: pip install Pillow")
    sys.exit(1)


# Base paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_OUTPUT_DIR = os.path.normpath(
    os.path.join(SCRIPT_DIR, "../../../../../../simulations_data/pairing_instability")
)
GIFS_DIR = os.path.join(BASE_OUTPUT_DIR, "gifs")

# Lattice configurations to process
LATTICE_CONFIGS = {
    "square": {
        "name": "Square Grid",
        "vtk_dir": os.path.join(BASE_OUTPUT_DIR, "square_sph_m4", "vtk"),
        "metadata": os.path.join(BASE_OUTPUT_DIR, "square_sph_m4", "metadata.json"),
        "color": "blue",
    },
    "fcc": {
        "name": "FCC Lattice",
        "vtk_dir": os.path.join(BASE_OUTPUT_DIR, "fcc_sph_m4", "vtk"),
        "metadata": os.path.join(BASE_OUTPUT_DIR, "fcc_sph_m4", "metadata.json"),
        "color": "red",
    },
}


def load_vtk_data(vtk_file):
    """Load particle data from VTK file."""
    mesh = pv.read(vtk_file)
    positions = np.array(mesh.points)

    data = {"positions": positions}
    for field in ["rho", "P", "h", "v"]:
        if field in mesh.array_names:
            data[field] = np.array(mesh[field])

    return data


def create_simple_frame(positions, time, lattice_name, frame_num, total_frames, color="blue"):
    """Create a simple 2D scatter plot frame."""
    import io

    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)

    ax.scatter(positions[:, 0], positions[:, 1], s=3, alpha=0.6, c=color)
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.6)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"SPH Pairing Instability ({lattice_name})\nt = {time:.3f}")
    ax.grid(True, alpha=0.3)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    image = np.array(Image.open(buf))

    plt.close(fig)
    buf.close()
    return image


def create_annotated_frame(
    data, time, lattice_name, lattice_type, frame_num, total_frames, metadata=None
):
    """Create an annotated frame with physics information."""
    import io

    positions = data["positions"]

    fig = plt.figure(figsize=(14, 10), dpi=100)

    # Main particle plot
    ax_main = fig.add_axes([0.05, 0.15, 0.55, 0.75])

    if "rho" in data:
        rho = data["rho"]
        scatter = ax_main.scatter(
            positions[:, 0],
            positions[:, 1],
            s=8,
            c=rho,
            cmap="viridis",
            alpha=0.7,
            vmin=0.8,
            vmax=1.2,
        )
        plt.colorbar(scatter, ax=ax_main, label="Density")
    else:
        ax_main.scatter(positions[:, 0], positions[:, 1], s=8, alpha=0.6, c="blue")

    ax_main.set_xlim(-0.55, 0.55)
    ax_main.set_ylim(-0.55, 0.55)
    ax_main.set_aspect("equal")
    ax_main.set_xlabel("x", fontsize=12)
    ax_main.set_ylabel("y", fontsize=12)
    ax_main.grid(True, alpha=0.3)

    ax_main.set_title(
        f"SPH Pairing Instability Test\n({lattice_name}, M4 Cubic Spline)",
        fontsize=14,
        fontweight="bold",
    )

    # Info panel
    ax_info = fig.add_axes([0.65, 0.15, 0.30, 0.75])
    ax_info.axis("off")

    gamma = 5.0 / 3.0
    rho_0 = 1.0
    P_0 = 1.0
    u_0 = P_0 / ((gamma - 1) * rho_0)

    n_particles = len(positions)

    # Get N from metadata if available
    N = metadata.get("N", int(np.sqrt(n_particles))) if metadata else int(np.sqrt(n_particles))

    info_text = f"""
    Simulation Parameters
    ---------------------

    Method: Standard SPH
    Kernel: M4 (Cubic Spline)

    Lattice: {lattice_name}
    Particles: {n_particles}
    Resolution: N = {N}
    Perturbation: +/-5% of dx

    Physics
    ---------------------
    gamma = {gamma:.4f}
    rho_0 = {rho_0:.1f}
    P_0 = {P_0:.1f}
    u_0 = {u_0:.2f}

    Time: t = {time:.4f}
    Frame: {frame_num + 1} / {total_frames}
    """

    ax_info.text(
        0.0,
        0.95,
        info_text,
        transform=ax_info.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Progress bar
    progress = (frame_num + 1) / total_frames
    ax_prog = fig.add_axes([0.65, 0.05, 0.30, 0.03])
    ax_prog.barh(0, progress, color="green", height=1)
    ax_prog.barh(0, 1 - progress, left=progress, color="lightgray", height=1)
    ax_prog.set_xlim(0, 1)
    ax_prog.set_ylim(-0.5, 0.5)
    ax_prog.axis("off")
    ax_prog.set_title(f"Progress: {progress*100:.0f}%", fontsize=10)

    if "rho" in data:
        rho = data["rho"]
        stats_text = f"rho: min={rho.min():.3f}, max={rho.max():.3f}, mean={rho.mean():.3f}"
        fig.text(0.30, 0.02, stats_text, ha="center", fontsize=10)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    buf.seek(0)
    image = np.array(Image.open(buf))

    plt.close(fig)
    buf.close()
    return image


def generate_gif(
    vtk_dir, output_file, lattice_name, lattice_type, annotated=False, fps=10, color="blue"
):
    """Generate GIF from VTK files."""
    vtk_files = sorted(glob.glob(os.path.join(vtk_dir, "pairing_*.vtk")))

    if not vtk_files:
        print(f"  No VTK files found in {vtk_dir}")
        return False

    print(f"  Found {len(vtk_files)} VTK files")

    # Load metadata
    metadata = None
    times = {}
    metadata_file = os.path.join(os.path.dirname(vtk_dir), "metadata.json")
    if os.path.exists(metadata_file):
        with open(metadata_file) as f:
            metadata = json.load(f)
            if "outputs" in metadata:
                times = {o["index"]: o["time"] for o in metadata["outputs"]}

    # Generate frames
    frames = []
    total_frames = len(vtk_files)

    for i, vtk_file in enumerate(vtk_files):
        data = load_vtk_data(vtk_file)
        t = times.get(i, i * 1.0 / total_frames)

        if annotated:
            frame = create_annotated_frame(
                data, t, lattice_name, lattice_type, i, total_frames, metadata
            )
        else:
            frame = create_simple_frame(data["positions"], t, lattice_name, i, total_frames, color)

        frames.append(frame)

        if (i + 1) % 10 == 0 or i == total_frames - 1:
            print(f"    Processed {i + 1}/{total_frames} frames")

    # Save GIF
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    imageio.mimsave(output_file, frames, fps=fps, loop=0)
    print(f"  Saved: {output_file}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate GIF animations for Pairing Instability Test"
    )
    parser.add_argument(
        "--lattice",
        choices=["all", "square", "fcc"],
        default="all",
        help="Lattice type to process (default: all)",
    )
    parser.add_argument("--annotated", action="store_true", help="Generate only annotated versions")
    parser.add_argument("--simple", action="store_true", help="Generate only simple versions")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second (default: 10)")
    args = parser.parse_args()

    print("=" * 60)
    print("PAIRING INSTABILITY - GIF GENERATION")
    print("=" * 60)
    print(f"Base directory: {BASE_OUTPUT_DIR}")
    print(f"Output directory: {GIFS_DIR}")
    print(f"Lattice filter: {args.lattice}")
    print(f"FPS: {args.fps}")
    print("=" * 60)

    # Determine which lattices to process
    if args.lattice == "all":
        lattices = list(LATTICE_CONFIGS.keys())
    else:
        lattices = [args.lattice]

    generated = 0

    for lattice in lattices:
        config = LATTICE_CONFIGS[lattice]
        vtk_dir = config["vtk_dir"]
        lattice_name = config["name"]
        color = config["color"]

        print(f"\n{'='*40}")
        print(f"Processing: {lattice_name}")
        print(f"{'='*40}")

        if not os.path.exists(vtk_dir):
            print(f"  VTK directory not found: {vtk_dir}")
            print("  Run simulation first:")
            print(
                f"    PAIRING_LATTICE={lattice} ./shamrock --sycl-cfg 0:0 "
                "--rscript pairing_instability.py"
            )
            continue

        # Generate simple version
        if not args.annotated:
            print("\n  Generating simple GIF...")
            output_simple = os.path.join(GIFS_DIR, f"pairing_{lattice}_sph_m4.gif")
            if generate_gif(
                vtk_dir,
                output_simple,
                lattice_name,
                lattice,
                annotated=False,
                fps=args.fps,
                color=color,
            ):
                generated += 1

        # Generate annotated version
        if not args.simple:
            print("\n  Generating annotated GIF...")
            output_annotated = os.path.join(GIFS_DIR, f"pairing_{lattice}_sph_m4_annotated.gif")
            if generate_gif(
                vtk_dir,
                output_annotated,
                lattice_name,
                lattice,
                annotated=True,
                fps=args.fps,
                color=color,
            ):
                generated += 1

    print("\n" + "=" * 60)
    print(f"Generated {generated} GIF(s)")
    print(f"Output: {GIFS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
