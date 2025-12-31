"""
Generate GIF animations for Pairing Instability Test.

Usage: python generate_gifs.py [--lattice square|fcc|all] [--mode simple|annotated|both] [--fps N]
"""

import argparse
import glob
import io
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

try:
    import imageio
    import pyvista as pv
    from PIL import Image
except ImportError as e:
    print(f"Error: {e}. Install with: pip install pyvista imageio Pillow")
    sys.exit(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.normpath(
    os.path.join(SCRIPT_DIR, "../../../../../../simulations_data/pairing_instability")
)
GIFS_DIR = os.path.join(BASE_DIR, "gifs")

CONFIGS = {
    "square": {"name": "Square Grid", "color": "blue"},
    "fcc": {"name": "FCC Lattice", "color": "red"},
}


def load_vtk(path):
    mesh = pv.read(path)
    data = {"positions": np.array(mesh.points)}
    for f in ["rho", "P", "h", "v"]:
        if f in mesh.array_names:
            data[f] = np.array(mesh[f])
    return data


def render_frame(data, time, name, frame, total, color, metadata, annotated):
    pos = data["positions"]
    if annotated:
        fig = plt.figure(figsize=(14, 10), dpi=100)
        ax = fig.add_axes([0.05, 0.15, 0.55, 0.75])
        if "rho" in data:
            sc = ax.scatter(
                pos[:, 0],
                pos[:, 1],
                s=8,
                c=data["rho"],
                cmap="viridis",
                alpha=0.7,
                vmin=0.8,
                vmax=1.2,
            )
            plt.colorbar(sc, ax=ax, label="Density")
        else:
            ax.scatter(pos[:, 0], pos[:, 1], s=8, alpha=0.6, c="blue")
        ax.set(xlim=(-0.55, 0.55), ylim=(-0.55, 0.55), aspect="equal", xlabel="x", ylabel="y")
        ax.set_title(f"SPH Pairing Instability ({name}, M4)", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Info panel - read physics from metadata
        gamma = metadata.get("gamma", 5.0 / 3.0)
        rho_0 = metadata.get("rho_0", 1.0)
        P_0 = metadata.get("P_0", 1.0)
        u_0 = metadata.get("u_0", P_0 / ((gamma - 1) * rho_0))
        N = metadata.get("N", int(np.sqrt(len(pos))))

        ax_info = fig.add_axes([0.65, 0.15, 0.30, 0.75])
        ax_info.axis("off")
        info = f"""
    Parameters
    ----------
    Kernel: M4 (Cubic Spline)
    Lattice: {name}
    Particles: {len(pos)}
    N = {N}, Perturb: +/-5% dx

    Physics
    ----------
    gamma = {gamma:.4f}
    rho_0 = {rho_0}, P_0 = {P_0}
    u_0 = {u_0:.2f}

    Time: t = {time:.4f}
    Frame: {frame + 1} / {total}
        """
        ax_info.text(
            0,
            0.95,
            info,
            transform=ax_info.transAxes,
            fontsize=11,
            va="top",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # Progress bar
        prog = (frame + 1) / total
        ax_p = fig.add_axes([0.65, 0.05, 0.30, 0.03])
        ax_p.barh(0, prog, color="green", height=1)
        ax_p.barh(0, 1 - prog, left=prog, color="lightgray", height=1)
        ax_p.set(xlim=(0, 1), ylim=(-0.5, 0.5))
        ax_p.axis("off")
        ax_p.set_title(f"Progress: {prog * 100:.0f}%", fontsize=10)

        if "rho" in data:
            rho = data["rho"]
            fig.text(
                0.30,
                0.02,
                f"rho: min={rho.min():.3f}, max={rho.max():.3f}, mean={rho.mean():.3f}",
                ha="center",
            )
    else:
        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
        ax.scatter(pos[:, 0], pos[:, 1], s=3, alpha=0.6, c=color)
        ax.set(xlim=(-0.6, 0.6), ylim=(-0.6, 0.6), aspect="equal", xlabel="x", ylabel="y")
        ax.set_title(f"SPH Pairing Instability ({name})\nt = {time:.3f}")
        ax.grid(True, alpha=0.3)

    with io.BytesIO() as buf:
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        buf.seek(0)
        img = np.array(Image.open(buf))
    plt.close(fig)
    return img


def generate_gif(lattice, annotated, fps):
    cfg = CONFIGS[lattice]
    vtk_dir = os.path.join(BASE_DIR, f"{lattice}_sph_m4", "vtk")
    if not os.path.exists(vtk_dir):
        print(f"  VTK dir not found: {vtk_dir}")
        return False

    vtk_files = sorted(glob.glob(os.path.join(vtk_dir, "pairing_*.vtk")))
    if not vtk_files:
        print(f"  No VTK files in {vtk_dir}")
        return False

    # Load metadata
    meta_file = os.path.join(BASE_DIR, f"{lattice}_sph_m4", "metadata.json")
    metadata, times = {}, {}
    if os.path.exists(meta_file):
        with open(meta_file) as f:
            metadata = json.load(f)
            times = {o["index"]: o["time"] for o in metadata.get("outputs", [])}

    print(f"  Processing {len(vtk_files)} frames...")
    frames = []
    for i, vf in enumerate(vtk_files):
        data = load_vtk(vf)
        t = times.get(i, i / len(vtk_files))
        frames.append(
            render_frame(data, t, cfg["name"], i, len(vtk_files), cfg["color"], metadata, annotated)
        )
        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{len(vtk_files)}")

    suffix = "_annotated" if annotated else ""
    out = os.path.join(GIFS_DIR, f"pairing_{lattice}_sph_m4{suffix}.gif")
    os.makedirs(GIFS_DIR, exist_ok=True)
    imageio.mimsave(out, frames, fps=fps, loop=0)
    print(f"  Saved: {out}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate GIF animations for Pairing Instability")
    parser.add_argument("--lattice", choices=["all", "square", "fcc"], default="all")
    parser.add_argument("--fps", type=int, default=10)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--annotated", action="store_true", help="Only annotated versions")
    group.add_argument("--simple", action="store_true", help="Only simple versions")
    args = parser.parse_args()

    lattices = list(CONFIGS.keys()) if args.lattice == "all" else [args.lattice]
    modes = [True] if args.annotated else ([False] if args.simple else [False, True])

    print("=== PAIRING INSTABILITY GIF GENERATION ===")
    count = 0
    for lat in lattices:
        print(f"\n{CONFIGS[lat]['name']}:")
        for ann in modes:
            print(f"  {'Annotated' if ann else 'Simple'}:")
            if generate_gif(lat, ann, args.fps):
                count += 1
    print(f"\nGenerated {count} GIF(s) in {GIFS_DIR}")


if __name__ == "__main__":
    main()
