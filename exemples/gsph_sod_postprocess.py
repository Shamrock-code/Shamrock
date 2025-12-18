"""
GSPH Sod Shock Tube - Post-Processing Animation Generator
=========================================================

This script reads VTK files from a GSPH Sod shock tube simulation
and generates a matplotlib animation without re-running the simulation.

Usage:
    python gsph_sod_postprocess.py [output_dir] [--fps 10] [--format gif|mp4]

Example:
    python gsph_sod_postprocess.py gsph_sod_output --fps 15 --format mp4
"""

import os
import sys
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

# =============================================================================
# VTK Parser
# =============================================================================

def parse_vtk_file(filepath):
    """
    Parse a legacy VTK unstructured grid file.
    Returns dict with positions and field data.
    """
    data = {'points': None, 'fields': {}}

    with open(filepath, 'r') as f:
        lines = f.readlines()

    i = 0
    n_points = 0

    while i < len(lines):
        line = lines[i].strip()

        # Parse POINTS section
        if line.startswith('POINTS'):
            parts = line.split()
            n_points = int(parts[1])
            points = []
            i += 1
            while len(points) < n_points * 3:
                values = lines[i].strip().split()
                points.extend([float(v) for v in values])
                i += 1
            data['points'] = np.array(points).reshape(n_points, 3)
            continue

        # Parse POINT_DATA section
        if line.startswith('POINT_DATA'):
            n_point_data = int(line.split()[1])
            i += 1
            continue

        # Parse SCALARS
        if line.startswith('SCALARS'):
            parts = line.split()
            field_name = parts[1]
            i += 1  # Skip LOOKUP_TABLE line
            i += 1
            values = []
            while len(values) < n_points and i < len(lines):
                line_values = lines[i].strip().split()
                if not line_values or line_values[0] in ['SCALARS', 'VECTORS', 'POINT_DATA']:
                    break
                values.extend([float(v) for v in line_values])
                i += 1
            data['fields'][field_name] = np.array(values[:n_points])
            continue

        # Parse VECTORS
        if line.startswith('VECTORS'):
            parts = line.split()
            field_name = parts[1]
            i += 1
            values = []
            while len(values) < n_points * 3 and i < len(lines):
                line_values = lines[i].strip().split()
                if not line_values or line_values[0] in ['SCALARS', 'VECTORS', 'POINT_DATA']:
                    break
                values.extend([float(v) for v in line_values])
                i += 1
            data['fields'][field_name] = np.array(values[:n_points*3]).reshape(n_points, 3)
            continue

        i += 1

    return data

# =============================================================================
# Analytical Solution (Sod Tube)
# =============================================================================

class SodTubeAnalytical:
    """
    Analytical solution for Sod shock tube problem.
    Uses exact Riemann solver approach.
    """
    def __init__(self, gamma=1.4, rho_L=1.0, P_L=1.0, rho_R=0.125, P_R=0.1, x0=0.0):
        self.gamma = gamma
        self.rho_L = rho_L
        self.P_L = P_L
        self.rho_R = rho_R
        self.P_R = P_R
        self.x0 = x0

        # Sound speeds
        self.c_L = np.sqrt(gamma * P_L / rho_L)
        self.c_R = np.sqrt(gamma * P_R / rho_R)

        # Solve for star region
        self._solve_star_region()

    def _solve_star_region(self):
        """Solve for pressure and velocity in star region using Newton-Raphson."""
        gamma = self.gamma

        def f(P, rho, p, c):
            """Pressure function for Riemann problem."""
            if P > p:  # Shock
                A = 2 / ((gamma + 1) * rho)
                B = (gamma - 1) / (gamma + 1) * p
                return (P - p) * np.sqrt(A / (P + B))
            else:  # Rarefaction
                return 2 * c / (gamma - 1) * ((P / p) ** ((gamma - 1) / (2 * gamma)) - 1)

        def df(P, rho, p, c):
            """Derivative of pressure function."""
            if P > p:  # Shock
                A = 2 / ((gamma + 1) * rho)
                B = (gamma - 1) / (gamma + 1) * p
                return np.sqrt(A / (P + B)) * (1 - (P - p) / (2 * (P + B)))
            else:  # Rarefaction
                return 1 / (rho * c) * (P / p) ** (-(gamma + 1) / (2 * gamma))

        # Initial guess
        P_star = 0.5 * (self.P_L + self.P_R)

        # Newton-Raphson iteration
        for _ in range(100):
            f_L = f(P_star, self.rho_L, self.P_L, self.c_L)
            f_R = f(P_star, self.rho_R, self.P_R, self.c_R)
            df_L = df(P_star, self.rho_L, self.P_L, self.c_L)
            df_R = df(P_star, self.rho_R, self.P_R, self.c_R)

            residual = f_L + f_R
            derivative = df_L + df_R

            P_new = P_star - residual / derivative

            if abs(P_new - P_star) < 1e-10:
                break
            P_star = max(P_new, 1e-10)

        self.P_star = P_star
        self.u_star = 0.5 * (f(P_star, self.rho_L, self.P_L, self.c_L)
                            - f(P_star, self.rho_R, self.P_R, self.c_R))

        # Densities in star region
        if self.P_star > self.P_L:  # Left shock
            self.rho_star_L = self.rho_L * ((self.P_star / self.P_L + (gamma - 1) / (gamma + 1))
                                            / ((gamma - 1) / (gamma + 1) * self.P_star / self.P_L + 1))
        else:  # Left rarefaction
            self.rho_star_L = self.rho_L * (self.P_star / self.P_L) ** (1 / gamma)

        if self.P_star > self.P_R:  # Right shock
            self.rho_star_R = self.rho_R * ((self.P_star / self.P_R + (gamma - 1) / (gamma + 1))
                                            / ((gamma - 1) / (gamma + 1) * self.P_star / self.P_R + 1))
        else:  # Right rarefaction
            self.rho_star_R = self.rho_R * (self.P_star / self.P_R) ** (1 / gamma)

    def get_value(self, t, x):
        """Get density, velocity, pressure at position x and time t."""
        if t <= 0:
            if x < self.x0:
                return self.rho_L, 0.0, self.P_L
            else:
                return self.rho_R, 0.0, self.P_R

        gamma = self.gamma
        xi = (x - self.x0) / t  # Similarity variable

        # Sound speed in star region (left side)
        c_star_L = self.c_L * (self.P_star / self.P_L) ** ((gamma - 1) / (2 * gamma))

        # Wave speeds
        if self.P_star > self.P_L:  # Left shock
            S_L = self.u_star - self.c_L * np.sqrt((gamma + 1) / (2 * gamma) * self.P_star / self.P_L
                                                   + (gamma - 1) / (2 * gamma))
            head_L = S_L
            tail_L = S_L
        else:  # Left rarefaction
            head_L = -self.c_L
            tail_L = self.u_star - c_star_L

        if self.P_star > self.P_R:  # Right shock
            S_R = self.u_star + self.c_R * np.sqrt((gamma + 1) / (2 * gamma) * self.P_star / self.P_R
                                                   + (gamma - 1) / (2 * gamma))
        else:  # Right rarefaction (not present in standard Sod)
            S_R = self.u_star + self.c_R * (self.P_star / self.P_R) ** ((gamma - 1) / (2 * gamma))

        # Determine region and return values
        if xi < head_L:
            # Left region (undisturbed)
            return self.rho_L, 0.0, self.P_L
        elif xi < tail_L:
            # Inside left rarefaction
            if self.P_star <= self.P_L:
                u = 2 / (gamma + 1) * (self.c_L + xi)
                c = self.c_L - (gamma - 1) / 2 * u
                rho = self.rho_L * (c / self.c_L) ** (2 / (gamma - 1))
                P = self.P_L * (c / self.c_L) ** (2 * gamma / (gamma - 1))
                return rho, u, P
            else:
                return self.rho_star_L, self.u_star, self.P_star
        elif xi < self.u_star:
            # Star region (left of contact)
            return self.rho_star_L, self.u_star, self.P_star
        elif xi < S_R:
            # Star region (right of contact)
            return self.rho_star_R, self.u_star, self.P_star
        else:
            # Right region (undisturbed)
            return self.rho_R, 0.0, self.P_R

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate animation from GSPH Sod VTK files')
    parser.add_argument('output_dir', nargs='?', default='gsph_sod_output',
                        help='Directory containing VTK files')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second')
    parser.add_argument('--format', choices=['gif', 'mp4'], default='gif',
                        help='Output format')
    parser.add_argument('--gamma', type=float, default=1.4, help='Adiabatic index')
    args = parser.parse_args()

    # Find VTK files
    vtk_pattern = os.path.join(args.output_dir, '*.vtk')
    vtk_files = sorted(glob.glob(vtk_pattern))

    if not vtk_files:
        print(f"Error: No VTK files found in {args.output_dir}")
        sys.exit(1)

    print(f"Found {len(vtk_files)} VTK files")

    # Parse all VTK files
    print("Parsing VTK files...")
    frames_data = []
    for vtk_file in vtk_files:
        data = parse_vtk_file(vtk_file)
        frames_data.append(data)
        print(f"  Parsed: {os.path.basename(vtk_file)}")

    # Determine x range from first frame
    x_coords = frames_data[0]['points'][:, 0]
    x_min, x_max = x_coords.min(), x_coords.max()
    x_center = (x_min + x_max) / 2

    # Analytical solution
    sod = SodTubeAnalytical(gamma=args.gamma, x0=x_center)

    # Estimate time from frame index (assume uniform spacing to t=0.245)
    t_target = 0.245
    n_frames = len(frames_data)
    frame_times = [i * t_target / (n_frames - 1) for i in range(n_frames)]

    # Setup figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('GSPH Sod Shock Tube', fontsize=14)

    ax_rho = axes[0, 0]
    ax_vx = axes[0, 1]
    ax_P = axes[1, 0]
    ax_info = axes[1, 1]

    # Initialize plots
    scat_rho, = ax_rho.plot([], [], 'b.', markersize=1, alpha=0.5, label='GSPH')
    scat_vx, = ax_vx.plot([], [], 'b.', markersize=1, alpha=0.5, label='GSPH')
    scat_P, = ax_P.plot([], [], 'b.', markersize=1, alpha=0.5, label='GSPH')

    line_rho, = ax_rho.plot([], [], 'r-', linewidth=1.5, label='Analytical')
    line_vx, = ax_vx.plot([], [], 'r-', linewidth=1.5, label='Analytical')
    line_P, = ax_P.plot([], [], 'r-', linewidth=1.5, label='Analytical')

    # Configure axes
    for ax, ylabel, title in [(ax_rho, 'Density', 'Density'),
                               (ax_vx, 'Velocity', 'Velocity'),
                               (ax_P, 'Pressure', 'Pressure')]:
        ax.set_xlim(x_min, x_max)
        ax.set_xlabel('x')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    ax_rho.set_ylim(0, 1.2)
    ax_vx.set_ylim(-0.1, 1.1)
    ax_P.set_ylim(0, 1.2)

    # Info panel
    ax_info.axis('off')
    info_text = ax_info.text(0.1, 0.5, '', transform=ax_info.transAxes,
                             fontsize=12, verticalalignment='center',
                             family='monospace')

    time_text = fig.text(0.5, 0.02, '', ha='center', fontsize=12)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    def init():
        scat_rho.set_data([], [])
        scat_vx.set_data([], [])
        scat_P.set_data([], [])
        line_rho.set_data([], [])
        line_vx.set_data([], [])
        line_P.set_data([], [])
        time_text.set_text('')
        info_text.set_text('')
        return scat_rho, scat_vx, scat_P, line_rho, line_vx, line_P, time_text, info_text

    def animate(frame_idx):
        data = frames_data[frame_idx]
        t = frame_times[frame_idx]

        x = data['points'][:, 0]

        # Get fields (try different possible names)
        rho = data['fields'].get('rho', data['fields'].get('density', np.zeros(len(x))))
        vx = data['fields'].get('vxyz', data['fields'].get('vel', np.zeros((len(x), 3))))
        if vx.ndim == 2:
            vx = vx[:, 0]
        P = data['fields'].get('P', data['fields'].get('pressure', np.zeros(len(x))))

        scat_rho.set_data(x, rho)
        scat_vx.set_data(x, vx)
        scat_P.set_data(x, P)

        # Analytical solution
        x_ana = np.linspace(x_min, x_max, 500)
        rho_ana, vx_ana, P_ana = [], [], []
        for xi in x_ana:
            r, v, p = sod.get_value(t, xi)
            rho_ana.append(r)
            vx_ana.append(v)
            P_ana.append(p)

        line_rho.set_data(x_ana, rho_ana)
        line_vx.set_data(x_ana, vx_ana)
        line_P.set_data(x_ana, P_ana)

        time_text.set_text(f't = {t:.4f}')

        info = f"Frame: {frame_idx + 1}/{n_frames}\n"
        info += f"Particles: {len(x)}\n"
        info += f"gamma: {args.gamma}"
        info_text.set_text(info)

        return scat_rho, scat_vx, scat_P, line_rho, line_vx, line_P, time_text, info_text

    # Create animation
    anim = FuncAnimation(fig, animate, init_func=init, frames=n_frames,
                         interval=1000//args.fps, blit=True)

    # Save animation
    output_file = os.path.join(args.output_dir, f'gsph_sod_animation.{args.format}')
    print(f"Saving animation to {output_file}...")

    if args.format == 'gif':
        anim.save(output_file, writer=PillowWriter(fps=args.fps))
    else:
        anim.save(output_file, writer=FFMpegWriter(fps=args.fps))

    print(f"Animation saved to {output_file}")

    # Save final frame
    final_png = os.path.join(args.output_dir, 'gsph_sod_final.png')
    animate(n_frames - 1)
    fig.savefig(final_png, dpi=150)
    print(f"Final frame saved to {final_png}")

    plt.close()

if __name__ == '__main__':
    main()
