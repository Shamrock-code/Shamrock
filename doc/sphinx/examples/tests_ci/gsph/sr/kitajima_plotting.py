"""
Kitajima-style Plotting Utilities
=================================

Common plotting functions for SR-GSPH tests matching Kitajima et al. (2025)
arXiv:2510.18251v1 figure style.
"""

import numpy as np


def plot_kitajima_4panel(
    x_sim,
    P_sim,
    n_sim,
    vx_sim,
    h_sim,
    x_exact,
    P_exact,
    n_exact,
    vx_exact,
    filename,
    title=None,
    h0=None,
):
    """Create Kitajima-style 4-panel plot (P, n, vx, h/h0) matching paper figures.

    Args:
        h0: Initial smoothing length for normalization. If None, uses median(h_sim).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(6, 8), sharex=True)
    fig.subplots_adjust(hspace=0.05, left=0.15, right=0.95, top=0.95, bottom=0.08)

    colors = ["#007F66", "#CC3311", "#003366", "#EE7733"]  # P, n, vx, h
    labels = [r"$P$", r"$n$", r"$v^x$", r"$h/h_0$"]

    # Plot P
    axes[0].plot(x_exact, P_exact, "k-", linewidth=1.0)
    axes[0].scatter(x_sim, P_sim, s=2, c=colors[0], marker="+", alpha=0.7)
    axes[0].set_ylabel(labels[0], fontsize=12)

    # Plot n
    axes[1].plot(x_exact, n_exact, "k-", linewidth=1.0)
    axes[1].scatter(x_sim, n_sim, s=2, c=colors[1], marker="^", alpha=0.7)
    axes[1].set_ylabel(labels[1], fontsize=12)

    # Plot vx
    axes[2].plot(x_exact, vx_exact, "k-", linewidth=1.0)
    axes[2].scatter(x_sim, vx_sim, s=2, c=colors[2], marker="x", alpha=0.7)
    axes[2].set_ylabel(labels[2], fontsize=12)

    # Plot h/h0 (normalized smoothing length, Kitajima style)
    if h0 is None:
        h0 = np.median(h_sim)  # Use median as reference if not provided
    h_normalized = h_sim / h0
    axes[3].scatter(x_sim, h_normalized, s=2, c=colors[3], marker="^", alpha=0.7)
    axes[3].set_ylabel(labels[3], fontsize=12)
    axes[3].set_xlabel(r"$x$", fontsize=12)

    for ax in axes:
        ax.set_xlim(-0.5, 0.5)
        ax.tick_params(direction="in", which="both")

    if title:
        fig.suptitle(title, fontsize=10, y=0.98)

    from pathlib import Path

    abs_path = Path(filename).resolve()
    plt.savefig(str(abs_path), dpi=150, bbox_inches="tight")
    print(f"✓ Plot saved: {abs_path}")
    plt.close(fig)
    return str(abs_path)


def plot_kitajima_3panel(
    x_sim, P_sim, n_sim, vx_sim, x_exact, P_exact, n_exact, vx_exact, filename, title=None
):
    """Create Kitajima-style 3-panel plot (P, n, vx) without h."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(6, 6), sharex=True)
    fig.subplots_adjust(hspace=0.05, left=0.15, right=0.95, top=0.95, bottom=0.10)

    colors = ["#007F66", "#CC3311", "#003366"]
    labels = [r"$P$", r"$n$", r"$v^x$"]

    axes[0].plot(x_exact, P_exact, "k-", linewidth=1.0)
    axes[0].scatter(x_sim, P_sim, s=2, c=colors[0], marker="+", alpha=0.7)
    axes[0].set_ylabel(labels[0], fontsize=12)

    axes[1].plot(x_exact, n_exact, "k-", linewidth=1.0)
    axes[1].scatter(x_sim, n_sim, s=2, c=colors[1], marker="^", alpha=0.7)
    axes[1].set_ylabel(labels[1], fontsize=12)

    axes[2].plot(x_exact, vx_exact, "k-", linewidth=1.0)
    axes[2].scatter(x_sim, vx_sim, s=2, c=colors[2], marker="x", alpha=0.7)
    axes[2].set_ylabel(labels[2], fontsize=12)
    axes[2].set_xlabel(r"$x$", fontsize=12)

    for ax in axes:
        ax.set_xlim(-0.5, 0.5)
        ax.tick_params(direction="in", which="both")

    if title:
        fig.suptitle(title, fontsize=10, y=0.98)

    from pathlib import Path

    abs_path = Path(filename).resolve()
    plt.savefig(str(abs_path), dpi=150, bbox_inches="tight")
    print(f"✓ Plot saved: {abs_path}")
    plt.close(fig)
    return str(abs_path)


def compute_L2_errors(x_sim, y_sim, x_exact, y_exact, x_min=-0.5, x_max=0.5):
    """Compute L2 error between simulation and exact solution."""
    mask = (x_sim >= x_min) & (x_sim <= x_max)
    x_f = x_sim[mask]
    y_f = y_sim[mask]

    if len(x_f) == 0:
        return float("inf")

    y_interp = np.interp(x_f, x_exact, y_exact)
    y_norm = np.mean(np.abs(y_interp)) + 1e-10
    err = np.sqrt(np.mean((y_f - y_interp) ** 2)) / y_norm
    return err


def plot_tangent_velocity_grid(results_list, filename):
    """
    Create Kitajima-style 3x3 grid plot for tangential velocity tests.
    Matches arXiv:2510.18251v1 Figure 7.

    Args:
        results_list: List of dicts with keys:
            'v_t': tangential velocity value
            'x_sim', 'P_sim', 'n_sim', 'vx_sim', 'vy_sim': simulation data
            'x_exact', 'P_exact', 'n_exact', 'vx_exact', 'vt_exact': exact solution
        filename: Output filename
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows = len(results_list)
    fig, axes = plt.subplots(n_rows, 3, figsize=(9, 3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    fig.subplots_adjust(hspace=0.15, wspace=0.25, left=0.08, right=0.97, top=0.95, bottom=0.08)

    for i, res in enumerate(results_list):
        v_t = res["v_t"]
        x_sim = res["x_sim"]
        P_sim = res["P_sim"]
        n_sim = res["n_sim"]
        vx_sim = res["vx_sim"]
        vy_sim = res["vy_sim"]  # tangential velocity
        x_exact = res["x_exact"]
        P_exact = res["P_exact"]
        n_exact = res["n_exact"]
        vx_exact = res["vx_exact"]
        vt_exact = res["vt_exact"]

        # Column 0: P/1000 (green) and n/5 (dark red)
        ax0 = axes[i, 0]
        ax0.plot(x_exact, P_exact / 1000, "g-", linewidth=1.5, label=r"$P/1000$")
        ax0.plot(x_exact, n_exact / 5, color="darkred", linewidth=1.5, label=r"$n/5$")
        ax0.scatter(x_sim, P_sim / 1000, s=8, c="green", marker="v", alpha=0.6)
        ax0.scatter(x_sim, n_sim / 5, s=8, c="darkred", marker="^", alpha=0.6)
        ax0.set_xlim(-0.5, 0.5)
        ax0.set_ylim(0, 1.2)
        ax0.set_ylabel(r"$P/1000$", color="green", fontsize=11)
        ax0.tick_params(direction="in")
        # Add n/5 label on left
        ax0.text(-0.45, 0.25, r"$n/5$", color="darkred", fontsize=11)
        ax0.text(-0.45, 0.95, r"$P/1000$", color="green", fontsize=11)

        # Column 1: v^x
        ax1 = axes[i, 1]
        ax1.plot(x_exact, vx_exact, "navy", linewidth=1.5)
        ax1.scatter(x_sim, vx_sim, s=8, c="navy", marker="x", alpha=0.6)
        ax1.set_xlim(-0.5, 0.5)
        ax1.set_ylabel(r"$v^x$", fontsize=11)
        ax1.tick_params(direction="in")

        # Column 2: v^t (tangential velocity)
        ax2 = axes[i, 2]
        ax2.plot(x_exact, vt_exact, "deepskyblue", linewidth=1.5)
        ax2.scatter(x_sim, vy_sim, s=8, c="deepskyblue", marker="x", alpha=0.6)
        ax2.set_xlim(-0.5, 0.5)
        ax2.set_ylabel(r"$v^t$", fontsize=11)
        ax2.tick_params(direction="in")

        # Only bottom row gets x-labels
        if i == n_rows - 1:
            ax0.set_xlabel(r"$x$", fontsize=11)
            ax1.set_xlabel(r"$x$", fontsize=11)
            ax2.set_xlabel(r"$x$", fontsize=11)

    from pathlib import Path

    abs_path = Path(filename).resolve()
    plt.savefig(str(abs_path), dpi=150, bbox_inches="tight")
    print(f"✓ Plot saved: {abs_path}")
    plt.close(fig)
    return str(abs_path)
