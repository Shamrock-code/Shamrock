"""
Advection test in RAMSES solver
=============================================

Compare advection with all slope limiters & Riemann solvers
"""

import os

import matplotlib.pyplot as plt
import numpy as np

import shamrock

tmax = 1.0
timestamps = 40

multx = 1
multy = 1
multz = 1

sz = 1 << 1
base = 16

positions = [(x, 0, 0) for x in np.linspace(0, 1, 256).tolist()[:-1]]


def run_advect(slope_limiter: str, riemann_solver: str, only_last_step: bool = True):
    ctx = shamrock.Context()
    ctx.pdata_layout_new()

    model = shamrock.get_Model_Ramses(context=ctx, vector_type="f64_3", grid_repr="i64_3")

    cfg = model.gen_default_config()
    scale_fact = 1 / (sz * base * multx)
    cfg.set_scale_factor(scale_fact)
    cfg.set_eos_gamma(1.00001)

    if slope_limiter == "none":
        cfg.set_slope_lim_none()
    elif slope_limiter == "vanleer":
        cfg.set_slope_lim_vanleer_f()
    elif slope_limiter == "vanleer_std":
        cfg.set_slope_lim_vanleer_std()
    elif slope_limiter == "vanleer_sym":
        cfg.set_slope_lim_vanleer_sym()
    elif slope_limiter == "minmod":
        cfg.set_slope_lim_minmod()
    else:
        raise ValueError(f"Invalid slope limiter: {slope_limiter}")

    if riemann_solver == "rusanov":
        cfg.set_riemann_solver_rusanov()
    elif riemann_solver == "hll":
        cfg.set_riemann_solver_hll()
    elif riemann_solver == "hllc":
        cfg.set_riemann_solver_hllc()
    else:
        raise ValueError(f"Invalid Riemann solver: {riemann_solver}")

    model.set_solver_config(cfg)

    model.init_scheduler(int(1e7), 1)
    model.make_base_grid((0, 0, 0), (sz, sz, sz), (base * multx, base * multy, base * multz))

    def rho_map(rmin, rmax):
        x, y, z = rmin

        if x < 0.6 and x > 0.4:
            return 2

        return 1.0

    def rhoe_map(rmin, rmax):
        rho = rho_map(rmin, rmax)
        return 1.0 * rho

    def rhovel_map(rmin, rmax):
        x, y, z = rmin
        rho = rho_map(rmin, rmax)
        return (1 * rho, 0 * rho, 0 * rho)

    model.set_field_value_lambda_f64("rho", rho_map)
    model.set_field_value_lambda_f64("rhoetot", rhoe_map)
    model.set_field_value_lambda_f64_3("rhovel", rhovel_map)

    results = []

    def analysis(iplot: int):
        rho_vals = model.render_slice("rho", "f64", positions)
        results.append(rho_vals)

    if only_last_step:
        model.evolve_until(tmax)
        analysis(timestamps)
    else:
        dt_evolve = tmax / timestamps

        for i in range(timestamps + 1):
            model.evolve_until(dt_evolve * i)
            analysis(i)

    return results


# %%
data = {}
data["none_rusanov"] = run_advect("none", "rusanov")
data["none_hll"] = run_advect("none", "hll")
data["none_hllc"] = run_advect("none", "hllc")
data["vanleer_sym_rusanov"] = run_advect("vanleer_sym", "rusanov")
data["vanleer_sym_hll"] = run_advect("vanleer_sym", "hll")
data["vanleer_sym_hllc"] = run_advect("vanleer_sym", "hllc")
data["minmod_rusanov"] = run_advect("minmod", "rusanov")
data["minmod_hll"] = run_advect("minmod", "hll")
data["minmod_hllc"] = run_advect("minmod", "hllc", only_last_step=False)

# %%
# Plot 1: Comparison grouped by Riemann solver (last timestep only)
riemann_solvers = ["rusanov", "hll", "hllc"]
slope_limiters = ["none", "vanleer", "vanleer_sym", "minmod"]

fig, axes = plt.subplots(3, 1, figsize=(6, 15))
fig.suptitle(f"t={tmax} (Last Step)", fontsize=14)

for idx, solver in enumerate(riemann_solvers):
    ax = axes[idx]
    ax.set_title(f"Riemann Solver: {solver}")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$\\rho$")
    ax.grid(True, alpha=0.3)

    for limiter in slope_limiters:
        key = f"{limiter}_{solver}"
        if key in data:
            # Get the last timestep
            last_step = data[key][-1]
            ax.plot([x[0] for x in positions], last_step, label=limiter, linewidth=2)

    ax.legend()

plt.tight_layout()
plt.show()

# %%
# Plot 2: Animation of vanleer_sym_hllc configuration

# sphinx_gallery_thumbnail_number = 2

from matplotlib.animation import FuncAnimation

fig2, ax2 = plt.subplots()
ax2.set_xlabel("$x$")
ax2.set_ylabel("$\\rho$")
ax2.set_ylim(0.9, 2.1)
ax2.grid(True, alpha=0.3)

x_positions = np.linspace(0, 1, len(data["minmod_hllc"][0]))
(line,) = ax2.plot(x_positions, data["minmod_hllc"][0])
ax2.set_title(f"minmod_hllc - t = {0.0:.3f} s")


def animate(frame):
    t = tmax * frame / timestamps
    line.set_ydata(data["minmod_hllc"][frame])
    ax2.set_title(f"minmod_hllc - t = {t:.3f} s")
    return (line,)


anim = FuncAnimation(fig2, animate, frames=timestamps + 1, interval=150, blit=False, repeat=True)
plt.tight_layout()
plt.show()
