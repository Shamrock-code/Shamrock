"""
Sod tube test in RAMSES solver
=============================================

Compare Sod tube with all slope limiters & Riemann solvers
Toro 3rd initial condition

rho_l = 1
rho_r = 1

v_l = -2
v_r = 2

P_l = 0.4
P_r = 0.4
"""

import os

import matplotlib.pyplot as plt
import numpy as np

import shamrock

tmax = 0.15
timestamps = 40
gamma = 1.4

multx = 4
multy = 1
multz = 1

sz = 1 << 1
base = 16

positions = [(x, 0, 0) for x in np.linspace(0.5, 1.5, 256).tolist()[:-1]]


def run_advect(slope_limiter: str, riemann_solver: str, only_last_step: bool = True):
    ctx = shamrock.Context()
    ctx.pdata_layout_new()

    model = shamrock.get_Model_Ramses(context=ctx, vector_type="f64_3", grid_repr="i64_3")

    cfg = model.gen_default_config()
    scale_fact = 2 / (sz * base * multx)
    cfg.set_scale_factor(scale_fact)
    cfg.set_eos_gamma(gamma)

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

    cfg.set_face_time_interpolation(True)
    model.set_solver_config(cfg)

    model.init_scheduler(int(1e7), 1)
    model.make_base_grid((0, 0, 0), (sz, sz, sz), (base * multx, base * multy, base * multz))

    rho_l = 1
    rho_r = 1

    v_l = -2
    v_r = 2

    P_l = 0.4
    P_r = 0.4

    def rho_map(rmin, rmax):
        x, y, z = rmin
        if x < 1:
            return rho_l
        else:
            return rho_r

    etot_L = P_l / (gamma - 1) + 0.5 * rho_l * v_l**2
    etot_R = P_r / (gamma - 1) + 0.5 * rho_r * v_r**2

    def rhoetot_map(rmin, rmax):
        rho = rho_map(rmin, rmax)

        x, y, z = rmin
        if x < 1:
            return etot_L
        else:
            return etot_R

    def rhovel_map(rmin, rmax):
        rho = rho_map(rmin, rmax)

        x, y, z = rmin
        if x < 1:
            return (rho_l * v_l, 0, 0)
        else:
            return (rho_r * v_r, 0, 0)

    model.set_field_value_lambda_f64("rho", rho_map)
    model.set_field_value_lambda_f64("rhoetot", rhoetot_map)
    model.set_field_value_lambda_f64_3("rhovel", rhovel_map)

    results = []

    def analysis(iplot: int):
        rho_vals = model.render_slice("rho", "f64", positions)
        rhov_vals = model.render_slice("rhovel", "f64_3", positions)
        rhoetot_vals = model.render_slice("rhoetot", "f64", positions)

        vx = np.array(rhov_vals)[:, 0] / np.array(rho_vals)
        P = (np.array(rhoetot_vals) - 0.5 * np.array(rho_vals) * vx**2) * (gamma - 1)
        results_dic = {
            "rho": np.array(rho_vals),
            "vx": np.array(vx),
            "P": np.array(P),
        }
        results.append(results_dic)

    print(f"running {slope_limiter} {riemann_solver} with only_last_step={only_last_step}")

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

cases = [
    ("none", "rusanov"),
    ("none", "hll"),
    ("none", "hllc"),
    ("minmod", "rusanov"),
    ("minmod", "hll"),
    # ("minmod", "hllc"), # Crash for now
]

case_anim = "minmod_hll"

data = {}
for slope_limiter, riemann_solver in cases:
    print(f"running {slope_limiter} {riemann_solver}")
    key = f"{slope_limiter}_{riemann_solver}"
    only_last_step = not (case_anim == key)
    data[key] = run_advect(slope_limiter, riemann_solver, only_last_step=only_last_step)

# %%
# Plot 1: Comparison against analytical solution

arr_x = [x[0] for x in positions]

fig, axes = plt.subplots(3, 1, figsize=(6, 15))
fig.suptitle(f"t={tmax} (Last Step)", fontsize=14)

for i in range(3):
    axes[i].set_xlabel("$x$")
    # axes[i].set_yscale("log")
    axes[i].grid(True, alpha=0.3)

ax1, ax2, ax3 = axes
ax1.set_xlabel("$x$")
ax1.set_ylabel("$\\rho$")

ax2.set_xlabel("$x$")
ax2.set_ylabel("$v_x$")

ax3.set_xlabel("$x$")
ax3.set_ylabel("$P$")

for limiter, solver in cases:
    key = f"{limiter}_{solver}"
    if key in data:
        # Get the last timestep

        ax1.plot(arr_x, data[key][-1]["rho"], label=f"{limiter} {solver} (rho)", linewidth=1)
        ax2.plot(arr_x, data[key][-1]["vx"], label=f"{limiter} {solver} (vx)", linewidth=1)
        ax3.plot(arr_x, data[key][-1]["P"], label=f"{limiter} {solver} (P)", linewidth=1)


ax1.legend()
ax2.legend()
ax3.legend()

plt.tight_layout()
plt.show()

# %%
# Plot 2: Animation of vanleer_sym_hllc configuration

# sphinx_gallery_thumbnail_number = 2

from matplotlib.animation import FuncAnimation

fig2, ax2 = plt.subplots()
ax2.set_xlabel("$x$")
ax2.set_ylabel("$\\rho$")
ax2.set_ylim(-2, 2)
ax2.set_xlim(0.5, 1.5)
ax2.grid(True, alpha=0.3)

(line_rho,) = ax2.plot(arr_x, data[case_anim][0]["rho"], label="rho", linewidth=2)
(line_vx,) = ax2.plot(arr_x, data[case_anim][0]["vx"], label="vx", linewidth=2)
(line_P,) = ax2.plot(arr_x, data[case_anim][0]["P"], label="P", linewidth=2)


ax2.legend()
ax2.set_title(f"{case_anim} - t = {0.0:.3f} s")


def animate(frame):
    t = tmax * frame / timestamps
    line_rho.set_ydata(data[case_anim][frame]["rho"])
    line_vx.set_ydata(data[case_anim][frame]["vx"])
    line_P.set_ydata(data[case_anim][frame]["P"])
    ax2.set_title(f"{case_anim} - t = {t:.3f} s")
    return (line_rho, line_vx, line_P)


anim = FuncAnimation(fig2, animate, frames=timestamps + 1, interval=50, blit=False, repeat=True)
plt.tight_layout()
plt.show()
