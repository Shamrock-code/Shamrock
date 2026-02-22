"""
Sod tube test in RAMSES solver
=============================================

Compare Sod tube with all slope limiters & Riemann solvers
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

import shamrock

tmax = 0.245
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

    cfg.from_json(
        {
            "hydro_riemann_solver": riemann_solver,
            "slope_limiter": slope_limiter,
            "eos_gamma": gamma,
            "face_half_time_interpolation": True,
            "grid_coord_to_pos_fact": 2 / (sz * base * multx),
        }
    )

    print("used config:\n" + json.dumps(cfg.to_json(), indent=4))

    model.set_solver_config(cfg)

    model.init_scheduler(int(1e7), 1)
    model.make_base_grid((0, 0, 0), (sz, sz, sz), (base * multx, base * multy, base * multz))

    def rho_map(rmin, rmax):
        x, y, z = rmin
        if x < 1:
            return 1
        else:
            return 0.125

    etot_L = 1.0 / (gamma - 1)
    etot_R = 0.1 / (gamma - 1)

    def rhoetot_map(rmin, rmax):
        rho = rho_map(rmin, rmax)

        x, y, z = rmin
        if x < 1:
            return etot_L
        else:
            return etot_R

    def rhovel_map(rmin, rmax):
        rho = rho_map(rmin, rmax)

        return (0, 0, 0)

    model.set_field_value_lambda_f64("rho", rho_map)
    model.set_field_value_lambda_f64("rhoetot", rhoetot_map)
    model.set_field_value_lambda_f64_3("rhovel", rhovel_map)

    results = []

    def analysis(iplot: int):
        rho_vals = np.array(model.render_slice("rho", "f64", positions))
        rhov_vals = np.array(model.render_slice("rhovel", "f64_3", positions))
        rhoetot_vals = np.array(model.render_slice("rhoetot", "f64", positions))

        vx = rhov_vals[:, 0] / rho_vals
        P = (rhoetot_vals - 0.5 * rho_vals * vx**2) * (gamma - 1)
        results_dic = {
            "rho": rho_vals,
            "vx": vx,
            "P": P,
        }
        results.append(results_dic)

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
data["minmod_rusanov"] = run_advect("minmod", "rusanov")
data["minmod_hll"] = run_advect("minmod", "hll")
data["minmod_hllc"] = run_advect("minmod", "hllc", only_last_step=False)

# %%
# Prepare data to be plotted
riemann_solvers = ["rusanov", "hll", "hllc"]
slope_limiters = ["none", "minmod"]

xref = 1.0
xrange = 0.5
sod = shamrock.phys.SodTube(gamma=gamma, rho_1=1, P_1=1, rho_5=0.125, P_5=0.1)

arr_x = [x[0] for x in positions]


def get_data_analytic(frame: int):
    t = tmax * frame / timestamps

    arr_rho = []
    arr_P = []
    arr_vx = []

    for i in range(len(arr_x)):
        x_ = arr_x[i] - xref

        _rho, _vx, _P = sod.get_value(t, x_)
        arr_rho.append(_rho)
        arr_vx.append(_vx)
        arr_P.append(_P)

    return {
        "rho": arr_rho,
        "vx": arr_vx,
        "P": arr_P,
    }


data_analytic = [get_data_analytic(i) for i in range(timestamps + 1)]


# %%
# Plot 1: Comparison against analytical solution
fig, axes = plt.subplots(3, 1, figsize=(6, 15))
fig.suptitle(f"t={tmax} (Last Step)", fontsize=14)

for i in range(3):
    axes[i].set_xlabel("$x$")
    axes[i].set_yscale("log")
    axes[i].grid(True, alpha=0.3)

ax1, ax2, ax3 = axes
ax1.set_xlabel("$x$")
ax1.set_ylabel("$\\delta \\rho$")

ax2.set_xlabel("$x$")
ax2.set_ylabel("$\\delta v_x$")

ax3.set_xlabel("$x$")
ax3.set_ylabel("$\\delta P$")


last_analytic = data_analytic[-1]


for limiter in slope_limiters:
    for solver in riemann_solvers:
        key = f"{limiter}_{solver}"
        if key in data:
            # Get the last timestep
            delta_rho = np.abs(data[key][-1]["rho"] - last_analytic["rho"])
            delta_vx = np.abs(data[key][-1]["vx"] - last_analytic["vx"])
            delta_P = np.abs(data[key][-1]["P"] - last_analytic["P"])

            ax1.plot(arr_x, delta_rho, label=f"{limiter} {solver} (rho)", linewidth=1)
            ax2.plot(arr_x, delta_vx, label=f"{limiter} {solver} (vx)", linewidth=1)
            ax3.plot(arr_x, delta_P, label=f"{limiter} {solver} (P)", linewidth=1)


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
ax2.set_ylim(-0.1, 1.1)
ax2.set_xlim(0.5, 1.5)
ax2.grid(True, alpha=0.3)

(line_analytic_rho,) = ax2.plot(
    arr_x, data_analytic[0]["rho"], color="black", label="analytic", linewidth=2
)
(line_analytic_vx,) = ax2.plot(
    arr_x, data_analytic[0]["vx"], color="black", label="analytic", linewidth=2
)
(line_analytic_P,) = ax2.plot(
    arr_x, data_analytic[0]["P"], color="black", label="analytic", linewidth=2
)
(line_rho,) = ax2.plot(arr_x, data["minmod_hllc"][0]["rho"], label="rho", linewidth=2)
(line_vx,) = ax2.plot(arr_x, data["minmod_hllc"][0]["vx"], label="vx", linewidth=2)
(line_P,) = ax2.plot(arr_x, data["minmod_hllc"][0]["P"], label="P", linewidth=2)

ax2.legend()
ax2.set_title(f"minmod_hllc - t = {0.0:.3f} s")


def animate(frame):
    t = tmax * frame / timestamps
    line_rho.set_ydata(data["minmod_hllc"][frame]["rho"])
    line_vx.set_ydata(data["minmod_hllc"][frame]["vx"])
    line_P.set_ydata(data["minmod_hllc"][frame]["P"])
    line_analytic_rho.set_ydata(data_analytic[frame]["rho"])
    line_analytic_vx.set_ydata(data_analytic[frame]["vx"])
    line_analytic_P.set_ydata(data_analytic[frame]["P"])
    ax2.set_title(f"minmod_hllc - t = {t:.3f} s")
    return (line_rho, line_vx, line_P, line_analytic_rho, line_analytic_vx, line_analytic_P)


anim = FuncAnimation(fig2, animate, frames=timestamps + 1, interval=50, blit=False, repeat=True)
plt.tight_layout()
plt.show()
