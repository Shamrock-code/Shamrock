"""
Toro shocks test in RAMSES solver
=============================================

Compare Sod tube with all slope limiters & Riemann solvers

Initial conditions from Toro "Riemann Solvers and Numerical Methods for Fluid Dynamics"

+------+-------+---------+------------+-----------+---------+-----------+-----------+-------+
| Test | x_0   | rho_L   | vx_L       | p_L       | rho_R   | vx_R      | p_R       | tend  |
+======+=======+=========+============+===========+=========+===========+===========+=======+
| 1    | 0.3   | 1.0     | 0.75       | 1.0       | 0.125   | 0.0       | 0.1       | 0.2   |
| 2    | 0.5   | 1.0     | -2.0       | 0.4       | 1.0     | 2.0       | 0.4       | 0.15  |
| 3    | 0.5   | 1.0     | 0.0        | 1000.0    | 1.0     | 0.0       | 0.01      | 0.012 |
| 4    | 0.4   | 5.99924 | 19.5975    | 460.894   | 5.99242 | -6.19633  | 46.0950   | 0.035 |
| 5    | 0.8   | 1.0     | -19.59745  | 1000.0    | 1.0     | -19.59745 | 0.01      | 0.012 |
| 6    | 0.5   | 1.4     | 0.0        | 1.0       | 1.0     | 0.0       | 1.0       | 2.0   |
| 7    | 0.5   | 1.4     | 0.1        | 1.0       | 1.0     | 0.1       | 1.0       | 2.0   |
+------+-------+---------+------------+-----------+---------+-----------+-----------+-------+
"""

import os

import matplotlib.pyplot as plt
import numpy as np

import shamrock

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")


def toro_initial_conditions(test_number: int):
    def cond(i):
        conditions = {
            1: (0.3, 1.0, 0.75, 1.0, 0.125, 0.0, 0.1, 0.2),
            2: (0.5, 1.0, -2.0, 0.4, 1.0, 2.0, 0.4, 0.15),
            3: (0.5, 1.0, 0.0, 1000.0, 1.0, 0.0, 0.01, 0.012),
            4: (0.4, 5.99924, 19.5975, 460.894, 5.99242, -6.19633, 46.0950, 0.035),
            5: (0.8, 1.0, -19.59745, 1000.0, 1.0, -19.59745, 0.01, 0.012),
            6: (0.5, 1.4, 0.0, 1.0, 1.0, 0.0, 1.0, 2.0),
            7: (0.5, 1.4, 0.1, 1.0, 1.0, 0.1, 1.0, 2.0),
        }
        if i not in conditions:
            raise ValueError(f"Invalid test number: {i}")
        return conditions[i]

    x_0, rho_L, vx_L, p_L, rho_R, vx_R, p_R, tend = cond(test_number)

    etot_L = p_L / (gamma - 1) + 0.5 * rho_L * vx_L**2
    etot_R = p_R / (gamma - 1) + 0.5 * rho_R * vx_R**2

    def rho(x):
        if x < x_0:
            return rho_L
        else:
            return rho_R

    def rhoetot(x):
        if x < x_0:
            return etot_L
        else:
            return etot_R

    def rhovel(x):
        if x < x_0:
            return (rho_L * vx_L, 0, 0)
        else:
            return (rho_R * vx_R, 0, 0)

    return {
        "x_0": x_0,
        "lambda_rho": rho,
        "lambda_rhoetot": rhoetot,
        "lambda_rhovel": rhovel,
        "tend": tend,
    }


timestamps = 40
gamma = 1.4

output_folder = "_to_trash/toro_tests/"
os.makedirs(output_folder, exist_ok=True)


multx = 2
multy = 1
multz = 1

sz = 1 << 1
base = 16

rez_plot = 256
positions_plot = [(x, 0, 0) for x in np.linspace(0, 1, rez_plot).tolist()[:-1]]


def run_test(
    test_number: int, slope_limiter: str, riemann_solver: str, only_last_step: bool = True
):
    ctx = shamrock.Context()
    ctx.pdata_layout_new()

    model = shamrock.get_Model_Ramses(context=ctx, vector_type="f64_3", grid_repr="i64_3")

    cfg = model.gen_default_config()
    scale_fact = 1 / (sz * base * multx)
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
    cfg.set_boundary_condition("x", "outflow")
    cfg.set_boundary_condition("y", "outflow")
    cfg.set_boundary_condition("z", "outflow")
    model.set_solver_config(cfg)

    model.init_scheduler(int(1e7), 1)
    model.make_base_grid((0, 0, 0), (sz, sz, sz), (base * multx, base * multy, base * multz))

    dic = toro_initial_conditions(test_number)
    tmax = dic["tend"]
    lambda_rho = dic["lambda_rho"]
    lambda_rhoetot = dic["lambda_rhoetot"]
    lambda_rhovel = dic["lambda_rhovel"]

    def rho_map(rmin, rmax):
        x, y, z = rmin
        return lambda_rho(x)

    def rhoetot_map(rmin, rmax):
        x, y, z = rmin
        return lambda_rhoetot(x)

    def rhovel_map(rmin, rmax):
        x, y, z = rmin
        return lambda_rhovel(x)

    model.set_field_value_lambda_f64("rho", rho_map)
    model.set_field_value_lambda_f64("rhoetot", rhoetot_map)
    model.set_field_value_lambda_f64_3("rhovel", rhovel_map)

    results = []

    def analysis(iplot: int):
        rho_vals = model.render_slice("rho", "f64", positions_plot)
        rhov_vals = model.render_slice("rhovel", "f64_3", positions_plot)
        rhoetot_vals = model.render_slice("rhoetot", "f64", positions_plot)

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

    return results, tmax


def plot_results(data, cases, tmax, test_number):

    arr_x = [x[0] for x in positions_plot]

    fig, axes = plt.subplots(3, 1, figsize=(6, 15))
    fig.suptitle(f"Test {test_number} - t={tmax} (Last Step)", fontsize=14)

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
        print(key)
        print(data)
        if key in data:
            # Get the last timestep

            ax1.plot(arr_x, data[key][-1]["rho"], label=f"{limiter} {solver} (rho)", linewidth=1)
            ax2.plot(arr_x, data[key][-1]["vx"], label=f"{limiter} {solver} (vx)", linewidth=1)
            ax3.plot(arr_x, data[key][-1]["P"], label=f"{limiter} {solver} (P)", linewidth=1)

    ax1.legend()
    ax2.legend()
    ax3.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"toro_shocks_test_{test_number}.png"))
    plt.close()


def gif_results(data, tmax, test_number, case_anim):

    arr_x = [x[0] for x in positions_plot]

    import matplotlib.animation as animation

    fig2, axes = plt.subplots(3, 1, figsize=(8, 10))
    fig2.suptitle(f"{case_anim} - t = {0.0:.3f} s", fontsize=14)

    ax_rho, ax_vx, ax_P = axes

    # Calculate global min/max across all frames for fixed y-axis limits
    rho_min = min(np.min(frame["rho"]) for frame in data)
    rho_max = max(np.max(frame["rho"]) for frame in data)
    vx_min = min(np.min(frame["vx"]) for frame in data)
    vx_max = max(np.max(frame["vx"]) for frame in data)
    P_min = min(np.min(frame["P"]) for frame in data)
    P_max = max(np.max(frame["P"]) for frame in data)

    # Add 5% margin to y-axis limits
    rho_margin = (rho_max - rho_min) * 0.05
    vx_margin = (vx_max - vx_min) * 0.05
    P_margin = (P_max - P_min) * 0.05

    # Configure each axis
    ax_rho.set_xlabel("$x$")
    ax_rho.set_ylabel("$\\rho$")
    ax_rho.set_ylim(rho_min - rho_margin, rho_max + rho_margin)
    ax_rho.grid(True, alpha=0.3)

    ax_vx.set_xlabel("$x$")
    ax_vx.set_ylabel("$v_x$")
    ax_vx.set_ylim(vx_min - vx_margin, vx_max + vx_margin)
    ax_vx.grid(True, alpha=0.3)

    ax_P.set_xlabel("$x$")
    ax_P.set_ylabel("$P$")
    ax_P.set_ylim(P_min - P_margin, P_max + P_margin)
    ax_P.grid(True, alpha=0.3)

    # Create lines for each variable on its own axis
    (line_rho,) = ax_rho.plot(arr_x, data[0]["rho"], label="$\\rho$", linewidth=2, color="C0")
    (line_vx,) = ax_vx.plot(arr_x, data[0]["vx"], label="$v_x$", linewidth=2, color="C1")
    (line_P,) = ax_P.plot(arr_x, data[0]["P"], label="$P$", linewidth=2, color="C2")

    ax_rho.legend()
    ax_vx.legend()
    ax_P.legend()

    def animate(frame):
        t = tmax * frame / timestamps
        line_rho.set_ydata(data[frame]["rho"])
        line_vx.set_ydata(data[frame]["vx"])
        line_P.set_ydata(data[frame]["P"])

        fig2.suptitle(f"{case_anim} - t = {t:.3f} s", fontsize=14)
        return (line_rho, line_vx, line_P)

    anim = animation.FuncAnimation(
        fig2, animate, frames=timestamps + 1, interval=50, blit=False, repeat=True
    )
    plt.tight_layout()
    writer = animation.PillowWriter(fps=15, metadata=dict(artist="Me"), bitrate=1800)
    anim.save(os.path.join(output_folder, f"toro_shocks_test_{test_number}.gif"), writer=writer)
    plt.close()


def run_and_plot(cases, test_number, case_anim):
    data = {}
    for slope_limiter, riemann_solver in cases:
        print(f"running {slope_limiter} {riemann_solver}")
        key = f"{slope_limiter}_{riemann_solver}"
        only_last_step = not (case_anim == key)
        data[key], tmax = run_test(
            test_number, slope_limiter, riemann_solver, only_last_step=only_last_step
        )
    print(data)
    plot_results(data, cases, tmax, test_number)
    return gif_results(data[case_anim], tmax, test_number, case_anim)


# %%

cases = [
    # ("none", "rusanov"),
    ("none", "hll"),
    # ("none", "hllc"),
    ("minmod", "rusanov"),
    ("minmod", "hll"),
    ("minmod", "hllc"),
]

cases_no_hllc = [
    # ("none", "rusanov"),
    ("none", "hll"),
    ("minmod", "rusanov"),
    ("minmod", "hll"),
]

run_and_plot(cases_no_hllc, 1, "minmod_hll")
run_and_plot(cases_no_hllc, 2, "minmod_hll")
run_and_plot(cases, 3, "minmod_hllc")
run_and_plot(cases, 4, "minmod_hllc")
run_and_plot(cases, 5, "minmod_hllc")
run_and_plot(cases, 6, "minmod_hllc")
run_and_plot(cases, 7, "minmod_hllc")
