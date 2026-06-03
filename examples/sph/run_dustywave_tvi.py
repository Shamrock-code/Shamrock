"""
Dusty wave SPH test
========================

Test that the diffusion of epsilon is correct when the
momentum & energy equation are disabled.
"""

# sphinx_gallery_multi_image = "single"

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import shamrock

shamrock.enable_experimental_features()

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

# %%
# Sim parameters
rho = 1
epsilon_0 = 0.5
cs_g_list = np.logspace(-4, -1, 3).tolist()
ts = 1
delta_v_0_list = [cs * 0.001 for cs in cs_g_list]

bmin = (-0.5, -0.5 / 4, -0.5 / 4)
bmax = (0.5, 0.5 / 4, 0.5 / 4)

N_target = 1e3

# %%
# mpl style
mpl.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "axes.facecolor": "#f2f2f2",
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 8,
        "ytick.major.size": 8,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "legend.frameon": True,
        "legend.fancybox": False,
        "legend.edgecolor": "black",
    }
)

# %%
# Do setup


def do_setup(model, cs, delta_v_0):
    global bmin, bmax, xm, xM

    xm, ym, zm = bmin
    xM, yM, zM = bmax
    vol_b = (xM - xm) * (yM - ym) * (zM - zm)

    part_vol = vol_b / N_target

    # lattice volume
    HCP_PACKING_DENSITY = 0.74
    part_vol_lattice = HCP_PACKING_DENSITY * part_vol

    dr = (part_vol_lattice / ((4.0 / 3.0) * np.pi)) ** (1.0 / 3.0)

    pmass = -1

    bmin, bmax = model.get_ideal_hcp_box(dr, bmin, bmax)
    xm, ym, zm = bmin
    xM, yM, zM = bmax

    cfg = model.gen_default_config()
    # cfg.set_artif_viscosity_Constant(alpha_u = 1, alpha_AV = 1, beta_AV = 2)
    # cfg.set_artif_viscosity_VaryingMM97(alpha_min = 0.1,alpha_max = 1,sigma_decay = 0.1, alpha_u = 1, beta_AV = 2)
    cfg.set_artif_viscosity_VaryingCD10(
        alpha_min=0.0, alpha_max=1, sigma_decay=0.1, alpha_u=1, beta_AV=2
    )
    cfg.set_dust_mode_monofluid_tvi(nvar=1)
    cfg.set_dust_drag_constant([ts])
    cfg.set_boundary_periodic()
    cfg.set_eos_isothermal(cs)
    cfg.print_status()
    model.set_solver_config(cfg)

    scheduler_split_val = int(2e7)
    scheduler_merge_val = int(1)

    model.init_scheduler(scheduler_split_val, scheduler_merge_val)

    model.resize_simulation_box(bmin, bmax)

    setup = model.get_setup()
    gen = setup.make_generator_lattice_hcp(dr, bmin, bmax)
    setup.apply_setup(gen, insert_step=scheduler_split_val)

    def func_s(r):
        return np.sqrt(rho * epsilon_0)

    model.set_field_value_lambda_f64("s_j", func_s, 0)

    print(delta_v_0)

    def vel_func(r):
        global mm, MM
        x, y, z = r

        f = 2 * np.pi / (xM - xm)

        vel = delta_v_0 * np.sin(x * f)

        return (vel, 0.0, 0.0)

    model.set_field_value_lambda_f64_3("vxyz", vel_func)

    vol_b = (xM - xm) * (yM - ym) * (zM - zm)
    totmass = rho * vol_b

    pmass = model.total_mass_to_part_mass(totmass)
    model.set_particle_mass(pmass)

    model.set_cfl_cour(0.1)
    model.set_cfl_force(0.1)

    model.timestep()


# %%
# Field recovery for plots
def get_field_results(model):
    def custom_getter_x(size: int, dic_out: dict) -> np.array:
        return dic_out["xyz"][:, 0]

    def custom_getter_vx(size: int, dic_out: dict) -> np.array:
        return dic_out["vxyz"][:, 0]
    x_field = model.compute_field("custom", "f64", custom_getter_x)
    vx_field = model.compute_field("custom", "f64", custom_getter_vx)
    rho_field = model.compute_field("rho", "f64")
    s_j_field = model.compute_field("s_j", "f64")

    def internal_eps(size: int, s: np.array, rho: np.array) -> np.array:
        return (s**2) / rho

    eps_field = shamrock.map_fields_f64(internal_eps, s=s_j_field, rho=rho_field)

    def internal_rho_g(size: int, rho: np.array, eps: np.array) -> np.array:
        return rho * (1 - eps)

    def internal_rho_d(size: int, rho: np.array, eps: np.array) -> np.array:
        return rho * eps

    rho_g_field = shamrock.map_fields_f64(internal_rho_g, rho=rho_field, eps=eps_field)
    rho_d_field = shamrock.map_fields_f64(internal_rho_d, rho=rho_field, eps=eps_field)

    x_data = np.asarray(x_field.collect_data())
    vx_data = np.asarray(vx_field.collect_data())
    eps_data = np.asarray(eps_field.collect_data())
    rho_data = np.asarray(rho_field.collect_data())
    rho_g_data = np.asarray(rho_g_field.collect_data())
    rho_d_data = np.asarray(rho_d_field.collect_data())
    return x_data, rho_data, rho_g_data, rho_d_data,vx_data, eps_data


# %%
# Analytics


def dustywave_tvi_matrix(k, cs, ts, eps):
    """
    Create the matrix

    [[0, 0, -i*k*cs],
     [-k^2*ts*cs^2*eps*(1-eps), k^2*ts*cs^2*eps, 0],
     [i*k*cs*(1-eps), -i*k*cs, 0]]
    """

    a = k * cs
    b = k*k*ts*cs*cs*eps

    return np.array(
        [
            [0, 0, -1j * a],
            [b*(1-eps),-b, 0],
            [-1j * a * (1 - eps), -1j * a, 0],
        ],
        dtype=complex,
    )


def eigensystem_dustywave_tvi(k, cs, ts, eps):
    M = dustywave_tvi_matrix(k, cs, ts, eps)
    vals, vecs = np.linalg.eig(M)
    return 1j*vals, vecs


def dustywave_dispersion_relation(omega_k: float, k: float, cs: float, ts: float, eps: float):
    # w^4 + i w^3 / ts - cs^2 k^2 w^2 - i cs^2 k^2 (1-eps) w / ts = 0
    return (
        omega_k**4
        + 1j * (omega_k**3 / ts)
        - (cs**2 * k**2 * omega_k**2)
        - 1j * (cs**2 * k**2 * (1 - eps) * omega_k / ts)
    )


def get_dustywave_omega_k(k: float, cs: float, ts: float, eps: float) -> np.ndarray:
    # w^4 + i w^3/ts - cs^2 k^2 w^2 - i cs^2 k^2 (1-eps) w/ts = 0
    coeffs = [
        1.0,
        1j / ts,
        -(cs**2 * k**2),
        -1j * (cs**2 * k**2 * (1.0 - eps) / ts),
        0.0,
    ]
    return np.roots(coeffs)

## Use the matrix version above this one give slightly different result i dunno why
def dustywave_dispersion_relation_tvi(omega_k: float, k: float, cs: float, ts: float, eps: float):
    # i w^3/ts - cs^2 k^2 w^2 eps - i cs^2 k^2 (1-eps) w/ts = 0
    return (
        +1j * (omega_k**3 / ts)
        - (cs**2 * k**2 * omega_k**2 * eps)
        - 1j * (cs**2 * k**2 * (1 - eps) * omega_k / ts)
    )


def get_dustywave_tvi_omega_k(k: float, cs: float, ts: float, eps: float) -> np.ndarray:
    # i w^3/ts - cs^2 k^2 w^2 eps - i cs^2 k^2 (1-eps) w/ts = 0
    coeffs = [
        1j / ts,
        -(cs**2 * k**2 * eps),
        -1j * (cs**2 * k**2 * (1.0 - eps) / ts),
        0.0,
    ]
    return np.roots(coeffs)


def corresponding_root(omega_list):
    omega_re_pos = 0
    for ome in omega_list:
        if np.real(ome) > np.real(omega_re_pos):
            omega_re_pos = ome
    return omega_re_pos


# def analytical_wave(omega, x, t):

# %%
# Curve fitting
from scipy.linalg import lstsq
from scipy.optimize import curve_fit


def fit_sine_wave(x, y):
    design = np.column_stack([np.ones_like(x), np.sin(k * x), np.cos(k * x)])
    offset, a, b = lstsq(design, y)[0]
    ampl = np.hypot(a, b)
    phi = np.arctan2(b, a)
    if phi < 0:
        phi += np.pi
        ampl = -ampl
    return offset, ampl, phi


def sine_model(x, offset, ampl, phi):
    return offset + ampl * np.sin(phi + k * x)


def damped_sine_ampl(t, a, b, c):
    return a * np.sin(-b * t) * np.exp(-c * t)


def fit_damped_sine_ampl(t, ampl, omega_guess):
    t_arr = np.asarray(t)
    ampl_arr = np.asarray(ampl)
    print("omega_guess=", omega_guess)
    w_re0 = float(np.real(omega_guess))
    w_im0 = -float(np.imag(omega_guess))
    p0 = [np.abs(np.max(ampl_arr)), w_re0, 0]

    up_bound = [np.abs(np.max(ampl_arr)) * 2, 2 * w_re0, max(1.0, 10 * w_im0)]
    print("up bound=", up_bound)

    print("p0=", p0)
    popt, _ = curve_fit(
        damped_sine_ampl,
        t_arr,
        ampl_arr,
        p0=p0,
        bounds=([0, 0.0, 0], up_bound),
        maxfev=20000,
    )

    print("popt=", popt)
    return popt


# %%
# Perform the simulation
for ics, cs in enumerate(cs_g_list):
    ctx = shamrock.Context()
    ctx.pdata_layout_new()

    model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M6")
    do_setup(model, cs, delta_v_0_list[ics])

    k = 2 * np.pi / (xM - xm)

    # Compute Omega
    omega_k = get_dustywave_omega_k(k, cs, ts, epsilon_0)
    omega_k_tvi = get_dustywave_tvi_omega_k(k, cs, ts, epsilon_0)
    print(omega_k)
    print(omega_k_tvi)
    eigval, eigvec = eigensystem_dustywave_tvi(k, cs, ts, epsilon_0)
    print(f"eigenval = {eigval}")
    print(f"eigenvec = {eigvec}")

    # Find the root corresponding to this setup
    omega_re_pos = corresponding_root(omega_k)
    omega_re_pos_tvi = corresponding_root(omega_k_tvi)
    print(f"omega_re_pos={omega_re_pos}")
    print(f"omega_re_pos_tvi={omega_re_pos_tvi}")

    Twave = 2 * np.pi / (np.real(omega_re_pos_tvi))
    print(Twave)

    Twave_cnt = 10
    nwave = 1

    t_list = []
    rho_g_ampl_list = []
    rho_d_ampl_list = []
    rho_g_phi_list = []
    rho_d_phi_list = []
    rho_g_offset_list = []
    rho_d_offset_list = []

    os.makedirs("_to_trash", exist_ok=True)
    for i in range(int(Twave_cnt * nwave)):
        t = Twave * i / (Twave_cnt)
        model.evolve_until(t)
        x_data, rho_data, rho_g_data, rho_d_data, vx_data, eps_data = get_field_results(model)

        x_ana = np.linspace(xm, xM, 256)

        offset_g, ampl_g, phi_g = fit_sine_wave(x_data, rho_g_data)
        offset_d, ampl_d, phi_d = fit_sine_wave(x_data, rho_d_data)

        if model.get_time() == 0:
            phi_g = np.pi / 2
            phi_d = np.pi / 2

        rho_g_fit = sine_model(x_ana, offset_g, ampl_g, phi_g)
        rho_d_fit = sine_model(x_ana, offset_d, ampl_d, phi_d)

        print(f"rho_g fit: offset={offset_g:.6g}, ampl={ampl_g:.6g}, phi={phi_g:.6g} rad")
        print(f"rho_d fit: offset={offset_d:.6g}, ampl={ampl_d:.6g}, phi={phi_d:.6g} rad")

        t_list.append(model.get_time())
        rho_g_ampl_list.append(ampl_g)
        rho_d_ampl_list.append(ampl_d)
        rho_g_phi_list.append(phi_g)
        rho_d_phi_list.append(phi_d)
        rho_g_offset_list.append(offset_g)
        rho_d_offset_list.append(offset_d)

        fig, axs = plt.subplots(1, 1, figsize=(10, 5))
        axs.plot(x_data, rho_g_data, ".", label="gas")
        axs.plot(x_data, rho_d_data, ".", label="dust")

        axs.plot(
            x_ana,
            rho_g_fit,
            "-",
            label=rf"rho_g fit: $A={ampl_g:.3g}$, $\phi={phi_g:.3g}$",
        )
        axs.plot(
            x_ana,
            rho_d_fit,
            "-",
            label=rf"rho_d fit: $A={ampl_d:.3g}$, $\phi={phi_d:.3g}$",
        )

        axs.set_xlabel(r"$x$")
        axs.set_ylabel(r"$\rho_{g,d}$")
        axs.set_xlim(xm, xM)
        axs.set_ylim(rho / 2 - 1e-3, rho / 2 + 1e-3)
        axs.text(
            0.02,
            0.98,
            f"t = {t:.2f} | cs = {cs:e}",
            transform=axs.transAxes,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(f"_to_trash/dump_dustywave_tvi_{ics:02d}_{i:02d}.png")
        plt.close()

    t_arr = np.asarray(t_list)
    t_fit = np.linspace(t_arr.min(), t_arr.max(), 256)

    try:
        popt_g = fit_damped_sine_ampl(t_arr, rho_g_ampl_list, omega_re_pos)
        popt_d = fit_damped_sine_ampl(t_arr, rho_d_ampl_list, omega_re_pos)

        print(f"rho_g ampl fit: a={popt_g[0]:.6g}, b={popt_g[1]:.6g}, c={popt_g[2]:.6g}")
        print(f"rho_d ampl fit: a={popt_d[0]:.6g}, b={popt_d[1]:.6g}, c={popt_d[2]:.6g}")
    except ValueError:
        popt_g = None
        popt_d = None

    plt.figure(dpi=150)
    plt.plot(t_list, rho_g_ampl_list, ".", label="rho_g_ampl")
    plt.plot(t_list, rho_d_ampl_list, ".", label="rho_d_ampl")
    if popt_g is not None and popt_d is not None:
        plt.plot(
            t_fit,
            damped_sine_ampl(t_fit, *popt_g),
            "-",
            label=rf"rho_g fit: $a={popt_g[0]:.3g}$, $b={popt_g[1]:.3g}$, $c={popt_g[2]:.3g}$",
        )
        plt.plot(
            t_fit,
            damped_sine_ampl(t_fit, *popt_d),
            "-",
            label=rf"rho_d fit: $a={popt_d[0]:.3g}$, $b={popt_d[1]:.3g}$, $c={popt_d[2]:.3g}$",
        )
    plt.xlabel("t")
    plt.ylabel("rho")
    plt.title(f"cs={cs:.6g}")
    plt.legend()
    plt.savefig(f"_to_trash/dustywave_tvi_scan_{ics:04}.png")

# %%
# make gifs
from shamrock.utils.plot import show_image_sequence

keep_list = []

# %%
# show them the gifs (i have to unroll the loop otherwise the doc does not capture the gifs ...)
ani0 = show_image_sequence(f"_to_trash/dump_dustywave_tvi_{0:02d}_*.png")
plt.show()
# %%
ani1 = show_image_sequence(f"_to_trash/dump_dustywave_tvi_{1:02d}_*.png")
plt.show()
# %%
ani2 = show_image_sequence(f"_to_trash/dump_dustywave_tvi_{2:02d}_*.png")
plt.show()

plt.show()
