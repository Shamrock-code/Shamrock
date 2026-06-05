"""
Dusty wave SPH test
========================

Test that the dust/gas wave evolution match the eigen mode analysis.
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

xm = None
ym = None
zm = None
xM = None
yM = None
zM = None
dr = None


def do_setup(model, cs, delta_v_0):
    global xm, ym, zm, xM, yM, zM, bmin, bmax, dr

    if xm is None:
        xm, ym, zm = bmin
        xM, yM, zM = bmax
        vol_b = (xM - xm) * (yM - ym) * (zM - zm)

        part_vol = vol_b / N_target

        # lattice volume
        HCP_PACKING_DENSITY = 0.74
        part_vol_lattice = HCP_PACKING_DENSITY * part_vol

        dr = (part_vol_lattice / ((4.0 / 3.0) * np.pi)) ** (1.0 / 3.0)

        print(f"dr={dr}, bmin={bmin}, bmax={bmax}")

        pmass = -1

        bmin, bmax = shamrock.math.get_ideal_hcp_box(dr, bmin, bmax)
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
    return x_data, rho_data, rho_g_data, rho_d_data, vx_data, eps_data


# %%
# Analytics


def dustywave_tvi_matrix(k, cs, ts, eps):
    a = k * cs
    b = k * k * ts * cs * cs * eps

    return np.array(
        [
            [0, 0, -1j * a],
            [b * (1 - eps), -b, 0],
            [-1j * a * (1 - eps), 1j * a, 0],
        ],
        dtype=complex,
    )


def eigensystem_dustywave_tvi(k, cs, ts, eps):
    M = dustywave_tvi_matrix(k, cs, ts, eps)
    vals, vecs = np.linalg.eig(M)
    return 1j * vals, vecs


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


def eigen_model(x, t, offset, ampl, omega, k):
    return offset + np.real(ampl * np.exp(1j * (k * x - omega * t)))


def project_eigenmode(
    eigenvec: complex, eigenval: complex, rho_on_rho_0: complex, eps: complex, v_on_cs: complex
):
    v = np.array([rho_on_rho_0, eps, v_on_cs], dtype=complex)
    print(f"eigenvec={eigenvec}")
    print(f"v={v}")

    c = np.linalg.solve(eigenvec, v)

    print(f"c={c}")
    return c


def find_eigen_decomp(x_data, rho_data, eps_data, vx_data, eigval, eigvec):

    offset_rho, ampl_rho, phi_rho = fit_sine_wave(x_data, rho_data)
    offset_eps, ampl_eps, phi_eps = fit_sine_wave(x_data, eps_data)
    offset_vx, ampl_vx, phi_vx = fit_sine_wave(x_data, vx_data)

    print(f"offset_rho={offset_rho:.6g}, ampl_rho={ampl_rho:.6g}, phi_rho={phi_rho:.6g} rad")
    print(f"offset_eps={offset_eps:.6g}, ampl_eps={ampl_eps:.6g}, phi_eps={phi_eps:.6g} rad")
    print(f"offset_vx={offset_vx:.6g}, ampl_vx={ampl_vx:.6g}, phi_vx={phi_vx:.6g} rad")

    print(f"eigenval = {eigval}")
    print(f"eigenvec = {eigvec}")

    coefs = project_eigenmode(eigvec, eigval, ampl_rho / rho, ampl_eps, -1j * ampl_vx / cs)
    print(f"coefs={coefs}")

    return coefs


# %%
# Curve fitting
from scipy.linalg import lstsq
from scipy.optimize import curve_fit


def fit_sine_wave(x, y, prev_phi=None):
    offset = np.mean(y)
    y0 = y - offset

    z = np.exp(1j * k * x)

    lam = np.vdot(z, y0) / np.vdot(z, z)

    ampl = 2 * np.abs(lam)
    phi = np.angle(lam)

    # enforce continuity with previous phase if available
    if prev_phi is not None:
        # candidate 1
        phi1 = phi
        a1 = ampl

        # candidate 2 (flip sign via phase shift)
        phi2 = phi + np.pi
        a2 = -ampl

        # choose whichever is closer to previous phase
        if abs(np.angle(np.exp(1j * (phi1 - prev_phi)))) > abs(
            np.angle(np.exp(1j * (phi2 - prev_phi)))
        ):
            phi, ampl = phi2, a2

    # wrap phase
    phi = np.mod(phi, 2 * np.pi)

    return offset, ampl, phi


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
    print(omega_k)

    print(f"k={k} cs={cs} ts={ts} epsilon_0={epsilon_0}")
    eigval, eigvec = eigensystem_dustywave_tvi(k, cs, ts, epsilon_0)

    print(f"eigenval = {eigval}")
    print(f"eigenvec = {eigvec}")

    Twave = 2 * np.pi / np.max(np.abs(np.real(eigval)))
    print(Twave)

    Twave_cnt = 40
    nwave = 2

    t_list = []
    rho_t_list = []
    eps_t_list = []
    vx_t_list = []
    rho_t_list_analytic = []
    eps_t_list_analytic = []
    vx_t_list_analytic = []

    rho_last_phi = np.pi
    eps_last_phi = 0
    vx_last_phi = 3 * np.pi / 2

    os.makedirs("_to_trash", exist_ok=True)
    for i in range(int(Twave_cnt * nwave)):
        t = Twave * i / (Twave_cnt)
        model.evolve_until(t)

        x_data, rho_data, rho_g_data, rho_d_data, vx_data, eps_data = get_field_results(model)

        x_ana = np.linspace(xm, xM, 256)

        if i == 0:
            coefs = find_eigen_decomp(x_data, rho_data, eps_data, vx_data, eigval, eigvec)
            print(f"coefs={coefs}")

            decomp = np.array(eigvec[0] * 0)
            for ieig in range(len(eigval)):
                decomp += coefs[ieig] * eigvec[:, ieig]
            print(f"decomp={decomp}")

        model_rho_on_rho_0 = np.zeros_like(x_ana, dtype=complex)
        model_eps = np.zeros_like(x_ana, dtype=complex)
        model_vx_on_cs = np.zeros_like(x_ana, dtype=complex)

        for ieig in range(len(eigval)):
            model_rho_on_rho_0 += eigen_model(
                x_ana, model.get_time(), 0.0, coefs[ieig] * eigvec[0, ieig], eigval[ieig], k
            )
            model_eps += eigen_model(
                x_ana, model.get_time(), 0.0, coefs[ieig] * eigvec[1, ieig], eigval[ieig], k
            )
            model_vx_on_cs += eigen_model(
                x_ana, model.get_time(), 0.0, coefs[ieig] * eigvec[2, ieig], eigval[ieig], k
            )

        model_rho = rho * np.real(model_rho_on_rho_0)
        model_eps = np.real(model_eps)
        model_vx = cs * np.real(model_vx_on_cs)

        model_rho += 1
        model_eps += 0.5
        model_vx += 0

        _, rho_t_ampl, rho_t_phi = fit_sine_wave(x_data, rho_data, rho_last_phi)
        _, eps_t_ampl, eps_t_phi = fit_sine_wave(x_data, eps_data, eps_last_phi)
        _, vx_t_ampl, vx_t_phi = fit_sine_wave(x_data, vx_data, vx_last_phi)
        _, rho_ana_ampl, _ = fit_sine_wave(x_ana, model_rho, rho_last_phi)
        _, eps_ana_ampl, _ = fit_sine_wave(x_ana, model_eps, eps_last_phi)
        _, vx_ana_ampl, _ = fit_sine_wave(x_ana, model_vx, vx_last_phi)

        print(f"rho_t_ampl={rho_t_ampl:.6g}, rho_t_phi={rho_t_phi:.6g} rad")
        print(f"eps_t_ampl={eps_t_ampl:.6g}, eps_t_phi={eps_t_phi:.6g} rad")
        print(f"vx_t_ampl={vx_t_ampl:.6g}, vx_t_phi={vx_t_phi:.6g} rad")

        t_list.append(model.get_time())
        rho_t_list.append(rho_t_ampl)
        eps_t_list.append(eps_t_ampl)
        vx_t_list.append(vx_t_ampl)
        rho_t_list_analytic.append(rho_ana_ampl)
        eps_t_list_analytic.append(eps_ana_ampl)
        vx_t_list_analytic.append(vx_ana_ampl)

        rho_last_phi = rho_t_phi
        eps_last_phi = eps_t_phi
        vx_last_phi = vx_t_phi

        fig, axs = plt.subplots(1, 1, figsize=(10, 5))

        axs.plot(x_data, rho_data - 1, ".", label=r"$\delta \rho$")
        axs.plot(x_data, eps_data - 0.5, ".", label=r"$\delta \epsilon$")
        axs.plot(x_data, vx_data / cs, ".", label=r"$\delta v_x / c_s$")

        axs.plot(x_ana, model_rho - 1, "-", label=r"$\delta \rho$ analytic")
        axs.plot(x_ana, model_eps - 0.5, "-", label=r"$\delta \epsilon$ analytic")
        axs.plot(x_ana, model_vx / cs, "-", label=r"$\delta v_x / c_s$ analytic")

        axs.set_xlabel(r"$x$ [code unit]")
        axs.set_ylabel(r"$\delta$ fields [code unit]")
        axs.set_xlim(xm, xM)
        # axs.set_ylim(rho / 2 - 1e-3, rho / 2 + 1e-3)
        axs.set_ylim(-1e-3, +1e-3)
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

        # if i == 1:
        #    break

    t_arr = np.asarray(t_list)

    rho_t_list = np.array(rho_t_list)
    eps_t_list = np.array(eps_t_list)
    vx_t_list = np.array(vx_t_list)
    rho_t_list_analytic = np.array(rho_t_list_analytic)
    eps_t_list_analytic = np.array(eps_t_list_analytic)
    vx_t_list_analytic = np.array(vx_t_list_analytic)

    plt.figure(dpi=150)
    plt.plot(t_arr, rho_t_list, ".", label=r"$\delta \rho (t)$")
    plt.plot(t_arr, eps_t_list, ".", label=r"$\delta \epsilon (t)$")
    plt.plot(t_arr, vx_t_list / cs, ".", label=r"$\delta v_x / c_s (t)$")
    plt.plot(t_arr, rho_t_list_analytic, "-", label=r"$\delta \rho (t)$ analytic")
    plt.plot(t_arr, eps_t_list_analytic, "-", label=r"$\delta \epsilon (t)$ analytic")
    plt.plot(t_arr, vx_t_list_analytic / cs, "-", label=r"$\delta v_x / c_s(t)$ analytic")
    plt.xlabel("$t$ [code unit]")
    plt.ylabel("$\delta$ fields [code unit]")
    plt.title(f"cs={cs:.6g}")
    plt.legend(fontsize=12, loc="upper right")
    plt.savefig(f"_to_trash/dustywave_tvi_scan_{ics:04}.png")

# %%
# make gifs
from matplotlib.animation import PillowWriter
from shamrock.utils.plot import show_image_sequence

keep_list = []

# %%
# show them the gifs (i have to unroll the loop otherwise the doc does not capture the gifs ...)
ani0 = show_image_sequence(f"_to_trash/dump_dustywave_tvi_{0:02d}_*.png")
writer = PillowWriter(fps=15, metadata=dict(artist="Me"), bitrate=1800)
ani0.save(f"_to_trash/dustywave_tvi_scan_{0:04}.gif", writer=writer)
plt.show()
# %%
ani1 = show_image_sequence(f"_to_trash/dump_dustywave_tvi_{1:02d}_*.png")
writer = PillowWriter(fps=15, metadata=dict(artist="Me"), bitrate=1800)
ani1.save(f"_to_trash/dustywave_tvi_scan_{1:04}.gif", writer=writer)
plt.show()
# %%
ani2 = show_image_sequence(f"_to_trash/dump_dustywave_tvi_{2:02d}_*.png")
writer = PillowWriter(fps=15, metadata=dict(artist="Me"), bitrate=1800)
ani2.save(f"_to_trash/dustywave_tvi_scan_{2:04}.gif", writer=writer)
plt.show()

plt.show()
