"""
Testing 3D circularly polarised Alfven wave with SPMHD
========================================================

CI test for the 3D circularly polarised Alfven wave (Toth 2000, 3D setup
of Gardiner & Stone 2008), following the description given in the Phantom
code paper (Price et al. 2018, Section 5.6.2 / 5.7.1).

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

# %%
gamma = 5.0 / 3.0
rho0 = 1.0
P0 = 0.1

wavelength = 1.0
amplitude = 0.1

# Alfven speed set by B1 = 1, rho = 1 -> v_A = 1, period = wavelength / v_A = 1
v_A = 1.0
n_periods = 5
t_target = n_periods * wavelength / v_A

resol = 32

plot_extra_resols = []  # 64, 128, too much for the ci

do_plot = True
dump_folder = "_to_trash"
if do_plot and shamrock.sys.world_rank() == 0:
    os.makedirs(dump_folder, exist_ok=True)


# %%
def best_transverse_counts(model, xcnt, target_ratio=0.5, search_frac=0.18):
    lo = max(2, int(xcnt * (target_ratio - search_frac)))
    hi = int(xcnt * (target_ratio + search_frac)) + 1
    lo += lo % 2
    best = None
    for ycnt in range(lo, hi, 2):
        for zcnt in range(lo, hi, 2):
            xs, ys, zs = model.get_box_dim_fcc_3d(1, xcnt, ycnt, zcnt)
            score = abs(ys / xs - target_ratio) + abs(zs / xs - target_ratio)
            if best is None or score < best[0]:
                best = (score, ycnt, zcnt)
    return best[1], best[2]


def wave_basis(xs, ys, zs):
    # Athena-style: r_hat proportional to (1/Lx, 1/Ly, 1/Lz) and lambda = 1/|(1/Lx, 1/Ly, 1/Lz)|,
    # so a periodic translation along any axis shifts x1 by exactly one
    # wavelength. A lattice-derived box is only approximately 2:1:1, and
    # keeping the paper's fixed angles would leave a phase seam at the y/z
    # boundaries. For an exact 3 x 1.5 x 1.5 box this reduces to
    # sin(a) = 2/3, sin(b) = 2/sqrt(5), lambda = 1.
    k = np.array([1.0 / xs, 1.0 / ys, 1.0 / zs])
    lam = 1.0 / np.linalg.norm(k)
    r_hat = k * lam

    sin_a = r_hat[2]
    cos_a = np.hypot(r_hat[0], r_hat[1])
    sin_b = r_hat[1] / cos_a
    cos_b = r_hat[0] / cos_a

    e2_hat = np.array([-sin_b, cos_b, 0.0])
    e3_hat = np.array([-sin_a * cos_b, -sin_a * sin_b, cos_a])
    return lam, r_hat, e2_hat, e3_hat


def run_alfven_wave(resol):
    ctx = shamrock.Context()
    ctx.pdata_layout_new()

    codeu = shamrock.UnitSystem(unit_time=1.0, unit_length=1.0, unit_mass=1.2566370621219e-06)

    model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")

    cfg = model.gen_default_config()
    cfg.set_units(codeu)
    cfg.set_artif_viscosity_None()
    cfg.set_IdealMHD(sigma_mhd=1, sigma_u=1)
    cfg.set_boundary_periodic()
    cfg.set_eos_adiabatic(gamma)
    model.set_solver_config(cfg)

    crit_split = int(1e7)
    crit_merge = 1
    model.init_scheduler(crit_split, crit_merge)

    ycnt, zcnt = best_transverse_counts(model, resol)

    # lambda scales linearly with dr: pick dr so that lambda == wavelength exactly
    lam_unit = wave_basis(*model.get_box_dim_fcc_3d(1, resol, ycnt, zcnt))[0]
    dr = wavelength / lam_unit
    (xs, ys, zs) = model.get_box_dim_fcc_3d(dr, resol, ycnt, zcnt)
    lam, r_hat, e2_hat, e3_hat = wave_basis(xs, ys, zs)
    rotmat = np.column_stack([r_hat, e2_hat, e3_hat])
    print(f"[resol={resol}] Box dims: xs={xs} ys={ys} zs={zs}")
    print(f"[resol={resol}] r_hat={r_hat} (paper: [1/3, 2/3, 2/3]) lambda={lam}")

    def vel_func(r):
        x1 = np.dot(r, r_hat)
        v_wave = np.array(
            [
                0.0,
                amplitude * np.sin(2.0 * np.pi * x1 / lam),
                amplitude * np.cos(2.0 * np.pi * x1 / lam),
            ]
        )
        return tuple(rotmat @ v_wave)

    def mag_func(r):
        x1 = np.dot(r, r_hat)
        B_wave = np.array(
            [
                1.0,
                amplitude * np.sin(2.0 * np.pi * x1 / lam),
                amplitude * np.cos(2.0 * np.pi * x1 / lam),
            ]
        )
        # field is stored as B/rho in SPMHD
        return tuple(rotmat @ B_wave / rho0)

    box_min = (-xs / 2, -ys / 2, -zs / 2)
    box_max = (xs / 2, ys / 2, zs / 2)

    model.resize_simulation_box(box_min, box_max)
    model.add_cube_fcc_3d(dr, box_min, box_max)

    gam1 = gamma - 1.0
    uuzero = P0 / (gam1 * rho0)
    model.set_value_in_a_box("uint", "f64", uuzero, box_min, box_max)

    model.set_field_value_lambda_f64_3("vxyz", vel_func)
    model.set_field_value_lambda_f64_3("B/rho", mag_func)

    vol_b = xs * ys * zs
    totmass = rho0 * vol_b
    pmass = model.total_mass_to_part_mass(totmass)
    model.set_particle_mass(pmass)

    print(f"[resol={resol}] Total mass :", totmass)
    print(f"[resol={resol}] Current part mass :", pmass)

    model.set_cfl_cour(0.3)
    model.set_cfl_force(0.25)

    model.timestep()

    model.evolve_until(t_target)

    # %%
    # Compare the transverse field component B2 against the exact solution.
    #
    # Because t_target is an integer number of wave periods, the exact
    # (undamped) solution coincides with the initial condition:
    #   B2_exact(x1) = amplitude * sin(2*pi*x1/wavelength)

    data = ctx.collect_data()

    xyz = data["xyz"]
    B_on_rho = data["B/rho"]

    x1 = xyz @ r_hat
    # SPMHD stores B/rho: recover the physical field before projecting it on e2
    rho = pmass * (model.get_hfact() / data["hpart"]) ** 3
    B2 = (B_on_rho * rho[:, None]) @ e2_hat

    return x1, B2


x1, B2 = run_alfven_wave(resol)

B2_exact = amplitude * np.sin(2.0 * np.pi * x1 / wavelength)

l2_err_B2 = np.sqrt(np.mean((B2 - B2_exact) ** 2))

print(f"L2 error on B2 : {l2_err_B2}")

# %%


# The box spans 3 wavelengths along x1, and the solution only depends on
# x1 modulo the wavelength, so fold x1 into a single period to overlay them.
def fold_x1(x1):
    return np.mod(x1 + 0.5 * wavelength, wavelength) - 0.5 * wavelength


if do_plot:
    all_resols = sorted(set([resol] + list(plot_extra_resols)))
    results = {res: (x1, B2) if res == resol else run_alfven_wave(res) for res in all_resols}

    # Fig. 26: B2 vs x1 (folded into one wavelength) for each resolution
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = plt.cm.viridis(np.linspace(0.0, 0.8, len(all_resols)))

    for color, res in zip(colors, all_resols):
        x1_r, B2_r = results[res]
        ax.plot(fold_x1(x1_r), B2_r, ".", color=color, markersize=1.5, label=f"resol={res}")

    x1_exact = np.linspace(-0.5 * wavelength, 0.5 * wavelength, 500)
    B2_exact_line = amplitude * np.sin(2.0 * np.pi * x1_exact / wavelength)
    ax.plot(x1_exact, B2_exact_line, "-", color="red", linewidth=1.5, label="exact solution")

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$B_2$")
    ax.set_title(f"3D circularly polarised Alfven wave, $t={t_target:.0f}$ periods")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(dump_folder, "mhd_alfven_wave_3d_b2_vs_x1.png"), dpi=150)
    plt.close(fig)

    nxs = np.array(all_resols, dtype=float)
    l1_errs = np.array(
        [
            np.mean(np.abs(B2_r - amplitude * np.sin(2.0 * np.pi * x1_r / wavelength)))
            for x1_r, B2_r in (results[res] for res in all_resols)
        ]
    )
    for res, err in zip(all_resols, l1_errs):
        print(f"L1 error on B2 (resol={res}) : {err}")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(nxs, l1_errs, "o-", color="black", label="Shamrock")
    if len(nxs) > 1:
        # Local order between consecutive resolutions: a single fit over all points
        # would be biased by coarse runs that are not yet in the asymptotic regime.
        orders = -np.diff(np.log(l1_errs)) / np.diff(np.log(nxs))
        for lo, hi, order in zip(all_resols[:-1], all_resols[1:], orders):
            print(f"L1 convergence order between resol={lo} and {hi} : {order}")
        ax.loglog(
            nxs,
            l1_errs[-1] * (nxs / nxs[-1]) ** -2,
            "--",
            color="gray",
            label="second order",
        )
        ax.set_title(f"Convergence, order between the two finest = {orders[-1]:.2f}")
    ax.set_xlabel("number of particles in x")
    ax.set_ylabel("L1 error on $B_2$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(dump_folder, "mhd_alfven_wave_3d_convergence.png"), dpi=150)
    plt.close(fig)

test_pass = True
err_log = ""

expect_l2_err_B2 = 0.07299812569247696
tol = 0.35  # too generous for now


def float_equal(val1, val2, prec):
    return abs(val1 - val2) < prec


if not float_equal(l2_err_B2, expect_l2_err_B2, tol * expect_l2_err_B2):
    err_log += "error on the L2 norm of B2 is outside of tolerances:\n"
    err_log += f"  expected error = {expect_l2_err_B2} +- {tol * expect_l2_err_B2}\n"
    err_log += f"  obtained error = {l2_err_B2} (relative error = {(l2_err_B2 - expect_l2_err_B2) / expect_l2_err_B2})\n"
    test_pass = False

if test_pass == False:
    exit("Test did not pass L2 margins : \n" + err_log)
