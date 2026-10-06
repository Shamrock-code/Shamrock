"""
Testing Orszag-Tang vortex (3D thin box) with SPMHD
=====================================================

CI test for the 3D (thin box) Orszag-Tang vortex, following the setup
described in the Phantom code paper (Price et al. 2018, Section 5.7.3):
a uniform density, periodic box with a sinusoidal velocity and magnetic
field perturbation that develops into a system of interacting MHD shocks.

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
# Parameters (Price et al. 2018, Sec 5.7.3)

kernel = "M4"

C_cour = 0.3
C_force = 0.25

nx = 32
ny = 32
nz = 3

xymin = -0.5
xmin = xymin
ymin = xymin

gamma = 5.0 / 3.0
betazero = 10.0 / 3.0
machzero = 1.0
vzero = 1.0
bzero = 1.0 / np.sqrt(4.0 * np.pi)

przero = 0.5 * bzero**2 * betazero
rhozero = gamma * przero * machzero
gam1 = gamma - 1.0
uuzero = przero / (gam1 * rhozero)

t_target = 0.5


def vel_func(r):
    x, y, z = r
    vx = -vzero * np.sin(2.0 * np.pi * (y - ymin))
    vy = vzero * np.sin(2.0 * np.pi * (x - xmin))
    return (vx, vy, 0.0)


def mag_func(r):
    x, y, z = r
    Bx = -bzero * np.sin(2.0 * np.pi * (y - ymin)) / rhozero
    By = bzero * np.sin(4.0 * np.pi * (x - xmin)) / rhozero
    return (Bx, By, 0.0)


def run_orszag_tang(nx, ny, nz=3):
    ctx = shamrock.Context()
    ctx.pdata_layout_new()

    codeu = shamrock.UnitSystem(unit_time=1.0, unit_length=1.0, unit_mass=1.2566370621219e-06)
    ucte = shamrock.Constants(codeu)
    model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel=kernel)

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

    (xs, ys, zs) = model.get_box_dim_fcc_3d(1, nx, ny, nz)
    dr = 1 / xs
    (xs, ys, zs) = model.get_box_dim_fcc_3d(dr, nx, ny, nz)

    model.resize_simulation_box((-xs / 2, -ys / 2, -zs / 2), (xs / 2, ys / 2, zs / 2))
    model.add_cube_fcc_3d(dr, (-xs / 2, -ys / 2, -zs / 2), (xs / 2, ys / 2, zs / 2))
    model.set_value_in_a_box(
        "uint", "f64", uuzero, (-xs / 2, -ys / 2, -zs / 2), (xs / 2, ys / 2, zs / 2)
    )

    model.set_field_value_lambda_f64_3("vxyz", vel_func)
    model.set_field_value_lambda_f64_3("B/rho", mag_func)

    vol_b = xs * ys * zs
    totmass = rhozero * vol_b
    pmass = model.total_mass_to_part_mass(totmass)
    model.set_particle_mass(pmass)

    print(f"[nx={nx}] Total mass :", totmass)
    print(f"[nx={nx}] Current part mass :", pmass)

    model.set_cfl_cour(C_cour)
    model.set_cfl_force(C_force)

    model.timestep()

    return ctx, model, xs, ys, zs, pmass, ucte.mu_0()


ctx, model, xs, ys, zs, pmass, mu_0 = run_orszag_tang(nx, ny, nz)

model.evolve_until(t_target)

# %%
# Diagnostics

kinetic_energy = shamrock.model_sph.analysisEnergyKinetic(model=model).get_kinetic_energy()
total_momentum = shamrock.model_sph.analysisTotalMomentum(model=model).get_total_momentum()

data = ctx.collect_data()
hpart = data["hpart"]
hfact = model.get_hfact()
rho = pmass * (hfact / hpart) ** 3
B_on_rho = data["B/rho"]

magnetic_energy = 0.5 * pmass * np.sum(rho * np.sum(B_on_rho**2, axis=1)) / mu_0

rho_max = np.max(rho)
rho_min = np.min(rho)

momentum_norm = np.sqrt(total_momentum[0] ** 2 + total_momentum[1] ** 2 + total_momentum[2] ** 2)

print(f"kinetic_energy   = {kinetic_energy}")
print(f"magnetic_energy  = {magnetic_energy}")
print(f"momentum_norm    = {momentum_norm}")
print(f"rho_max          = {rho_max}")
print(f"rho_min          = {rho_min}")


plot_extra_resols = []  # 128, 148

do_plot = False
dump_folder = "_to_trash"
if do_plot and shamrock.sys.world_rank() == 0:
    os.makedirs(dump_folder, exist_ok=True)


def render_density_z0(model, xs, ys, render_res=300, min_normalization=1e-9):
    kwargs = dict(
        center=(0.0, 0.0, 0.0),
        delta_x=(xs, 0.0, 0.0),
        delta_y=(0.0, ys, 0.0),
        nx=render_res,
        ny=render_res,
    )
    raw = np.asarray(model.render_cartesian_slice("rho", "f64", **kwargs))
    unity = np.asarray(model.render_cartesian_slice("unity", "f64", **kwargs))
    return np.where(unity < min_normalization, np.nan, raw / unity)


def pressure_cut(model, y0, xs, n_pts=400, min_normalization=1e-9):
    x_arr = np.linspace(-xs / 2, xs / 2, n_pts)
    positions = [(x, y0, 0.0) for x in x_arr]

    rho_field = model.compute_field("rho", "f64")
    uint_field = model.compute_field("uint", "f64")

    def compute_pressure(size, rho, uint):
        return (gamma - 1.0) * rho * uint

    P_field = shamrock.map_fields_f64(compute_pressure, rho=rho_field, uint=uint_field)

    P_raw = np.asarray(model.render_slice(P_field, positions))
    unity = np.asarray(model.render_slice("unity", "f64", positions))
    P_cut = np.where(unity < min_normalization, np.nan, P_raw / unity)
    return x_arr, P_cut


if do_plot:
    y_cuts = [0.3125, 0.4277]

    # snapshot at t=0.5 (already reached above) for the primary resolution
    resols = [(nx, ny)]
    density_t05 = [render_density_z0(model, xs, ys)]
    cuts_t05 = [[pressure_cut(model, y0, xs) for y0 in y_cuts]]

    # continue to t=1 for the Fig. 32 bottom row
    model.evolve_until(1.0)
    density_t10 = [render_density_z0(model, xs, ys)]

    for nx_e, ny_e in plot_extra_resols:
        ctx_e, model_e, xs_e, ys_e, zs_e, pmass_e, mu_0_e = run_orszag_tang(nx_e, ny_e, nz)

        model_e.evolve_until(0.5)
        resols.append((nx_e, ny_e))
        density_t05.append(render_density_z0(model_e, xs_e, ys_e))
        cuts_t05.append([pressure_cut(model_e, y0, xs_e) for y0 in y_cuts])

        model_e.evolve_until(1.0)
        density_t10.append(render_density_z0(model_e, xs_e, ys_e))

    if shamrock.sys.world_rank() == 0:
        # Figure 32: density z=0, rows = t=0.5 / t=1, columns = resolutions
        n_res = len(resols)
        fig32, axs32 = plt.subplots(2, n_res, figsize=(max(4 * n_res, 6), 8), squeeze=False)
        for icol, (nxr, nyr) in enumerate(resols):
            extent = [-xs / 2, xs / 2, -ys / 2, ys / 2]
            axs32[0, icol].imshow(
                density_t05[icol], origin="lower", extent=extent, cmap="gist_heat"
            )
            axs32[0, icol].set_title(f"nx={nxr}, t=0.5")
            axs32[1, icol].imshow(
                density_t10[icol], origin="lower", extent=extent, cmap="gist_heat"
            )
            axs32[1, icol].set_title(f"nx={nxr}, t=1")
            for ax in (axs32[0, icol], axs32[1, icol]):
                ax.set_xlabel("x")
                ax.set_ylabel("y")
        fig32.suptitle("Orszag-Tang vortex: density in a z=0 cross section")
        fig32.tight_layout()
        fig32.savefig(os.path.join(dump_folder, "mhd_orszag_tang_density_z0.png"), dpi=150)
        plt.close(fig32)

        # Figure 33: horizontal pressure cuts at t=0.5, one panel per y0,
        # one line per resolution
        fig33, axs33 = plt.subplots(len(y_cuts), 1, figsize=(7, 8), squeeze=False)
        axs33 = axs33[:, 0]
        for irow, y0 in enumerate(y_cuts):
            for (nxr, nyr), cuts in zip(resols, cuts_t05):
                x_arr, P_cut = cuts[irow]
                axs33[irow].plot(x_arr, P_cut, label=f"nx={nxr}")
            axs33[irow].set_title(f"y={y0}")
            axs33[irow].set_xlabel("x")
            axs33[irow].set_ylabel("P")
            axs33[irow].legend()
        fig33.suptitle("Orszag-Tang vortex: horizontal pressure cuts (z=0, t=0.5)")
        fig33.tight_layout()
        fig33.savefig(os.path.join(dump_folder, "mhd_orszag_tang_pressure_cuts.png"), dpi=150)
        plt.close(fig33)

test_pass = True
err_log = ""


expect_kinetic_energy = 0.0021986783020805675
expect_magnetic_energy = 0.0029601504573239087
expect_rho_max = 0.3400457561374047
expect_rho_min = 0.14169354827263864

tol = 1e-4  # relative tolerance
max_momentum_norm = 1e-3


def float_equal(val1, val2, prec):
    return abs(val1 - val2) < prec


if not float_equal(kinetic_energy, expect_kinetic_energy, tol * expect_kinetic_energy):
    err_log += "error on the kinetic energy is outside of tolerances:\n"
    err_log += f"  expected error = {expect_kinetic_energy} +- {tol * expect_kinetic_energy}\n"
    err_log += f"  obtained error = {kinetic_energy}\n"
    test_pass = False

if not float_equal(magnetic_energy, expect_magnetic_energy, tol * expect_magnetic_energy):
    err_log += "error on the magnetic energy is outside of tolerances:\n"
    err_log += f"  expected error = {expect_magnetic_energy} +- {tol * expect_magnetic_energy}\n"
    err_log += f"  obtained error = {magnetic_energy}\n"
    test_pass = False

if momentum_norm > max_momentum_norm:
    err_log += "total momentum norm is outside of tolerances:\n"
    err_log += f"  expected |p| < {max_momentum_norm}\n"
    err_log += f"  obtained |p| = {momentum_norm}\n"
    test_pass = False

if not float_equal(rho_max, expect_rho_max, tol * expect_rho_max):
    err_log += "error on rho_max is outside of tolerances:\n"
    err_log += f"  expected error = {expect_rho_max} +- {tol * expect_rho_max}\n"
    err_log += f"  obtained error = {rho_max}\n"
    test_pass = False

if not float_equal(rho_min, expect_rho_min, tol * expect_rho_min):
    err_log += "error on rho_min is outside of tolerances:\n"
    err_log += f"  expected error = {expect_rho_min} +- {tol * expect_rho_min}\n"
    err_log += f"  obtained error = {rho_min}\n"
    test_pass = False

if test_pass == False:
    exit("Test did not pass L2 margins : \n" + err_log)
