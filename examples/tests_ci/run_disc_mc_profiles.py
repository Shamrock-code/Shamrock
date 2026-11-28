"""
Disc Monte-Carlo setup density profiles
=================================================

Setup a disc with the Monte-Carlo disc generator, run a single step and
compare the resulting density profiles to the setup surface density profile.
"""

import matplotlib.pyplot as plt
import numpy as np

import shamrock

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

si = shamrock.UnitSystem()
sicte = shamrock.Constants(si)
codeu = shamrock.UnitSystem(
    unit_time=sicte.year() / (2 * np.pi),
    unit_length=sicte.au(),
    unit_mass=sicte.sol_mass(),
)
ucte = shamrock.Constants(codeu)
G = ucte.G()

R_in = 1.0
R_out = 10.0
R_ref = 1.0
disc_mass = 0.05
Npart = 100000
p_index_list = [1.0, 1.5, 2.0]
p_index = p_index_list[0]  # overwritten in the loop below
q_index = 0.50
H_R = 0.05
alpha_SS = 0.005
m1 = 1.0

pmass = disc_mass / Npart


# Disc profiles
def sigma_profile(r):
    sigma_0 = 1.0  # We do not care as it will be renormalized
    return sigma_0 * (r / R_ref) ** (-p_index)


def kep_profile(r):
    return (G * m1 / r) ** 0.5


def omega_k(r):
    return kep_profile(r) / r


def cs_profile(r):
    cs_in = (H_R * R_ref) * omega_k(R_ref)
    return ((r / R_ref) ** (-q_index)) * cs_in


def get_sigma_norm():
    x_list = np.linspace(R_in, R_out, 2048)
    term = [sigma_profile(x) * 2 * np.pi * x for x in x_list]
    return disc_mass / (np.sum(term) * (x_list[1] - x_list[0]))


def rot_profile(r):
    return ((kep_profile(r) ** 2) - (2 * p_index + q_index) * cs_profile(r) ** 2) ** 0.5


def H_profile(r):
    H = cs_profile(r) / omega_k(r)
    fact = 1.0
    return fact * H


def setup_shamrock_disc():
    ctx = shamrock.Context()
    ctx.pdata_layout_new()
    model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")

    alpha_AV = 0.0765404518492
    alpha_u = 1.0
    beta_AV = 2.0
    C_cour = 0.3
    C_force = 0.25
    cs0 = cs_profile(R_ref)
    bsize = R_out * 1.2

    # Generate the default config
    cfg = model.gen_default_config()
    cfg.set_artif_viscosity_ConstantDisc(alpha_u=alpha_u, alpha_AV=alpha_AV, beta_AV=beta_AV)
    cfg.set_eos_locally_isothermalLP07(cs0=cs0, q=q_index, r0=R_ref)

    cfg.add_kill_sphere(center=(0, 0, 0), radius=bsize)  # kill particles outside the simulation box

    cfg.set_units(codeu)
    cfg.set_particle_mass(pmass)
    # Set the CFL
    cfg.set_cfl_cour(C_cour)
    cfg.set_cfl_force(C_force)

    cfg.set_smoothing_length_density_based()

    # Set the solver config to be the one stored in cfg
    model.set_solver_config(cfg)

    # Print the solver config
    model.get_current_config().print_status()

    # Init the scheduler & fields
    model.init_scheduler(int(1e8), 1)

    # Set the simulation box size
    ext = R_out * 1.2
    model.resize_simulation_box((-ext, -ext, -ext), (ext, ext, ext))

    # Create the setup
    setup = model.get_setup()
    gen_disc = setup.make_generator_disc_mc(
        part_mass=pmass,
        disc_mass=disc_mass,
        r_in=R_in,
        r_out=R_out,
        sigma_profile=sigma_profile,
        H_profile=H_profile,
        rot_profile=rot_profile,
        cs_profile=cs_profile,
        random_seed=666,
    )

    # Apply the setup
    setup.apply_setup(gen_disc, insert_step=1000000)

    # now that the barycenter & momentum are 0, we can add the sink
    model.add_sink(m1, (0, 0, 0), (0, 0, 0), 1.0)

    return ctx, model


def positions_to_rays(positions):
    return [shamrock.math.Ray_f64_3(tuple(position), (0.0, 0.0, 1.0)) for position in positions]


def compute_avg_sigma_profile(model, ntheta, r):
    theta = np.linspace(0, 2 * np.pi, ntheta)

    r_grid, theta_grid = np.meshgrid(r, theta)
    x_grid = r_grid * np.cos(theta_grid)
    y_grid = r_grid * np.sin(theta_grid)
    z_grid = np.zeros_like(r_grid)

    positions = np.column_stack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()])

    rays = positions_to_rays(positions)
    arr_sigma = model.render_column_integ("rho", "f64", rays)

    arr_sigma = np.array(arr_sigma).reshape(ntheta, len(r))

    # average over the theta direction
    arr_sigma = np.mean(arr_sigma, axis=0)
    return arr_sigma


def get_profiles(model):
    x_list = np.linspace(0, R_out * 1.2, 2049)[1:]
    positions = [(x, 0.0, 0.0) for x in x_list.tolist()]

    arr_rho = np.array(model.render_slice("rho", "f64", positions))
    arr_sigma_avg = compute_avg_sigma_profile(model, 32, x_list)

    arr_sigma = np.array([sigma_profile(x) * sigma_norm for x in x_list])
    arr_sigma[(x_list < R_in) | (x_list > R_out)] = 0.0

    return x_list, arr_rho, arr_sigma_avg, arr_sigma


dpi = 200

fig_sigma, ax_sigma = plt.subplots(dpi=dpi)
fig_rho, ax_rho = plt.subplots(dpi=dpi)
fig_err, ax_err = plt.subplots(dpi=dpi)

max_delta_sigma = {}
fitted_slope = {}

for i, p_index in enumerate(p_index_list):
    # the disc profiles read p_index & sigma_norm from the module globals
    sigma_norm = get_sigma_norm()

    ctx, model = setup_shamrock_disc()

    # Run a single step so that the smoothing length & density are computed
    model.timestep()

    x_list, arr_rho, arr_sigma_avg, arr_sigma = get_profiles(model)

    del model
    del ctx

    H = np.array([H_profile(x) for x in x_list])
    # midplane density expected from the setup surface density profile
    arr_rho_setup = arr_sigma / (np.sqrt(2 * np.pi) * H)

    color = f"C{i}"
    lbl = f" (p={p_index})"

    # Surface density profile
    ax_sigma.plot(x_list, arr_sigma_avg, color=color, label="sigma avg (shamrock)" + lbl)
    ax_sigma.plot(x_list, arr_sigma, color=color, ls="--", label="sigma (setup)" + lbl)

    # Midplane density profile
    ax_rho.plot(x_list, arr_rho, color=color, label="rho (shamrock)" + lbl)
    ax_rho.plot(
        x_list,
        arr_sigma_avg / (np.sqrt(2 * np.pi) * H),
        color=color,
        ls=":",
        label="sigma avg/(sqrt(2*pi)*H)" + lbl,
    )
    ax_rho.plot(
        x_list, arr_rho_setup, color=color, ls="--", label="sigma/(sqrt(2*pi)*H) (setup)" + lbl
    )

    # Relative error on the surface density
    mask = arr_sigma > 0
    delta_sigma = np.zeros_like(arr_sigma)
    delta_sigma[mask] = (arr_sigma_avg[mask] - arr_sigma[mask]) / arr_sigma[mask]
    ax_err.plot(x_list, np.abs(delta_sigma), color=color, label="|delta sigma| / sigma" + lbl)

    # ignore the disc edges where the SPH smoothing makes the profile deviate
    mask_inner = (x_list > 2 * R_in) & (x_list < 0.8 * R_out)
    max_delta_sigma[p_index] = np.max(np.abs(delta_sigma[mask_inner]))

    # power law index of the measured surface density (should be -p_index)
    fitted_slope[p_index] = np.polyfit(
        np.log(x_list[mask_inner]), np.log(arr_sigma_avg[mask_inner]), 1
    )[0]

ax_sigma.set_xlabel("r")
ax_sigma.set_ylabel("sigma")
ax_sigma.set_title("Surface density after one step")
ax_sigma.legend(fontsize="small")
ax_sigma.set_ylim(1e-6, 1e-2)
ax_sigma.set_yscale("log")

ax_rho.set_xlabel("r")
ax_rho.set_ylabel("rho")
ax_rho.set_title("Midplane density after one step")
ax_rho.legend(fontsize="small")
ax_rho.set_ylim(1e-5, 1e-1)
ax_rho.set_yscale("log")

ax_err.set_xlabel("r")
ax_err.set_ylabel("relative error")
ax_err.set_yscale("log")
ax_err.set_title("Surface density relative error after one step")
ax_err.legend(fontsize="small")

plt.show()

# With a biased radial sampling (e.g. the rejection sampling bound being too low, which
# makes r uniform) the max relative error is ~0.36 for p=1.5 and ~1.05 for p=2,
# and the fitted slope is -1 regardless of p
max_rel_err_tol = 0.2
slope_tol = 0.1

errors = []
for p_index in p_index_list:
    err = max_delta_sigma[p_index]
    slope = fitted_slope[p_index]
    print(
        f"p={p_index}: max relative delta sigma (2 R_in < r < 0.8 R_out): {err:.4e}, "
        f"fitted slope: {slope:.4f} (expected {-p_index})"
    )

    if err > max_rel_err_tol:
        errors.append(f"p={p_index}: max relative delta sigma {err:.4e} > {max_rel_err_tol}")
    if abs(slope + p_index) > slope_tol:
        errors.append(
            f"p={p_index}: fitted slope {slope:.4f} differs from {-p_index} by more than {slope_tol}"
        )

if errors:
    raise ValueError("Disc MC profile check failed:\n" + "\n".join(errors))
