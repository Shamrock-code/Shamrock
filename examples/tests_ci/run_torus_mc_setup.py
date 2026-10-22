"""
Monte-Carlo polytropic torus setup
==================================

Check of ``make_generator_torus_mc`` against the analytic equilibrium
:py:class:`shamrock.phys.PolytropicTorus`:

- a Papaloizou & Pringle (1984) torus (constant angular momentum, point mass):
  sampled radial and vertical mass distributions, equilibrium velocity and
  internal energy, and that the torus stays in equilibrium over a short evolution;
- a magnetised Kato et al. (2004) like torus in the Paczynski-Wiita potential:
  poloidal field derived from ``A_phi = A0 * max(rho / rho0 - A_cut, 0)`` and its
  normalisation to the requested plasma beta.
"""

import matplotlib.pyplot as plt
import numpy as np

import shamrock

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

errors = []
is_rank0 = shamrock.sys.world_rank() == 0


def check(name, value, tol):
    print(f"{name:55s} = {value:.4e} (tol {tol:.1e})")
    if not value < tol:
        errors.append(f"{name} = {value:.4e} >= {tol}")


# %%
# Papaloizou-Pringle torus
# ^^^^^^^^^^^^^^^^^^^^^^^^
# Units with G = 1, central mass 1, pressure maximum at r0 = 1.

si = shamrock.UnitSystem()
sicte = shamrock.Constants(si)
codeu = shamrock.UnitSystem(unit_time=1.0 / np.sqrt(sicte.G()), unit_length=1.0, unit_mass=1.0)
ucte = shamrock.Constants(codeu)
G = ucte.G()

gamma = 5.0 / 3.0
n_poly = 1.5
r0 = 1.0
r_in = 0.7
torus_mass = 1e-3
Npart = 50000
pmass = torus_mass / Npart

torus = shamrock.phys.PolytropicTorus(GM=G, r_s=0.0, r0=r0, r_in=r_in, a_ang=0.0, n_poly=n_poly)
r_out = torus.get_r_out()
z_max = torus.get_z_max()
rho0 = torus_mass / torus.volume_integral()
print(f"r_out = {r_out}, z_max = {z_max}, rho0 = {rho0}")

ctx = shamrock.Context()
ctx.pdata_layout_new()
model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")

cfg = model.gen_default_config()
cfg.set_units(codeu)
cfg.set_artif_viscosity_VaryingCD10(
    alpha_min=0.0, alpha_max=1.0, sigma_decay=0.1, alpha_u=1.0, beta_AV=2.0
)
cfg.set_eos_adiabatic(gamma)
# h starts far above its root at the torus surface, where Newton alone diverges
cfg.set_smoothing_length_newton_fallback(True)
cfg.add_ext_force_point_mass(1.0, 0.5 * r_in)
cfg.set_particle_mass(pmass)
model.set_solver_config(cfg)
model.init_scheduler(int(1e8), 1)
model.resize_simulation_box((-2 * r_out,) * 3, (2 * r_out,) * 3)

setup = model.get_setup()
gen = setup.make_generator_torus_mc(
    part_mass=pmass,
    torus_mass=torus_mass,
    central_mass=1.0,
    r0=r0,
    r_in=r_in,
    a_ang=0.0,
    n_poly=n_poly,
    potential="point_mass",
    random_seed=42,
)
setup.apply_setup(gen)


def get_data():

    data = ctx.collect_data()
    if not is_rank0:
        return data, None, None, None, None
    xyz = np.asarray(data["xyz"])
    vxyz = np.asarray(data["vxyz"])
    w = np.hypot(xyz[:, 0], xyz[:, 1])
    z = xyz[:, 2]
    v_phi = (-vxyz[:, 0] * xyz[:, 1] + vxyz[:, 1] * xyz[:, 0]) / w
    v_R = (vxyz[:, 0] * xyz[:, 0] + vxyz[:, 1] * xyz[:, 1]) / w
    return data, w, z, v_phi, v_R


data, w, z, v_phi, v_R = get_data()

if is_rank0:
    check("particle count error", abs(len(w) - Npart), 1)
    inside = np.array([torus.rho_shape(a, b) > 0 for a, b in zip(w, z)])
    check("fraction of particles outside of the torus", 1 - np.mean(inside), 1e-12)

    v_an = np.array([torus.v_phi(a) for a in w])
    check("max |v_phi - v_phi_an| / v_phi_an", np.max(np.abs(v_phi - v_an) / v_an), 1e-12)
    check("max |v_R| / v_phi", np.max(np.abs(v_R) / v_an), 1e-12)

    u_an = np.array([torus.p_over_rho(a, b) for a, b in zip(w, z)]) / (gamma - 1)
    check(
        "max |u - u_an| / max(u_an)",
        np.max(np.abs(np.asarray(data["uint"]) - u_an)) / u_an.max(),
        1e-12,
    )

# %%
# Sampled mass distributions against the analytic ones (independent of the SPH density)

if is_rank0:
    nbins = 25
    w_edges = np.linspace(r_in, r_out, nbins + 1)
    z_edges = np.linspace(-z_max, z_max, nbins + 1)
    w_c = 0.5 * (w_edges[1:] + w_edges[:-1])
    z_c = 0.5 * (z_edges[1:] + z_edges[:-1])

    zq = np.linspace(-z_max, z_max, 801)
    wq = np.linspace(r_in, r_out, 801)
    dM_dw = np.array(
        [np.trapezoid([torus.rho_shape(a, b) for b in zq], zq) * 2 * np.pi * a for a in w_c]
    )
    dM_dz = np.array(
        [np.trapezoid([torus.rho_shape(a, b) * 2 * np.pi * a for a in wq], wq) for b in z_c]
    )
    dM_dw *= rho0
    dM_dz *= rho0

    hist_w = np.histogram(w, w_edges)[0] * pmass / np.diff(w_edges)
    hist_z = np.histogram(z, z_edges)[0] * pmass / np.diff(z_edges)

    # only bins holding enough particles for the Poisson noise to be small (~ 1/sqrt(N_bin))
    mask_w = dM_dw * np.diff(w_edges) / pmass > 1000
    mask_z = dM_dz * np.diff(z_edges) / pmass > 1000
    check(
        "max radial dM/dvarpi relative error",
        np.max(np.abs(hist_w[mask_w] / dM_dw[mask_w] - 1)),
        0.1,
    )
    check(
        "max vertical dM/dz relative error", np.max(np.abs(hist_z[mask_z] / dM_dz[mask_z] - 1)), 0.1
    )

    fig, axs = plt.subplots(1, 3, figsize=(14, 4), dpi=120)
    axs[0].scatter(w[::5], z[::5], s=0.2)
    axs[0].set_xlabel("$\\varpi$")
    axs[0].set_ylabel("z")
    axs[0].set_aspect("equal")
    axs[0].set_title("particles (1/5)")
    axs[1].plot(w_c, hist_w, "o", label="sampled")
    axs[1].plot(w_c, dM_dw, "-", label="analytic")
    axs[1].set_xlabel("$\\varpi$")
    axs[1].set_ylabel("dM / d$\\varpi$")
    axs[1].legend()
    axs[2].plot(z_c, hist_z, "o", label="sampled")
    axs[2].plot(z_c, dM_dz, "-", label="analytic")
    axs[2].set_xlabel("z")
    axs[2].set_ylabel("dM / dz")
    axs[2].legend()
    plt.tight_layout()
    plt.show()

# %%
# Short evolution: the torus must stay close to equilibrium (MC noise only)

T_orb = 2 * np.pi * r0**1.5
model.evolve_until(0.25 * T_orb)
data, w, z, v_phi, v_R = get_data()
if is_rank0:
    v_an = np.array([torus.v_phi(a) for a in w])
    check("rms v_R / v_phi after 1/4 orbit", np.sqrt(np.mean((v_R / v_an) ** 2)), 0.05)
    hist_w = np.histogram(w, w_edges)[0] * pmass / np.diff(w_edges)
    check(
        "max radial dM/dvarpi relative error after 1/4 orbit",
        np.max(np.abs(hist_w[mask_w] / dM_dw[mask_w] - 1)),
        0.15,
    )

del model
del ctx

# %%
# Magnetised torus in the Paczynski-Wiita potential
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Kato et al. (2004) like torus, units r_s, r_s / c, M (c = 1, G = 1/2).

bh_mass = 10 * sicte.sol_mass()
r_s_si = 2 * sicte.G() * bh_mass / sicte.c() ** 2
codeu = shamrock.UnitSystem(unit_time=r_s_si / sicte.c(), unit_length=r_s_si, unit_mass=bh_mass)
ucte = shamrock.Constants(codeu)
GM = ucte.G()
r_s = 2 * GM / ucte.c() ** 2
mu_0 = ucte.mu_0()

plasma_beta = 10.0
A_cut = 0.2
Npart = 20000
pmass = torus_mass / Npart

torus = shamrock.phys.PolytropicTorus(GM=GM, r_s=r_s, r0=40.0, r_in=20.0, a_ang=0.46, n_poly=3.0)
rho0 = torus_mass / torus.volume_integral()
A0 = np.sqrt(
    2
    * mu_0
    * rho0
    * torus.pressure_integral()
    / (plasma_beta * torus.b_poloidal_unit_sq_integral(A_cut))
)
print(f"r_cusp = {torus.get_r_cusp()}, r_out = {torus.get_r_out()}, z_max = {torus.get_z_max()}")

ctx = shamrock.Context()
ctx.pdata_layout_new()
model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M6")

cfg = model.gen_default_config()
cfg.set_units(codeu)
cfg.set_artif_viscosity_Constant(alpha_u=1, alpha_AV=1, beta_AV=2)
cfg.set_IdealMHD(sigma_mhd=1.0)
cfg.set_eos_adiabatic(gamma)
# h starts far above its root at the torus surface, where Newton alone diverges
cfg.set_smoothing_length_newton_fallback(True)
cfg.add_ext_force_paczynski_wiita(1.0, (0.0, 0.0, 0.0), 2.0)
cfg.set_particle_mass(pmass)
model.set_solver_config(cfg)
model.init_scheduler(int(1e8), 1)
model.resize_simulation_box((-100,) * 3, (100,) * 3)

setup = model.get_setup()
gen = setup.make_generator_torus_mc(
    part_mass=pmass,
    torus_mass=torus_mass,
    central_mass=1.0,
    r0=40.0,
    r_in=20.0,
    a_ang=0.46,
    n_poly=3.0,
    potential="paczynski_wiita",
    plasma_beta=plasma_beta,
    A_cut=A_cut,
    random_seed=42,
)
setup.apply_setup(gen)

data = ctx.collect_data()
if is_rank0:
    xyz = np.asarray(data["xyz"])
    B_on_rho = np.asarray(data["B/rho"])
    w = np.hypot(xyz[:, 0], xyz[:, 1])
    z = xyz[:, 2]
    rho_an = rho0 * np.array([torus.rho_shape(a, b) for a, b in zip(w, z)])
    b_an = np.array([torus.b_poloidal_unit(a, b, A_cut) for a, b in zip(w, z)]) * A0

    B_on_rho_an = np.zeros_like(B_on_rho)
    B_on_rho_an[:, 0] = b_an[:, 0] * xyz[:, 0] / w / rho_an
    B_on_rho_an[:, 1] = b_an[:, 0] * xyz[:, 1] / w / rho_an
    B_on_rho_an[:, 2] = b_an[:, 1] / rho_an
    check(
        "max |B/rho - (B/rho)_an| / max|(B/rho)_an|",
        np.max(np.abs(B_on_rho - B_on_rho_an)) / np.max(np.abs(B_on_rho_an)),
        1e-12,
    )

    # Monte Carlo estimate of int P dV / int P_mag dV with dV = m / rho_an
    P_on_rho = np.array([torus.p_over_rho(a, b) for a, b in zip(w, z)])
    Pmag_on_rho = np.sum((B_on_rho_an * rho_an[:, None]) ** 2, axis=1) / (2 * mu_0) / rho_an
    beta_mc = np.sum(P_on_rho) / np.sum(Pmag_on_rho)
    check("|beta_MC / beta - 1|", abs(beta_mc / plasma_beta - 1), 0.05)

# The SPH density of a Monte Carlo distribution is biased high (self contribution), so the
# beta measured with it is lower than the requested one
model.timestep()

if errors:
    raise ValueError("Torus MC setup check failed:\n" + "\n".join(errors))
