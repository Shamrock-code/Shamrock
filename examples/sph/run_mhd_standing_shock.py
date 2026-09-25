"""
Standing shock test for the Hall term in non-ideal MHD
=====================================================

Reproduces the "standing shock" test of Falle (2003) and O'Sullivan & Downes
(2006)as in section 5.7.2 of Price et al. (2018). It consists in
 relaxing a 1D discontinuity toward the
analytic steady-state (Falle/O'Sullivan-Downes) shock profile.

currents issues:
-uses periodic boundaries in all three directions, instead ixed/inflow boundary particles
and an inflow-adjusted asymmetric domain
-reduced resolution (512x14x15 / 781x12x12 in Phantom paper)
"""

# sphinx_gallery_multi_image = "single"

import os

import matplotlib.pyplot as plt
import numpy as np

import shamrock

shamrock.enable_experimental_features()

if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

shamrock.matplotlib.set_shamrock_mpl_style()

# %%
rho_L, vx_L, vy_L, vz_L, By_L, Bz_L = 1.7942, -0.9759, -0.6561, 0.0, 1.74885, 0.0
rho_R, vx_R, vy_R, vz_R, By_R, Bz_R = 1.0, -1.751, 0.0, 0.0, 0.6, 0.0
Bx0 = 1.0
cs = 0.1

COR = 1.12e-12
CHE = -3.53e-2
CAD = 7.83e-3

t_end = 0.3

nx_L, ny, nz = 64, 8, 8
half_width = 2.0

# code units with mu_0 = 1
codeu = shamrock.UnitSystem(
    unit_time=1.0,
    unit_length=1.0,
    unit_mass=1.2566370621219e-06,
)
ucte = shamrock.Constants(codeu)
mu_0 = ucte.mu_0()

# %%
# Analytic steady-state solution

# The steady 1D isothermal MHD equations with Bx = const reduce (Falle 2003,
# O'Sullivan & Downes 2006) to a first-order ODE system for By(x), Bz(x).
# Mass and transverse-momentum conservation give vx, vy, vz as algebraic
# functions of (By, Bz); the non-ideal induction equation, integrated once in
# x using that the upstream (left) state is field-gradient-free, gives a
# linear 2x2 system for (dBy/dx, dBz/dx).

Q = rho_L * vx_L
Ky = Q * vy_L - Bx0 * By_L
Kz = Q * vz_L - Bx0 * Bz_L
Kx = Q * vx_L + cs**2 * rho_L + 0.5 * (By_L**2 + Bz_L**2)


def vx_of_B(By, Bz):
    Bperp2 = By**2 + Bz**2
    a = Kx - 0.5 * Bperp2
    disc = max(a * a - 4 * cs**2 * Q**2, 0.0)
    # root branch chosen so that plugging in the left state reproduces vx_L exactly
    return (a + np.sqrt(disc)) / (2 * Q)


def vy_of_B(By):
    return (Ky + Bx0 * By) / Q


def vz_of_B(Bz):
    return (Kz + Bx0 * Bz) / Q


def M1M2(By, Bz):
    vx, vy, vz = vx_of_B(By, Bz), vy_of_B(By), vz_of_B(Bz)
    M1 = vx * By - vy * Bx0 - (vx_L * By_L - vy_L * Bx0)
    M2 = vx * Bz - vz * Bx0 - (vx_L * Bz_L - vz_L * Bx0)
    return M1, M2


assert np.allclose(M1M2(By_L, Bz_L), 0.0, atol=1e-8)
assert np.allclose(M1M2(By_R, Bz_R), 0.0, atol=1e-3)


def shock_rhs(By, Bz):
    B2 = Bx0**2 + By**2 + Bz**2
    B = np.sqrt(B2)
    bx, by, bz = Bx0 / B, By / B, Bz / B

    etaO = COR
    etaH = CHE * B
    vx = vx_of_B(By, Bz)
    rho = Q / vx
    etaAD = CAD * (B2 / rho)  # CAD * vA^2, mu_0 = 1

    R11 = etaO + etaAD * (bx**2 + by**2)
    R12 = etaH * bx + etaAD * by * bz
    R21 = -etaH * bx + etaAD * by * bz
    R22 = etaO + etaAD * (bx**2 + bz**2)

    M1, M2 = M1M2(By, Bz)
    det = R11 * R22 - R12 * R21
    dBy = (M1 * R22 - M2 * R12) / det
    dBz = (R11 * M2 - R21 * M1) / det
    return dBy, dBz


def _jacobian_at_left(eps=1e-6):
    J = np.zeros((2, 2))
    for j, (dBy_ax, dBz_ax) in enumerate([(eps, 0.0), (0.0, eps)]):
        fp = shock_rhs(By_L + dBy_ax, Bz_L + dBz_ax)
        fm = shock_rhs(By_L - dBy_ax, Bz_L - dBz_ax)
        J[0, j] = (fp[0] - fm[0]) / (2 * eps)
        J[1, j] = (fp[1] - fm[1]) / (2 * eps)
    return J


def _unstable_eigendirection():
    w, v = np.linalg.eig(_jacobian_at_left())
    idx = np.argmax(w.real)
    vec = v[:, idx].real
    return vec / np.linalg.norm(vec)


def integrate_shock_profile(x_l, x_r, n=20000, epsilon=-1e-4):
    """RK4 shooting solution from the left (upstream) boundary, perturbed
    along the unstable manifold, integrated forward in x."""
    eigvec = _unstable_eigendirection()
    xs = np.linspace(x_l, x_r, n)
    dx = xs[1] - xs[0]
    By = np.empty(n)
    Bz = np.empty(n)
    By[0] = By_L + epsilon * eigvec[0]
    Bz[0] = Bz_L + epsilon * eigvec[1]
    for i in range(n - 1):
        k1 = shock_rhs(By[i], Bz[i])
        k2 = shock_rhs(By[i] + 0.5 * dx * k1[0], Bz[i] + 0.5 * dx * k1[1])
        k3 = shock_rhs(By[i] + 0.5 * dx * k2[0], Bz[i] + 0.5 * dx * k2[1])
        k4 = shock_rhs(By[i] + dx * k3[0], Bz[i] + dx * k3[1])
        By[i + 1] = By[i] + (dx / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        Bz[i + 1] = Bz[i] + (dx / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    return xs, By, Bz


xs_th, By_th, Bz_th = integrate_shock_profile(-half_width, half_width)
vx_th = np.array([vx_of_B(b, z) for b, z in zip(By_th, Bz_th)])
print(
    f"analytic shooting solution: final By={By_th[-1]:.4f} (target {By_R}), "
    f"final Bz={Bz_th[-1]:.4f} (target {Bz_R})"
)

# %%
ctx = shamrock.Context()
ctx.pdata_layout_new()
model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="C4")

cfg = model.gen_default_config()
cfg.set_units(codeu)
cfg.set_artif_viscosity_Constant(alpha_u=1.0, alpha_AV=1.0, beta_AV=2.0)
cfg.set_NonIdealMHD(
    sigma_mhd=0,
    sigma_u=0,
    etaO=0,
    etaH=0,
    etaAD=0,
    alpha_B=1.0,
    alpha_AV=1.0,
    beta_AV=2.0,
    eta_fields=True,
)
cfg.set_boundary_periodic()
cfg.set_eos_isothermal(cs)
cfg.print_status()
model.set_solver_config(cfg)

model.init_scheduler(int(1e7), 1)

(xs, ys, zs) = model.get_box_dim_fcc_3d(1, nx_L, ny, nz)
dr_L = half_width / xs
(xs, ys, zs) = model.get_box_dim_fcc_3d(dr_L, nx_L, ny, nz)

dr_R = dr_L * (rho_L / rho_R) ** (1.0 / 3.0)

model.resize_simulation_box((-xs, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))

V_L_min = (-xs, -ys / 2, -zs / 2)
V_L_max = (0, ys / 2, zs / 2)
V_R_min = (0, -ys / 2, -zs / 2)
V_R_max = (xs, ys / 2, zs / 2)

setup = model.get_setup()
gen_L = setup.make_generator_lattice_hcp(dr_L, V_L_min, V_L_max)
gen_R = setup.make_generator_lattice_hcp(dr_R, V_R_min, V_R_max)
comb = setup.make_combiner_add(gen_L, gen_R)
setup.apply_setup(comb)

vol_b = xs * ys * zs
totmass = rho_L * vol_b + rho_R * vol_b
pmass = model.total_mass_to_part_mass(totmass)
model.set_particle_mass(pmass)

u_iso = 0.0


def vel_func(r):
    x, y, z = r
    return (vx_L, vy_L, vz_L) if x < 0 else (vx_R, vy_R, vz_R)


def B_func(r):
    x, y, z = r
    By = By_L if x < 0 else By_R
    return (Bx0, By, 0.0)


def u_func(r):
    return u_iso


model.set_field_value_lambda_f64_3("vxyz", vel_func)
model.set_field_value_lambda_f64_3("B/rho", B_func)
model.set_field_value_lambda_f64("uint", u_func)


def eta_o_func(r):
    return COR


def eta_h_func(r):
    x, y, z = r
    By = By_L if x < 0 else By_R
    Bmag = np.sqrt(Bx0**2 + By**2)
    return CHE * Bmag


def eta_ad_func(r):
    x, y, z = r
    By = By_L if x < 0 else By_R
    rho = rho_L if x < 0 else rho_R
    Bmag2 = Bx0**2 + By**2
    vA2 = Bmag2 / rho  # mu_0 = 1
    return CAD * vA2


model.set_field_value_lambda_f64("eta_o", eta_o_func)
model.set_field_value_lambda_f64("eta_h", eta_h_func)
model.set_field_value_lambda_f64("eta_ad", eta_ad_func)

model.set_cfl_cour(0.3)
model.set_cfl_force(0.25)

dump_folder = "_to_trash/standing_shock"
if shamrock.sys.world_rank() == 0:
    os.makedirs(dump_folder, exist_ok=True)

model.evolve_until(t_end)

data = ctx.collect_data()
x = data["xyz"][:, 0]
vx = data["vxyz"][:, 0]
hpart = data["hpart"]
By_sim = data["B/rho"][:, 1] * pmass * (model.get_hfact() / hpart) ** 3

By_mid = 0.5 * (By_L + By_R)
i_th = np.argmin(np.abs(By_th - By_mid))
x0_th = xs_th[i_th]

order = np.argsort(x)
x_sorted, By_sim_sorted = x[order], By_sim[order]
bins = np.linspace(x.min(), x.max(), 60)
By_binned = np.array(
    [
        By_sim_sorted[(x_sorted >= bins[i]) & (x_sorted < bins[i + 1])].mean()
        for i in range(len(bins) - 1)
    ]
)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
valid = ~np.isnan(By_binned)
i_sim = np.argmin(np.abs(By_binned[valid] - By_mid))
x0_sim = bin_centers[valid][i_sim]

x_shift = x0_sim - x0_th

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
xlim = (x.min() - 0.2, x.max() + 0.2)

axs[0].plot(x, vx, ".", markersize=1, alpha=0.3, label="SPH")
axs[0].plot(xs_th + x_shift, vx_th, "r-", linewidth=2, label="analytic (shifted)")
axs[0].set_xlabel("x")
axs[0].set_ylabel("vx")
axs[0].set_xlim(xlim)
axs[0].legend()
axs[0].grid(alpha=0.3)

axs[1].plot(x, By_sim, ".", markersize=1, alpha=0.3, label="SPH")
axs[1].plot(xs_th + x_shift, By_th, "r-", linewidth=2, label="analytic (shifted)")
axs[1].set_xlabel("x")
axs[1].set_ylabel("By")
axs[1].set_xlim(xlim)
axs[1].legend()
axs[1].grid(alpha=0.3)

plt.suptitle(f"Standing shock test, t = {model.get_time():.3f}")
plt.tight_layout()
plt.savefig(os.path.join(dump_folder, "standing_shock.png"), dpi=150)
plt.show()
