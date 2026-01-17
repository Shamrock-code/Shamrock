import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate

import shamrock

si = shamrock.UnitSystem()
sicte = shamrock.Constants(si)


# Differential equation
def deriv(y, t):
    y_0, y_1 = y
    dydt = [y_1, np.exp(-y_0) - (2 / t) * y_1]
    return dydt


cs = 190.0  # m/s
G = sicte.G()
M = 1.0 * sicte.sol_mass()  # kg
dens_fact = 1.01
num_pts = 1000

xi_in = 1e-7
xi_out = 10

t = np.linspace(xi_in, xi_out, num_pts)

# initial condition
y0 = [0, 0]
sol = scipy.integrate.odeint(deriv, y0, t)

xidudxi = t * t * sol[:, 1]
dens = np.exp(-sol[:, 0])

plt.plot(t, sol[:, 0], "b", label="$\Psi(x)$")
plt.plot(t, sol[:, 1], "g", label="$\dot{\Psi(x)}$")
plt.legend(loc="best")
plt.xlabel("x")
plt.grid()


rho_c_rho_0 = []
dimless_arr = []
for i in range(len(t)):

    dimless_mass = xidudxi[i] * (4 * np.pi * dens[i] ** -1) ** (-1 / 2)
    # print("rho_c/rho_0",dens[i]**-1)
    # print("xidudxi[-1]",xidudxi[i])
    # print("(4*np.pi*dens[-1]**-1)**(1/2)",(4*np.pi*dens[i]**-1)**(-1/2))
    # print(xidudxi[i]*(4*np.pi*dens[i]**-1)**(-1/2))

    rho_c_rho_0.append(dens[i] ** -1)
    dimless_arr.append(dimless_mass)

plt.figure()
plt.plot(t, dens, "-", label="rho(x)")
plt.xlabel("x")
plt.ylabel(r"$\rho(x)$")
plt.xscale("log")
plt.yscale("log")
plt.grid()

plt.figure()
plt.plot(t, xidudxi, "-")
plt.xlabel("x")
plt.ylabel(r"$\xi^2 \dot{u}(\xi)$")
plt.xscale("log")
plt.yscale("log")
plt.grid()

plt.figure()
plt.plot(rho_c_rho_0, dimless_arr)
plt.xlabel(r"$\rho_c / \rho$")
plt.ylabel("m")
plt.xscale("log")


print("inner to outer density ratio  :", "{:.3f}".format(dens[-1] ** -1))

# print(xidudxi[-1]**2)
# print(M,G**(3/2),cs**(-3))
# print((M*G**(3/2)*cs**(-3))**(-2))

# print((mpfr(xidudxi[-1])**2)*(  (cs**(-3) * G**(3/2) * M)**(-2) ))

print("final density                 :", "{:.2E}".format(dens[-1]))


I = scipy.integrate.simpson(dens * t * t * 4 * np.pi, t)
print("integral before normalisation :", "{:.2E}".format(I))


beta = cs**2 / (4 * np.pi * G)
print("beta              : ", "{:.2E}".format(beta))
# beta = 2.13e13
lambda_ = (beta ** (3 / 2) * I / M) ** 2
print("peak density      : ", "{:.2E}".format(lambda_ / 1000))
print(
    "outer radius (pc) : ",
    "{:.2E}".format(t[-1] * beta ** (1 / 2) * lambda_ ** (-1 / 2) / si.get("pc")),
)


r_ = t * beta ** (1 / 2) * lambda_ ** (-1 / 2)
rho_ = dens * lambda_ * dens_fact


I = scipy.integrate.simpson(rho_ * r_ * r_ * 4 * np.pi, r_)
print("integral after normalisation / total mass", I / M)


# apply units
r_ = r_
rho_ = rho_

plt.figure()
plt.plot(r_, rho_, "-", label="rho(x) iso")

plt.xlabel("x [m]")
plt.ylabel("rho(x) [kg/m^3]")
plt.xscale("log")
plt.yscale("log")

# print(r_)

primitive = []
accum = 0
dx = np.diff(r_)
dx = np.concatenate([[dx[0]], dx])
for r_val, rho_val, dx_val in zip(r_, rho_, dx):
    r_val = r_val
    primitive.append(accum)
    # print(r_val, rho_val, accum, dx_val)
    accum += rho_val * dx_val

primitive = np.array(primitive)

plt.figure()
plt.plot(r_, primitive / sicte.sol_mass())
plt.xlabel("x [m]")
plt.ylabel("primitive(x) [sol mass]")
plt.grid()


# normalize f so that primitive[-1] = 1
norm = primitive[-1]
primitive = primitive / norm


# plot primitive
plt.figure()
plt.plot(r_, primitive)
plt.xlabel("x")
plt.ylabel("primitive(x)")
plt.title("primitive(x) = integral(f(x))")

# plot finv
plt.figure()
plt.plot(primitive, r_)
plt.xlabel("x")
plt.ylabel("finv(x)")
plt.title("finv(x) = inverse(primitive(x))")


# random set of points between 0 and 1
np.random.seed(111)
points = np.random.rand(100000)[:]

range_end = (0, r_[-1])
range_start = (0, 1)

# interpolate primitive using scipy.interpolate.interp1d
from scipy.interpolate import interp1d

mapping_interp = interp1d(primitive, r_, kind="linear")

points_mapped = [mapping_interp(point) for point in points]

# print(f"points = {points}")
# print(f"points_mapped = {points_mapped}")

plt.figure()
hist_r, bins_r = np.histogram(points, bins=101, density=True, range=range_start)
r = np.linspace(bins_r[0], bins_r[-1], 101)

plt.bar(bins_r[:-1], hist_r, np.diff(bins_r), alpha=0.5)
plt.xlabel("$r$")
plt.ylabel("$f(r)$")

plt.figure()
hist_r, bins_r = np.histogram(points_mapped, bins=101, density=True, range=range_end)
r = np.linspace(bins_r[0], bins_r[-1], 101)

plt.bar(bins_r[:-1], hist_r, np.diff(bins_r), alpha=0.5)
plt.plot(r_, rho_)
plt.xlabel("$r$")
plt.ylabel("$f(r)$")


shamrock.enable_experimental_features()
codeu = shamrock.UnitSystem(
    unit_time=sicte.year(),
    unit_length=sicte.au(),
    unit_mass=sicte.sol_mass(),
)
ucte = shamrock.Constants(codeu)


m_s_codeu = codeu.get("m") * codeu.get("s", power=-1)
kg_m3_codeu = codeu.get("kg") * codeu.get("m", power=-3)

print(f"m_s_codeu: {kg_m3_codeu}")
print(f"kg_m3_codeu: {kg_m3_codeu}")

cs = cs * m_s_codeu  # m/s
rho_c1 = 1.92e-13 * 1000 * kg_m3_codeu  # g/cm^3 -> kg/m^3
rho_c2 = 3.84e-8 * 1000 * kg_m3_codeu  # g/cm^3 -> kg/m^3
rho_c3 = 1.92e-3 * 1000 * kg_m3_codeu  # g/cm^3 -> kg/m^3

print(f"rho_c1: {rho_c1}")
print(f"rho_c2: {rho_c2}")
print(f"rho_c3: {rho_c3}")
print(f"cs: {cs}")


kb = ucte.kb()
print(f"kb: {kb}")
mu = 2.375
mh = 1.00784 * ucte.dalton()
print(f"mu * mh * kb: {mu * mh * kb}")

sphere_radius = r_[-1] * codeu.get("m")
sim_radius = sphere_radius * 1.5

Npart = 5e5 * (8 / (4 * np.pi / 3))  # because the injection is a cube truncated to a sphere

bmin = (-sim_radius, -sim_radius, -sim_radius)
bmax = (sim_radius, sim_radius, sim_radius)


scheduler_split_val = int(2e7)
scheduler_merge_val = int(1)


ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")

cfg = model.gen_default_config()
# cfg.set_artif_viscosity_Constant(alpha_u = 1, alpha_AV = 1, beta_AV = 2)
# cfg.set_artif_viscosity_VaryingMM97(alpha_min = 0.1,alpha_max = 1,sigma_decay = 0.1, alpha_u = 1, beta_AV = 2)
cfg.set_artif_viscosity_VaryingCD10(
    alpha_min=0.0, alpha_max=1, sigma_decay=0.1, alpha_u=1, beta_AV=2
)
cfg.set_boundary_periodic()
cfg.set_eos_machida06(rho_c1=rho_c1, rho_c2=rho_c2, rho_c3=rho_c3, cs=cs, mu=mu, mh=mh, kb=kb)

cfg.set_self_gravity_fmm(order=5, opening_angle=0.5)
cfg.set_softening_plummer(epsilon=1e-1)

# The important part to enable killing
cfg.add_kill_sphere(center=(0.0, 0.0, 0.0), radius=sphere_radius * 1.3)


cfg.set_units(codeu)
# cfg.print_status()
model.set_solver_config(cfg)
model.init_scheduler(scheduler_split_val, scheduler_merge_val)

model.resize_simulation_box(bmin, bmax)

setup = model.get_setup()


init_part_bmin = (-1, -1, -1)
init_part_bmax = (1, 1, 1)

xm, ym, zm = init_part_bmin
xM, yM, zM = init_part_bmax
vol_b = (xM - xm) * (yM - ym) * (zM - zm)

part_vol = vol_b / Npart

# lattice volume
part_vol_lattice = 0.74 * part_vol

dr = (part_vol_lattice / ((4.0 / 3.0) * 3.1416)) ** (1.0 / 3.0)

gen = setup.make_generator_lattice_hcp(dr, init_part_bmin, init_part_bmax)


def is_in_sphere(pt):
    x, y, z = pt
    return (x**2 + y**2 + z**2) < 1


thesphere = setup.make_modifier_filter(parent=gen, filter=is_in_sphere)

# On aurora /2 was correct to avoid out of memory
setup.apply_setup(thesphere, insert_step=int(scheduler_split_val / 2))

pmass = model.total_mass_to_part_mass(M * codeu.get("kg"))
print(f"pmass: {pmass}")
model.set_particle_mass(pmass)


def remap_func(r):

    x, y, z = r
    r = np.sqrt(x**2 + y**2 + z**2)

    if r == 0:
        x_norm = 0
        y_norm = 0
        z_norm = 0
    else:
        x_norm = x / r
        y_norm = y / r
        z_norm = z / r

    r_new = mapping_interp(r) * codeu.get("m")

    x = x_norm * r_new
    y = y_norm * r_new
    z = z_norm * r_new

    return (x, y, z)


model.remap_positions(remap_func)


def hpart_func(r):
    return dr


model.set_field_value_lambda_f64("hpart", hpart_func)


model.set_cfl_cour(0.1)
model.set_cfl_force(0.1)

model.set_cfl_multipler(1e-4)
model.set_cfl_mult_stiffness(10)

model.change_htolerances(coarse=1.3, fine=1.1)
model.timestep()
# 1.1**(1/4) means that the smoothing length will perform 4 subcycles per coarse cycle
model.change_htolerances(coarse=1.1, fine=1.1 ** (1 / 4))

# model.timestep()

### plot (r,rho)
data = ctx.collect_data()
r = np.sqrt(data["xyz"][:, 0] ** 2 + data["xyz"][:, 1] ** 2 + data["xyz"][:, 2] ** 2)
hpart = data["hpart"]
rho = pmass * (model.get_hfact() / hpart) ** 3

plt.figure()
plt.scatter(r * codeu.to("m"), rho)
plt.xlabel("r [m]")
plt.ylabel("rho [kg/m^3]")

plt.grid()

plt.figure()
plt.scatter(r * codeu.to("m"), hpart)
plt.xlabel("r [m]")
plt.ylabel("hpart")

plt.grid()

plt.show()

ext = sphere_radius * 2
arr_rho = model.render_cartesian_slice(
    "rho",
    "f64",
    center=(0.0, 0.0, 0.0),
    delta_x=(ext * 2, 0, 0.0),
    delta_y=(0.0, ext * 2, 0.0),
    nx=1000,
    ny=1000,
)

dpi = 200
plt.figure(num=1, clear=True, dpi=dpi)
import copy

my_cmap = copy.copy(matplotlib.colormaps.get_cmap("gist_heat"))  # copy the default cmap
my_cmap.set_bad(color="black")

res = plt.imshow(
    arr_rho,
    cmap=my_cmap,
    origin="lower",
    extent=[-ext, ext, -ext, ext],
    # norm="log",
    # vmin=1e0,
    # vmax=1e3,
)

plt.xlabel("x")
plt.ylabel("y")
plt.title("t = {:0.3f} [year]".format(model.get_time()))

plt.colorbar()
plt.show()


def analysis(i):
    ext = sphere_radius * 2

    arr_rho = model.render_cartesian_slice(
        "rho",
        "f64",
        center=(0.0, 0.0, 0.0),
        delta_x=(ext * 2, 0, 0.0),
        delta_y=(0.0, ext * 2, 0.0),
        nx=1000,
        ny=1000,
    )

    arr_rho2 = model.render_cartesian_column_integ(
        "rho",
        "f64",
        center=(0.0, 0.0, 0.0),
        delta_x=(ext * 2, 0, 0.0),
        delta_y=(0.0, ext * 2, 0.0),
        nx=1000,
        ny=1000,
    )

    arr_cs = model.render_cartesian_column_integ(
        "soundspeed",
        "f64",
        center=(0.0, 0.0, 0.0),
        delta_x=(ext * 2, 0, 0.0),
        delta_y=(0.0, ext * 2, 0.0),
        nx=1000,
        ny=1000,
    )

    arr_vxyz = model.render_cartesian_slice(
        "vxyz",
        "f64_3",
        center=(0.0, 0.0, 0.0),
        delta_x=(ext * 2, 0, 0.0),
        delta_y=(0.0, ext * 2, 0.0),
        nx=1000,
        ny=1000,
    )

    dpi = 200

    plt.figure(num=1, clear=True, dpi=dpi)
    import copy

    my_cmap = copy.copy(matplotlib.colormaps.get_cmap("gist_heat"))  # copy the default cmap
    my_cmap.set_bad(color="black")

    res = plt.imshow(
        arr_rho,
        cmap=my_cmap,
        origin="lower",
        extent=[-ext, ext, -ext, ext],
        # norm="log",
        # vmin=1e0,
        # vmax=1e3,
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("t = {:0.3f} [year]".format(model.get_time()))

    plt.colorbar()
    plt.savefig(f"collapse/rho_slice_{i:04}.png")
    plt.close()

    plt.figure(num=1, clear=True, dpi=dpi)
    import copy

    my_cmap = copy.copy(matplotlib.colormaps.get_cmap("gist_heat"))  # copy the default cmap
    my_cmap.set_bad(color="black")

    res = plt.imshow(
        arr_rho2,
        cmap=my_cmap,
        origin="lower",
        extent=[-ext, ext, -ext, ext],
        # norm="log",
        # vmin=1e0,
        # vmax=1e3,
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("t = {:0.3f} [year]".format(model.get_time()))

    plt.colorbar()
    plt.savefig(f"collapse/rho_{i:04}.png")
    plt.close()

    dpi = 200
    plt.figure(num=1, clear=True, dpi=dpi)
    import copy

    my_cmap = copy.copy(matplotlib.colormaps.get_cmap("gist_heat"))  # copy the default cmap
    my_cmap.set_bad(color="black")

    res = plt.imshow(
        arr_cs,
        cmap=my_cmap,
        origin="lower",
        extent=[-ext, ext, -ext, ext],
        # norm="log",
        # vmin=1e0,
        # vmax=1e3,
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("t = {:0.3f} [year]".format(model.get_time()))

    plt.colorbar()
    plt.savefig(f"collapse/cs_{i:04}.png")
    plt.close()

    dpi = 200
    plt.figure(num=1, clear=True, dpi=dpi)
    import copy

    my_cmap = copy.copy(matplotlib.colormaps.get_cmap("seismic"))  # copy the default cmap
    my_cmap.set_bad(color="black")

    res = plt.imshow(
        arr_vxyz[:, :, 0],
        cmap=my_cmap,
        origin="lower",
        extent=[-ext, ext, -ext, ext],
        # norm="log",
        # vmin=1e0,
        # vmax=1e3,
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("t = {:0.3f} [year]".format(model.get_time()))

    plt.colorbar()
    plt.savefig(f"collapse/vx_{i:04}.png")
    plt.close()

    dat = ctx.collect_data()

    print(f"total mass: {pmass * len(dat["xyz"] )}")

    xyz = dat["xyz"]
    vxyz = dat["vxyz"]
    rho = pmass * (model.get_hfact() / dat["hpart"]) ** 3
    soundspeed = dat["soundspeed"] * codeu.to("m") / codeu.to("s")
    r = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2 + xyz[:, 2] ** 2) * codeu.to("au")
    vr = xyz[:, 0] * vxyz[:, 0] + xyz[:, 1] * vxyz[:, 1] + xyz[:, 2] * vxyz[:, 2]
    vr = vr * codeu.to("m") / codeu.to("s")
    rho = rho * codeu.to("kg") * codeu.to("m", power=-3)

    cs_min_plot = 1e2
    cs_max_plot = 1e4

    rho_min_plot = 1e-17
    rho_max_plot = 1e-5

    r_min_plot = 1e-2
    r_max_plot = sphere_radius * 1.4

    fig, axs = plt.subplots(2, 2, figsize=(12, 6), dpi=dpi)
    axs[0, 0].scatter(r, rho, s=2)
    axs[0, 0].set_xlabel("r [au]")
    axs[0, 0].set_ylabel("rho [kg/m^3]")
    axs[0, 0].set_xlim(r_min_plot, r_max_plot)
    axs[0, 0].set_ylim(rho_min_plot, rho_max_plot)
    axs[0, 0].set_xscale("log")
    axs[0, 0].set_yscale("log")
    axs[0, 0].grid()

    axs[0, 1].scatter(r, soundspeed, s=2)
    axs[0, 1].set_xlabel("r [au]")
    axs[0, 1].set_ylabel("soundspeed [m/s]")
    axs[0, 1].set_xlim(r_min_plot, r_max_plot)
    axs[0, 1].set_ylim(cs_min_plot, cs_max_plot)
    axs[0, 1].set_xscale("log")
    axs[0, 1].set_yscale("log")
    axs[0, 1].grid()

    axs[1, 0].scatter(r, vr, s=2)
    axs[1, 0].set_xlabel("r [au]")
    axs[1, 0].set_ylabel("vr [m/s]")
    axs[1, 0].set_xlim(r_min_plot, r_max_plot)
    axs[1, 0].set_ylim(-1e5, 1e5)
    axs[1, 0].set_xscale("log")
    axs[1, 0].set_yscale("linear")
    axs[1, 0].grid()

    rho_conv = codeu.to("kg") * codeu.to("m", power=-3)
    cs_conv = codeu.to("m") / codeu.to("s")

    rho_plot = np.logspace(-15, 1, 1000)
    P_plot = []
    cs_plot = []
    for rho_ in rho_plot:
        P, _cs, T = shamrock.phys.eos.eos_Machida06(
            cs=cs * cs_conv,
            rho=rho_,
            rho_c1=rho_c1 * rho_conv,
            rho_c2=rho_c2 * rho_conv,
            rho_c3=rho_c3 * rho_conv,
            mu=mu,
            mh=mh,
            kb=kb,
        )
        P_plot.append(P)
        cs_plot.append(_cs)

    axs[1, 1].plot(rho_plot, cs_plot, label="P", color="grey", alpha=0.2)

    axs[1, 1].scatter(rho, soundspeed, s=2)
    axs[1, 1].set_xlabel("rho [kg/m^3]")
    axs[1, 1].set_ylabel("soundspeed [m/s]")
    axs[1, 1].set_xlim(rho_min_plot, rho_max_plot)
    axs[1, 1].set_ylim(cs_min_plot, cs_max_plot)
    axs[1, 1].set_xscale("log")
    axs[1, 1].set_yscale("log")
    axs[1, 1].grid()

    fig.suptitle("t = {:0.3f} [year]".format(model.get_time()))
    fig.tight_layout()
    fig.savefig(f"collapse/rho_soundspeed_{i:04}.png")
    plt.close()

    return np.max(rho)


i = 0
t = 0

steps = 10
while True:
    rho_max = analysis(i)

    for j in range(steps):
        model.timestep()

    i += 1
    t += 2.0e-3

    if i > 1:
        analysis(i)
        break
