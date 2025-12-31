"""
Stretch mapping
==========================================

This simple example shows how to set up a sphere with a given density profile following the stretch mapping procedure.
These are three examples to explain how it works. The first two are illustrative; the third shows the desired result."
"""

# %%
# First some generic stuff to initialize shamrock and common properties of our setup.
import shamrock

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

import numpy as np

figsize = (4, 7)

# HCP Lattice properties
N_target = 200
bsize = 1
bmin = (-bsize, -bsize, -bsize)
bmax = (bsize, bsize, bsize)
xm, ym, zm = bmin
xM, yM, zM = bmax
vol_b = (xM - xm) * (yM - ym) * (zM - zm)
part_vol = vol_b / N_target
HCP_PACKING_DENSITY = 0.74
part_vol_lattice = HCP_PACKING_DENSITY * part_vol
dr = (part_vol_lattice / ((4.0 / 3.0) * np.pi)) ** (1.0 / 3.0)


def init_model():
    global bmin, bmax
    ctx = shamrock.Context()
    ctx.pdata_layout_new()
    model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")
    cfg = model.gen_default_config()
    cfg.set_boundary_periodic()
    n = 1
    gamma = 1 + 1 / n
    K = 1
    cfg.set_eos_polytropic(
        K, gamma
    )  # In this case, the density profile at hydrostatic equilibrium should be a sinc
    model.set_solver_config(cfg)
    model.init_scheduler(int(1e8), 1)

    bmin, bmax = model.get_ideal_hcp_box(dr, bmin, bmax)
    model.resize_simulation_box(bmin, bmax)

    return model, ctx


def compute_rho(model, data):
    return model.get_particle_mass() * (model.get_hfact() / data["hpart"]) ** 3


# Plot the result
import matplotlib.pyplot as plt

# fig, axs = plt.subplot_mosaic(
#     [
#         ["hcp", "hcp_stretched", "hcp_stretched_filtered"],
#         ["hcp_profile", "hcp_stretched_profile", "hcp_stretched_filtered_profile"],
#     ],
#     per_subplot_kw={
#         "hcp": {"projection": "3d"},
#         "hcp_stretched": {"projection": "3d"},
#         "hcp_stretched_filtered": {"projection": "3d"},
#     },
# )


def plot_lattice(ax, model, ctx):
    data = ctx.collect_data()
    pos = data["xyz"]
    rho = compute_rho(model, data)
    ax.set_xlim(bmin[0], bmax[0])
    ax.set_ylim(bmin[1], bmax[1])
    ax.set_zlim(bmin[2], bmax[2])
    ax.set_aspect("equal")
    X = pos[:, 0]
    Y = pos[:, 1]
    Z = pos[:, 2]
    sc = ax.scatter(X, Y, Z, c=rho, s=15, vmin=0, vmax=1)
    fig.colorbar(sc, ax=ax, label=r"$\rho$")


def plot_profile(ax, model, ctx):
    data = ctx.collect_data()
    r = np.linalg.norm(data["xyz"], axis=1)
    rho = compute_rho(model, data)
    ax.scatter(r, rho)
    ax.set_xlim(0, 2.25)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$\rho$")


# %%
# We start with an HCP lattice, which should have uniform density (with periodic boundary conditions)
totmass = 1
model_hcp, ctx_hcp = init_model()
setup = model_hcp.get_setup()
hcp = setup.make_generator_lattice_hcp(dr, bmin, bmax)
setup.apply_setup(hcp)
pmass = model_hcp.total_mass_to_part_mass(totmass)
model_hcp.set_particle_mass(pmass)
model_hcp.evolve_once_override_time(0.0, 0.0)


fig = plt.figure(figsize=figsize)
fig.subplots_adjust(left=0.15, right=0.9)
ax0 = fig.add_subplot(211, projection="3d")
ax1 = fig.add_subplot(
    212,
)

rmax = bsize
tabx = np.linspace(0, rmax)
tabrho = np.sinc(
    tabx / (rmax)
)  # the profile that we want! (without the normalization factor)

if shamrock.sys.world_rank() == 0:
    plot_lattice(ax0, model_hcp, ctx_hcp)
    plot_profile(ax1, model_hcp, ctx_hcp)
    norm_factor = totmass / np.trapezoid(
        tabrho * 4 * np.pi * tabx**2, tabx
    )  # this would be the normalization factor
    ax1.plot(tabx, norm_factor * tabrho, label="target (sinc)")
    ax1.legend()


# %%
# We indeed got a uniform density (thanks to periodic boundary conditions). Now let's try to do the same thing but by stretching it afterwards.
model_hcp_smap, ctx_hcp_smap = init_model()

rmax = bsize
tabx = np.linspace(0, rmax)
tabrho = np.sinc(
    tabx / (rmax)
)  # This is for example the hydrostatic density profile for polytropic EoS (n=1). It doesn't have to be normalized
# Note that the stretch mapping doesn't need the particles' mass to work
totmass = 1  # The stretch mapping modifier will automatically set the particle mass given the total mass desired.
setup = model_hcp_smap.get_setup()
hcp = setup.make_generator_lattice_hcp(dr, bmin, bmax)
stretched_hcp = setup.make_modifier_stretch_mapping(
    parent=hcp,
    system="spherical",
    axis="r",
    box_min=bmin,
    box_max=bmax,
    tabx=tabx,
    tabrho=tabrho,
    mtot=totmass,
)
setup.apply_setup(stretched_hcp)
model_hcp_smap.evolve_once_override_time(0.0, 0.0)

fig = plt.figure(figsize=figsize)
fig.subplots_adjust(left=0.15, right=0.9)
ax0 = fig.add_subplot(211, projection="3d")
ax1 = fig.add_subplot(
    212,
)
if shamrock.sys.world_rank() == 0:
    plot_lattice(ax0, model_hcp_smap, ctx_hcp_smap)
    plot_profile(ax1, model_hcp_smap, ctx_hcp_smap)
    norm_factor = totmass / np.trapezoid(
        tabrho * 4 * np.pi * tabx**2, tabx
    )  # this would be the normalization factor
    ax1.plot(tabx, norm_factor * tabrho, label="target (sinc)")
    ax1.legend()


# %%
# .. note::
# Notice that here, the particles that are outside the sphere are not stretched. We are almost here: to finally get a sphere, we can just crop it using a filter_modifier
def is_in_sphere(pt):
    x, y, z = pt
    return (x**2 + y**2 + z**2) < 1


model_hcp_smap_sphere, ctx_hcp_smap_sphere = init_model()
setup = model_hcp_smap_sphere.get_setup()
hcp = setup.make_generator_lattice_hcp(dr, bmin, bmax)
stretched_hcp = setup.make_modifier_stretch_mapping(
    parent=hcp,
    system="spherical",
    axis="r",
    box_min=bmin,
    box_max=bmax,
    tabx=tabx,
    tabrho=tabrho,
    mtot=totmass,
)
cropped_stretched_hcp = setup.make_modifier_filter(
    parent=stretched_hcp, filter=is_in_sphere
)
setup.apply_setup(cropped_stretched_hcp)
model_hcp_smap_sphere.evolve_once_override_time(0.0, 0.0)

fig = plt.figure(figsize=figsize)
fig.subplots_adjust(left=0.15, right=0.9)
ax0 = fig.add_subplot(211, projection="3d")
ax1 = fig.add_subplot(
    212,
)
if shamrock.sys.world_rank() == 0:
    plot_lattice(ax0, model_hcp_smap_sphere, ctx_hcp_smap_sphere)
    plot_profile(
        ax1,
        model_hcp_smap_sphere,
        ctx_hcp_smap_sphere,
    )
    norm_factor = totmass / np.trapezoid(
        tabrho * 4 * np.pi * tabx**2, tabx
    )  # this would be the normalization factor
    ax1.plot(tabx, norm_factor * tabrho, label="target (sinc)")
    ax1.legend()

# %%
# This is the desired result - our star is ready to reach hydrostatic equilibrium!
# You can also use it for different geometries (probably).
