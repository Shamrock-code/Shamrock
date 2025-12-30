"""
Stretch mapping
==========================================

This simple example shows how to setup a sphere with the stretch mapping procedure
"""

# %%
# Init shamrock
import shamrock
import numpy as np

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")


# N_target = 200
# bsize = 1

# bmin = (-bsize, -bsize, -bsize)
# bmax = (bsize, bsize, bsize)
# xm, ym, zm = bmin
# xM, yM, zM = bmax
# vol_b = (xM - xm) * (yM - ym) * (zM - zm)
# part_vol = vol_b / N_target
# # lattice volume
# HCP_PACKING_DENSITY = 0.74
# part_vol_lattice = HCP_PACKING_DENSITY * part_vol

# dr = (part_vol_lattice / ((4.0 / 3.0) * np.pi)) ** (1.0 / 3.0)


# def init_model():
#     global bmin, bmax
#     ctx = shamrock.Context()
#     ctx.pdata_layout_new()
#     model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")
#     cfg = model.gen_default_config()
#     cfg.set_boundary_periodic()
#     model.set_solver_config(cfg)
#     model.init_scheduler(int(1e8), 1)

#     bmin, bmax = model.get_ideal_hcp_box(dr, bmin, bmax)
#     model.resize_simulation_box(bmin, bmax)

#     return model, ctx


# def compute_rho(model, data):
#     return model.get_particle_mass() * (model.get_hfact() / data["hpart"]) ** 3


# # Plot the result
# import matplotlib.pyplot as plt

# # fig, axs = plt.subplot_mosaic(
# #     [
# #         ["hcp", "hcp_stretched", "hcp_stretched_filtered"],
# #         ["hcp_profile", "hcp_stretched_profile", "hcp_stretched_filtered_profile"],
# #     ],
# #     per_subplot_kw={
# #         "hcp": {"projection": "3d"},
# #         "hcp_stretched": {"projection": "3d"},
# #         "hcp_stretched_filtered": {"projection": "3d"},
# #     },
# # )


# def plot_lattice(ax, model, ctx):
#     if shamrock.sys.world_rank() == 0:
#         data = ctx.collect_data()
#         pos = data["xyz"]
#         rho = compute_rho(model, data)
#         ax.set_xlim(bmin[0], bmax[0])
#         ax.set_ylim(bmin[1], bmax[1])
#         ax.set_zlim(bmin[2], bmax[2])
#         ax.set_aspect("equal")
#         X = pos[:, 0]
#         Y = pos[:, 1]
#         Z = pos[:, 2]
#         sc = ax.scatter(X, Y, Z, c=rho, s=15, vmin=0, vmax=1)
#     # fig.colorbar(sc, ax=ax, label=r"$\rho$")


# def plot_profile(ax, model, ctx):
#     data = ctx.collect_data()
#     r = np.linalg.norm(data["xyz"], axis=1)
#     rho = compute_rho(model, data)
#     ax.scatter(r, rho)
#     ax.set_xlim(0, 2.25)
#     ax.set_ylim(0, 1)
#     ax.set_xlabel(r"$r$")
#     ax.set_ylabel(r"$\rho$")


# # %%
# # Start with a HCP lattice
# totmass = 1
# model_hcp, ctx_hcp = init_model()
# setup = model_hcp.get_setup()
# hcp = setup.make_generator_lattice_hcp(dr, bmin, bmax)
# setup.apply_setup(hcp)
# pmass = model_hcp.total_mass_to_part_mass(totmass)
# model_hcp.set_particle_mass(pmass)
# model_hcp.evolve_once_override_time(0.0, 0.0)


# # if shamrock.sys.world_rank() == 0:
# fig = plt.figure(figsize=(3, 6))
# ax0 = fig.add_subplot(111, projection="3d")
# ax1 = fig.add_subplot(
#     212,
# )
# plot_lattice(ax0, model_hcp, ctx_hcp)
# plot_profile(ax1, model_hcp, ctx_hcp)


# # %%
# # Now let's do a second setup but with a stretched HCP lattice
# model_hcp_smap, ctx_hcp_smap = init_model()

# rmax = bsize
# tabx = np.linspace(0, rmax)
# tabrho = np.sinc(
#     tabx / (rmax)
# )  # This is for example the hydrostatic density profile for polytropic EoS (n=1). It doesn't have to be normalized
# # Note that the stretch mapping doesn't need the particles' mass to work
# totmass = 1  # The strecth mapping modifier will automatically set the particle mass given the total mass wanted.
# setup = model_hcp_smap.get_setup()
# hcp = setup.make_generator_lattice_hcp(dr, bmin, bmax)
# stretched_hcp = setup.make_modifier_stretch_mapping(
#     parent=hcp,
#     system="spherical",
#     axis="r",
#     box_min=bmin,
#     box_max=bmax,
#     tabx=tabx,
#     tabrho=tabrho,
#     mtot=totmass,
# )
# setup.apply_setup(stretched_hcp)
# model_hcp_smap.evolve_once_override_time(0.0, 0.0)


# if shamrock.sys.world_rank() == 0:
#     plot_lattice(axs["hcp_stretched"], model_hcp_smap, ctx_hcp_smap)
#     plot_profile(axs["hcp_stretched_profile"], model_hcp_smap, ctx_hcp_smap)
#     norm_factor = totmass / np.trapezoid(
#         tabrho * 4 * np.pi * tabx**2, tabx
#     )  # this would be the normalization factor
#     axs["hcp_stretched_profile"].plot(tabx, norm_factor * tabrho)


# # %%
# # .. note::
# # Notice that here, the particles that are outside the sphere are not stretched.
# # To finally get a sphere, we can just crop it
# def is_in_sphere(pt):
#     x, y, z = pt
#     return (x**2 + y**2 + z**2) < 1


# model_hcp_smap_sphere, ctx_hcp_smap_sphere = init_model()
# setup = model_hcp_smap_sphere.get_setup()
# hcp = setup.make_generator_lattice_hcp(dr, bmin, bmax)
# stretched_hcp = setup.make_modifier_stretch_mapping(
#     parent=hcp,
#     system="spherical",
#     axis="r",
#     box_min=bmin,
#     box_max=bmax,
#     tabx=tabx,
#     tabrho=tabrho,
#     mtot=totmass,
# )
# cropped_stretched_hcp = setup.make_modifier_filter(
#     parent=stretched_hcp, filter=is_in_sphere
# )
# setup.apply_setup(cropped_stretched_hcp)
# model_hcp_smap_sphere.evolve_once_override_time(0.0, 0.0)


# if shamrock.sys.world_rank() == 0:
#     plot_lattice(
#         axs["hcp_stretched_filtered"], model_hcp_smap_sphere, ctx_hcp_smap_sphere
#     )
#     plot_profile(
#         axs["hcp_stretched_filtered_profile"],
#         model_hcp_smap_sphere,
#         ctx_hcp_smap_sphere,
#     )
#     norm_factor = totmass / np.trapezoid(
#         tabrho * 4 * np.pi * tabx**2, tabx
#     )  # this would be the normalization factor
#     axs["hcp_stretched_filtered_profile"].plot(tabx, norm_factor * tabrho)


# plt.show()
