# sphinx_gallery_multi_image = "single"

import shamrock

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")


# %%
# Setup parameters
import numpy as np

kernel = "M4"  # SPH kernel to use
Npart = 100000
disc_mass = 0.01  # sol mass
center_mass = 1
center_racc = 0.1

rout = 10
rin = 1

# alpha_ss ~ alpha_AV * 0.08
alpha_AV = 1e-3 / 0.08
alpha_u = 1
beta_AV = 2

q = 0.5
p = 3.0 / 2.0
r0 = 1

C_cour = 0.3
C_force = 0.25

H_r_in = 0.05

dump_folder = "_to_trash"
sim_name = "disc_sph"


import os

# Create the dump directory if it does not exist
if shamrock.sys.world_rank() == 0:
    os.makedirs(dump_folder, exist_ok=True)

# %%
# Deduced quantities


si = shamrock.UnitSystem()
sicte = shamrock.Constants(si)
codeu = shamrock.UnitSystem(
    unit_time=3600 * 24 * 365,
    unit_length=sicte.au(),
    unit_mass=sicte.sol_mass(),
)
ucte = shamrock.Constants(codeu)

# Deduced quantities
pmass = disc_mass / Npart
bmin = (-rout * 2, -rout * 2, -rout * 2)
bmax = (rout * 2, rout * 2, rout * 2)
G = ucte.G()


def sigma_profile(r):
    sigma_0 = 1
    return sigma_0 * (r / rin) ** (-p)


def kep_profile(r):
    return (G * center_mass / r) ** 0.5


def omega_k(r):
    return kep_profile(r) / r


def cs_profile(r):
    cs_in = (H_r_in * rin) * omega_k(rin)
    return ((r / rin) ** (-q)) * cs_in


cs0 = cs_profile(rin)


def rot_profile(r):
    # return kep_profile(r)

    # subkeplerian correction
    return ((kep_profile(r) ** 2) - (2 * p + q) * cs_profile(r) ** 2) ** 0.5


def H_profile(r):
    H = cs_profile(r) / omega_k(r)
    # fact = (2.**0.5) * 3.
    fact = 1
    return fact * H  # factor taken from phantom, to fasten thermalizing


# %%
# Configure the solver
ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel=kernel)

cfg = model.gen_default_config()
cfg.set_artif_viscosity_ConstantDisc(alpha_u=alpha_u, alpha_AV=alpha_AV, beta_AV=beta_AV)
cfg.set_eos_locally_isothermalLP07(cs0=cs0, q=q, r0=r0)
cfg.print_status()
cfg.set_units(codeu)
model.set_solver_config(cfg)

# Set scheduler criteria to effectively disable patch splitting and merging.
crit_split = int(1e9)
crit_merge = 1
model.init_scheduler(crit_split, crit_merge)
model.resize_simulation_box(bmin, bmax)

# %%
# Setup the sink particles

sink_list = [
    {"mass": center_mass, "racc": center_racc, "pos": (0, 0, 0), "vel": (0, 0, 0)},
]

model.set_particle_mass(pmass)
for s in sink_list:
    mass = s["mass"]
    x, y, z = s["pos"]
    vx, vy, vz = s["vel"]
    racc = s["racc"]

    print("add sink : mass {} pos {} vel {} racc {}".format(mass, (x, y, z), (vx, vy, vz), racc))

    model.add_sink(mass, (x, y, z), (vx, vy, vz), racc)

# %%
# Setup the simulation

setup = model.get_setup()
gen_disc = setup.make_generator_disc_mc(
    part_mass=pmass,
    disc_mass=disc_mass,
    r_in=rin,
    r_out=rout,
    sigma_profile=sigma_profile,
    H_profile=H_profile,
    rot_profile=rot_profile,
    cs_profile=cs_profile,
    random_seed=666,
)
# print(comb.get_dot())
setup.apply_setup(gen_disc)

model.set_cfl_cour(C_cour)
model.set_cfl_force(C_force)

model.change_htolerance(1.3)
model.timestep()
model.change_htolerance(1.1)


# %%
# Plotting functions
import copy

import matplotlib
import matplotlib.pyplot as plt


def save_plot_state(iplot):

    pixel_x = 1080
    pixel_y = 1080
    radius = rout * 1.5
    center = (0.0, 0.0, 0.0)
    aspect = pixel_x / pixel_y
    pic_range = [-radius * aspect, radius * aspect, -radius, radius]
    delta_x = (radius * 2 * aspect, 0.0, 0.0)
    delta_y = (0.0, radius * 2, 0.0)
    delta_z = (0.0, 0.0, radius * 2)

    def _render_xy(field, field_type, center):
        # Helper to reduce code duplication
        return model.render_cartesian_slice(
            field,
            field_type,
            center=center,
            delta_x=delta_x,
            delta_y=delta_y,
            nx=pixel_x,
            ny=pixel_y,
        )

    def _render_xz(field, field_type, center):
        # Helper to reduce code duplication
        return model.render_cartesian_slice(
            field,
            field_type,
            center=center,
            delta_x=delta_x,
            delta_y=delta_z,
            nx=pixel_x,
            ny=pixel_y,
        )

    arr_rho_xy = _render_xy("rho", "f64", center)
    arr_vel_xy = _render_xy("vxyz", "f64_3", center)
    arr_rho_xz = _render_xz("rho", "f64", center)
    arr_vel_xz = _render_xz("vxyz", "f64_3", center)

    np.save(os.path.join(dump_folder, f"{sim_name}_rho_xy_{iplot:04}.npy"), arr_rho_xy)
    np.save(os.path.join(dump_folder, f"{sim_name}_vxyz_xy_{iplot:04}.npy"), arr_vel_xy)
    np.save(os.path.join(dump_folder, f"{sim_name}_rho_xz_{iplot:04}.npy"), arr_rho_xz)
    np.save(os.path.join(dump_folder, f"{sim_name}_vxyz_xz_{iplot:04}.npy"), arr_vel_xz)


# %%
# Running the simulation

t_sum = 0
t_target = 0.1

save_plot_state(0)

i_dump = 1
dt_dump = 0.05

while t_sum < t_target:
    model.evolve_until(t_sum + dt_dump)

    save_plot_state(i_dump)

    t_sum += dt_dump
    i_dump += 1

reference_folder = "reference-files/regression_sph_disc"

for iplot in range(i_dump):
    rho_ref_xy = np.load(os.path.join(reference_folder, f"disc_sph_rho_xy_{iplot:04}.npy"))
    vxyz_ref_xy = np.load(os.path.join(reference_folder, f"disc_sph_vxyz_xy_{iplot:04}.npy"))
    rho_ref_xz = np.load(os.path.join(reference_folder, f"disc_sph_rho_xz_{iplot:04}.npy"))
    vxyz_ref_xz = np.load(os.path.join(reference_folder, f"disc_sph_vxyz_xz_{iplot:04}.npy"))

    rho_sham_xy = np.load(os.path.join(dump_folder, f"disc_sph_rho_xy_{iplot:04}.npy"))
    vxyz_sham_xy = np.load(os.path.join(dump_folder, f"disc_sph_vxyz_xy_{iplot:04}.npy"))
    rho_sham_xz = np.load(os.path.join(dump_folder, f"disc_sph_rho_xz_{iplot:04}.npy"))
    vxyz_sham_xz = np.load(os.path.join(dump_folder, f"disc_sph_vxyz_xz_{iplot:04}.npy"))

    diff_rho_xy = rho_ref_xy - rho_sham_xy
    diff_vxyz_xy = vxyz_ref_xy - vxyz_sham_xy
    diff_rho_xz = rho_ref_xz - rho_sham_xz
    diff_vxyz_xz = vxyz_ref_xz - vxyz_sham_xz

    #########################################################
    # Plotting parameters
    #########################################################

    pixel_x = 1080
    pixel_y = 1080
    radius = rout * 1.5
    center = (0.0, 0.0, 0.0)
    aspect = pixel_x / pixel_y
    pic_range = [-radius * aspect, radius * aspect, -radius, radius]
    delta_x = (radius * 2 * aspect, 0.0, 0.0)
    delta_y = (0.0, radius * 2, 0.0)
    delta_z = (0.0, 0.0, radius * 2)

    #########################################################
    # Colormaps
    #########################################################

    my_cmap = copy.copy(matplotlib.colormaps.get_cmap("gist_heat"))
    my_cmap.set_bad(color="black")

    my_cmap2 = copy.copy(matplotlib.colormaps.get_cmap("nipy_spectral"))
    my_cmap2.set_bad(color="black")

    #########################################################
    # Plot the current sim results
    #########################################################

    def plot(field, label, xlabel, ylabel, savefig):

        fig = plt.figure(dpi=200)
        im0 = plt.imshow(field, cmap=my_cmap, origin="lower", extent=pic_range)

        cbar0 = plt.colorbar(im0, extend="both")
        cbar0.set_label(label)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(savefig)

        plt.close(fig)

    plot(
        rho_sham_xy,
        r"$\rho$ [code unit]",
        "x",
        "y",
        os.path.join(dump_folder, f"{sim_name}_rho_xy_{iplot:04}.png"),
    )
    plot(
        rho_sham_xz,
        r"$\rho$ [code unit]",
        "x",
        "z",
        os.path.join(dump_folder, f"{sim_name}_rho_xz_{iplot:04}.png"),
    )
    plot(
        vxyz_sham_xy[:, :, 0],
        r"$v_x$ [code unit]",
        "x",
        "y",
        os.path.join(dump_folder, f"{sim_name}_vx_xy_{iplot:04}.png"),
    )
    plot(
        vxyz_sham_xz[:, :, 0],
        r"$v_x$ [code unit]",
        "x",
        "z",
        os.path.join(dump_folder, f"{sim_name}_vx_xz_{iplot:04}.png"),
    )
    plot(
        vxyz_sham_xy[:, :, 1],
        r"$v_y$ [code unit]",
        "x",
        "y",
        os.path.join(dump_folder, f"{sim_name}_vy_xy_{iplot:04}.png"),
    )
    plot(
        vxyz_sham_xz[:, :, 1],
        r"$v_y$ [code unit]",
        "x",
        "z",
        os.path.join(dump_folder, f"{sim_name}_vy_xz_{iplot:04}.png"),
    )
    plot(
        vxyz_sham_xy[:, :, 2],
        r"$v_z$ [code unit]",
        "x",
        "y",
        os.path.join(dump_folder, f"{sim_name}_vz_xy_{iplot:04}.png"),
    )
    plot(
        vxyz_sham_xz[:, :, 2],
        r"$v_z$ [code unit]",
        "x",
        "z",
        os.path.join(dump_folder, f"{sim_name}_vz_xz_{iplot:04}.png"),
    )

    plot(
        diff_rho_xy,
        r"$\rho$ [code unit]",
        "x",
        "y",
        os.path.join(dump_folder, f"{sim_name}_diff_rho_xy_{iplot:04}.png"),
    )
    plot(
        diff_rho_xz,
        r"$\rho$ [code unit]",
        "x",
        "z",
        os.path.join(dump_folder, f"{sim_name}_diff_rho_xz_{iplot:04}.png"),
    )
    plot(
        diff_vxyz_xy[:, :, 0],
        r"$v_x$ [code unit]",
        "x",
        "y",
        os.path.join(dump_folder, f"{sim_name}_diff_vx_xy_{iplot:04}.png"),
    )
    plot(
        diff_vxyz_xz[:, :, 0],
        r"$v_x$ [code unit]",
        "x",
        "z",
        os.path.join(dump_folder, f"{sim_name}_diff_vx_xz_{iplot:04}.png"),
    )
    plot(
        diff_vxyz_xy[:, :, 1],
        r"$v_y$ [code unit]",
        "x",
        "y",
        os.path.join(dump_folder, f"{sim_name}_diff_vy_xy_{iplot:04}.png"),
    )
    plot(
        diff_vxyz_xz[:, :, 1],
        r"$v_y$ [code unit]",
        "x",
        "z",
        os.path.join(dump_folder, f"{sim_name}_diff_vy_xz_{iplot:04}.png"),
    )
    plot(
        diff_vxyz_xy[:, :, 2],
        r"$v_z$ [code unit]",
        "x",
        "y",
        os.path.join(dump_folder, f"{sim_name}_diff_vz_xy_{iplot:04}.png"),
    )
    plot(
        diff_vxyz_xz[:, :, 2],
        r"$v_z$ [code unit]",
        "x",
        "z",
        os.path.join(dump_folder, f"{sim_name}_diff_vz_xz_{iplot:04}.png"),
    )

    #########################################################
    # Compute max error
    #########################################################

    max_error_rho_xy = np.abs(diff_rho_xy).max()
    max_error_rho_xz = np.abs(diff_rho_xz).max()
    max_error_vx_xy = np.abs(diff_vxyz_xy[:, :, 0]).max()
    max_error_vx_xz = np.abs(diff_vxyz_xz[:, :, 0]).max()
    max_error_vy_xy = np.abs(diff_vxyz_xy[:, :, 1]).max()
    max_error_vy_xz = np.abs(diff_vxyz_xz[:, :, 1]).max()
    max_error_vz_xy = np.abs(diff_vxyz_xy[:, :, 2]).max()
    max_error_vz_xz = np.abs(diff_vxyz_xz[:, :, 2]).max()

    print(f"Max error rho_xy: {max_error_rho_xy}")
    print(f"Max error vx_xy: {max_error_vx_xy}")
    print(f"Max error vy_xy: {max_error_vy_xy}")
    print(f"Max error vz_xy: {max_error_vz_xy}")
    print(f"Max error rho_xz: {max_error_rho_xz}")
    print(f"Max error vx_xz: {max_error_vx_xz}")
    print(f"Max error vy_xz: {max_error_vy_xz}")
    print(f"Max error vz_xz: {max_error_vz_xz}")

    if (
        max_error_rho_xy > 1e-20
        or max_error_vx_xy > 1e-20
        or max_error_vy_xy > 1e-20
        or max_error_vz_xy > 1e-20
        or max_error_rho_xz > 1e-20
        or max_error_vx_xz > 1e-20
        or max_error_vy_xz > 1e-20
        or max_error_vz_xz > 1e-20
    ):
        print("Error is too high")
        exit(1)
