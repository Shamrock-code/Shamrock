"""
Production run: Circular disc & central potential
=================================================

This simple example shows how to run a circular disc simulation in SPH with a central potential
"""

# sphinx_gallery_multi_image = "single"

import glob
import json
import os  # for makedirs

import numpy as np

import shamrock

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

# %%
# Setup units

si = shamrock.UnitSystem()
sicte = shamrock.Constants(si)
codeu = shamrock.UnitSystem(
    unit_time=sicte.year(),
    unit_length=sicte.au(),
    unit_mass=sicte.sol_mass(),
)
ucte = shamrock.Constants(codeu)
G = ucte.G()

# %%
# List parameters

# Resolution
Npart = 100000

# Domain decomposition parameters
scheduler_split_val = int(1.0e7)  # split patches with more than 1e7 particles
scheduler_merge_val = scheduler_split_val // 16

# Dump and plot frequency and duration of the simulation
dump_freq_stop = 2
plot_freq_stop = 1

dt_stop = 0.01
nstop = 100

# The list of times at which the simulation will pause for analysis / dumping
t_stop = [i * dt_stop for i in range(nstop + 1)]


# Sink parameters
center_mass = 1.0
center_racc = 0.1

# Disc parameter
disc_mass = 0.01  # sol mass
rout = 10.0  # au
rin = 1.0  # au
H_r_0 = 0.05
q = 0.5
p = 3.0 / 2.0
r0 = 1.0

# Viscosity parameter
alpha_AV = 1.0e-3 / 0.08
alpha_u = 1.0
beta_AV = 2.0

# Integrator parameters
C_cour = 0.3
C_force = 0.25


dump_folder = "_to_trash/circular_disc_central_pot_" + str(Npart) + "/"
dump_prefix = dump_folder + "dump_"


# Disc profiles
def sigma_profile(r):
    sigma_0 = 1.0  # We do not care as it will be renormalized
    return sigma_0 * (r / r0) ** (-p)


def kep_profile(r):
    return (G * center_mass / r) ** 0.5


def omega_k(r):
    return kep_profile(r) / r


def cs_profile(r):
    cs_in = (H_r_0 * r0) * omega_k(r0)
    return ((r / r0) ** (-q)) * cs_in


# %%
# Create the dump directory if it does not exist
if shamrock.sys.world_rank() == 0:
    os.makedirs(dump_folder, exist_ok=True)

# %%
# Utility functions and quantities deduced from the base one

# Deduced quantities
pmass = disc_mass / Npart

bsize = rout * 2
bmin = (-bsize, -bsize, -bsize)
bmax = (bsize, bsize, bsize)

cs0 = cs_profile(r0)


def rot_profile(r):
    return ((kep_profile(r) ** 2) - (2 * p + q) * cs_profile(r) ** 2) ** 0.5


def H_profile(r):
    H = cs_profile(r) / omega_k(r)
    # fact = (2.**0.5) * 3. # factor taken from phantom, to fasten thermalizing
    fact = 1.0
    return fact * H


# %%
# Start the context
# The context holds the data of the code
# We then init the layout of the field (e.g. the list of fields used by the solver)

ctx = shamrock.Context()
ctx.pdata_layout_new()

# %%
# Attach a SPH model to the context

model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")


# %%
# Dump handling
def get_dump_name(idump):
    return dump_prefix + f"{idump:07}" + ".sham"


def get_vtk_dump_name(idump):
    return dump_prefix + f"{idump:07}" + ".vtk"


def get_ph_dump_name(idump):
    return dump_prefix + f"{idump:07}" + ".phdump"


def get_last_dump():
    res = glob.glob(dump_prefix + "*.sham")

    f_max = ""
    num_max = -1

    for f in res:
        try:
            dump_num = int(f[len(dump_prefix) : -5])
            if dump_num > num_max:
                f_max = f
                num_max = dump_num
        except:
            pass

    if num_max == -1:
        return None
    else:
        return num_max


def purge_old_dumps():
    if shamrock.sys.world_rank() == 0:
        res = glob.glob(dump_prefix + "*.sham")
        res.sort()

        torem = res[1:-3]
        # print(res,torem)

        for f in torem:
            os.remove(f)


idump_last_dump = get_last_dump()

print("Last dump:", idump_last_dump)

# %%
# Load the last dump if it exists, setup otherwise

if idump_last_dump is not None:
    model.load_from_dump(get_dump_name(idump_last_dump))
else:
    # Generate the default config
    cfg = model.gen_default_config()
    cfg.set_artif_viscosity_ConstantDisc(alpha_u=alpha_u, alpha_AV=alpha_AV, beta_AV=beta_AV)
    cfg.set_eos_locally_isothermalLP07(cs0=cs0, q=q, r0=r0)

    cfg.add_ext_force_point_mass(center_mass, center_racc)
    cfg.add_kill_sphere(center=(0, 0, 0), radius=bsize)  # kill particles outside the simulation box

    cfg.set_units(codeu)
    cfg.set_particle_mass(pmass)
    # Set the CFL
    cfg.set_cfl_cour(C_cour)
    cfg.set_cfl_force(C_force)

    # Set the solver config to be the one stored in cfg
    model.set_solver_config(cfg)

    # Print the solver config
    model.get_current_config().print_status()

    # Init the scheduler & fields
    model.init_scheduler(scheduler_split_val, scheduler_merge_val)

    # Set the simulation box size
    model.resize_simulation_box(bmin, bmax)

    # Create the setup

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

    # Print the dot graph of the setup
    print(gen_disc.get_dot())

    # Apply the setup
    setup.apply_setup(gen_disc)

    # Run a single step to init the integrator and smoothing length of the particles
    # Here the htolerance is the maximum factor of evolution of the smoothing length in each
    # Smoothing length iterations, increasing it affect the performance negatively but increse the
    # convergence rate of the smoothing length
    # this is why we increase it temporely to 1.3 before lowering it back to 1.1 (default value)
    # Note that both ``change_htolerance`` can be removed and it will work the same but would converge
    # more slowly at the first timestep

    model.change_htolerance(1.3)
    model.timestep()
    model.change_htolerance(1.1)


# %%
# On the fly analysis
def save_rho_integ(ext, arr_rho, iplot):
    if shamrock.sys.world_rank() == 0:
        metadata = {"extent": [-ext, ext, -ext, ext], "time": model.get_time()}
        np.save(dump_folder + f"rho_integ_{iplot:07}.npy", arr_rho)

        with open(dump_folder + f"rho_integ_{iplot:07}.json", "w") as fp:
            json.dump(metadata, fp)


def analysis_plot(iplot):

    x1, y1, z1 = (0, 0, 0)  # central star pos

    ext = rout * 1.5
    nx = 1024
    ny = 1024

    arr_rho2 = model.render_cartesian_column_integ(
        "rho",
        "f64",
        center=(0.0, 0.0, 0.0),
        delta_x=(ext * 2, 0, 0.0),
        delta_y=(0.0, ext * 2, 0.0),
        nx=nx,
        ny=ny,
    )

    save_rho_integ(ext, arr_rho2, iplot)


# %%
# Evolve the simulation
t_start = model.get_time()

idump = 0
iplot = 0
istop = 0
for ttarg in t_stop:

    if ttarg >= t_start:
        model.evolve_until(ttarg)

        if istop % dump_freq_stop == 0:
            model.do_vtk_dump(get_vtk_dump_name(idump), True)
            model.dump(get_dump_name(idump))

            # dump = model.make_phantom_dump()
            # dump.save_dump(get_ph_dump_name(idump))

            purge_old_dumps()

        if istop % plot_freq_stop == 0:
            analysis_plot(iplot)

    if istop % dump_freq_stop == 0:
        idump += 1

    if istop % plot_freq_stop == 0:
        iplot += 1

    istop += 1

# %%
# Generate the plots from the on-the-fly analysis
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Load the on-the-fly analysis after the run to make the plots
# (everything in this section can be in another file)

import matplotlib
import matplotlib.pyplot as plt

# Uncomment this and replace by you dump folder, here since it is just above i comment it out
# dump_folder = "my_masterpiece"
# dump_folder += "/"


def plot_rho_integ(metadata, arr_rho, iplot):

    ext = metadata["extent"]

    dpi = 200

    # Reset the figure using the same memory as the last one
    plt.figure(num=1, clear=True, dpi=dpi)
    import copy

    my_cmap = copy.copy(matplotlib.colormaps.get_cmap("gist_heat"))  # copy the default cmap
    my_cmap.set_bad(color="black")

    res = plt.imshow(
        arr_rho, cmap=my_cmap, origin="lower", extent=ext, norm="log", vmin=1e-8, vmax=1e-4
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"t = {metadata['time']:0.3f} [years]")

    cbar = plt.colorbar(res, extend="both")
    cbar.set_label(r"$\int \rho \, \mathrm{d}z$ [code unit]")

    plt.savefig(dump_folder + "plot_rho_integ_{:04}.png".format(iplot))
    plt.close()


def get_list_dumps_id():
    import glob

    list_files = glob.glob(dump_folder + "rho_integ_*.npy")
    list_files.sort()
    list_dumps_id = []
    for f in list_files:
        list_dumps_id.append(int(f.split("_")[-1].split(".")[0]))
    return list_dumps_id


def load_rho_integ(iplot):
    with open(dump_folder + f"rho_integ_{iplot:07}.json") as fp:
        metadata = json.load(fp)
    return np.load(dump_folder + f"rho_integ_{iplot:07}.npy"), metadata


if shamrock.sys.world_rank() == 0:
    for iplot in get_list_dumps_id():
        print("Rendering rho integ plot for dump", iplot)
        arr_rho, metadata = load_rho_integ(iplot)
        plot_rho_integ(metadata, arr_rho, iplot)


# %%
# Make gifs for the foc
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Convert PNG sequence to Image sequence in mpl
import matplotlib.animation as animation


def show_image_sequence(glob_str):

    if render_gif and shamrock.sys.world_rank() == 0:

        import glob

        files = sorted(glob.glob(glob_str))

        from PIL import Image

        image_array = []
        for my_file in files:
            image = Image.open(my_file)
            image_array.append(image)

        if not image_array:
            raise RuntimeError(f"Warning: No images found for glob pattern: {glob_str}")

        pixel_x, pixel_y = image_array[0].size

        # Create the figure and axes objects
        # Remove axes, ticks, and frame & set aspect ratio
        dpi = 200
        fig = plt.figure(dpi=dpi)
        plt.gca().set_position((0, 0, 1, 1))
        plt.gcf().set_size_inches(pixel_x / dpi, pixel_y / dpi)
        plt.axis("off")

        # Set the initial image with correct aspect ratio
        im = plt.imshow(image_array[0], animated=True, aspect="auto")

        def update(i):
            im.set_array(image_array[i])
            return (im,)

        # Create the animation object
        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(image_array),
            interval=50,
            blit=True,
            repeat_delay=10,
        )

        return ani


# %%
# Do it for rho integ
render_gif = True
glob_str = os.path.join(dump_folder, "plot_rho_integ_*.png")

# If the animation is not returned only a static image will be shown in the doc
ani = show_image_sequence(glob_str)

if render_gif and shamrock.sys.world_rank() == 0:
    # To save the animation using Pillow as a gif
    # writer = animation.PillowWriter(fps=15,
    #                                 metadata=dict(artist='Me'),
    #                                 bitrate=1800)
    # ani.save('scatter.gif', writer=writer)

    # Show the animation
    plt.show()
