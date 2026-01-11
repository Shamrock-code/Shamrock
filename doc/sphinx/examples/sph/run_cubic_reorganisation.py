"""
Cubic lattice reorganisation in SPH
===================================

The cubic lattice is not stable in SPH, this test is a way to explore that.
"""

# sphinx_gallery_multi_image = "single"

import os
import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import shamrock

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

# %%
# Setup parameters

gamma = 5.0 / 3.0
rho_g = 1
P_0 = 1.0
u_0 = P_0 / ((gamma - 1.0) * rho_g)
perturb = 3e-3

bmin = (-0.6, -0.6, -0.6)
bmax = (0.6, 0.6, 0.6)

dump_folder = "_to_trash"
sim_name = "cubic_reorganisation"

N_side = 8
scheduler_split_val = int(2e7)
scheduler_merge_val = int(1)

os.makedirs(dump_folder, exist_ok=True)

# %%
# Deduced quantities

xm, ym, zm = bmin
xM, yM, zM = bmax
vol_b = (xM - xm) * (yM - ym) * (zM - zm)

dr = (bmax[0] - bmin[0]) / N_side

pmass = -1

# %%
# Setup

ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M6")

cfg = model.gen_default_config()
cfg.set_artif_viscosity_VaryingCD10(
    alpha_min=0.0, alpha_max=1, sigma_decay=0.1, alpha_u=1, beta_AV=2
)
cfg.set_boundary_periodic()
cfg.set_eos_adiabatic(gamma)
cfg.print_status()
model.set_solver_config(cfg)
model.init_scheduler(scheduler_split_val, scheduler_merge_val)


model.resize_simulation_box(bmin, bmax)

setup = model.get_setup()
gen = setup.make_generator_lattice_cubic(dr, bmin, bmax)
setup.apply_setup(gen, insert_step=scheduler_split_val)


xc, yc, zc = model.get_closest_part_to((0, 0, 0))

if shamrock.sys.world_rank() == 0:
    print("closest part to (0,0,0) is in :", xc, yc, zc)


vol_b = (xM - xm) * (yM - ym) * (zM - zm)

totmass = rho_g * vol_b

pmass = model.total_mass_to_part_mass(totmass)

model.set_value_in_a_box("uint", "f64", u_0, bmin, bmax)

tot_u = pmass * model.get_sum("uint", "f64")
if shamrock.sys.world_rank() == 0:
    print("total u :", tot_u)

model.set_particle_mass(pmass)

model.set_cfl_cour(0.1)
model.set_cfl_force(0.1)


def periodic_modulo(x, xmin, xmax):
    tmp = x - xmin
    tmp = tmp % (xmax - xmin)
    tmp += xmin
    return tmp


random.seed(111)


def perturb_func(r):
    x, y, z = r
    x += perturb * (random.random() - 0.5) * 2
    y += perturb * (random.random() - 0.5) * 2
    z += perturb * (random.random() - 0.5) * 2

    x = periodic_modulo(x, xm, xM)
    y = periodic_modulo(y, ym, yM)
    z = periodic_modulo(z, zm, zM)

    return (x, y, z)


model.remap_positions(perturb_func)

# %%
# Single timestep to iterate the smoothing length
model.timestep()

hmean = None
scatter_range = None


def make_plot(model, iplot):
    # %%
    # Recover data
    dat = ctx.collect_data()

    min_hpart = np.min(dat["hpart"])
    max_hpart = np.max(dat["hpart"])
    mean_hpart = np.mean(dat["hpart"])

    global hmean
    if hmean is None:
        hmean = mean_hpart

    global scatter_range
    if scatter_range is None:
        scatter_range = (min_hpart, max_hpart)

    print(f"hpart min={min_hpart} max={max_hpart} delta={max_hpart-min_hpart}")

    # Compute all pairwise distances
    from scipy.spatial.distance import pdist

    pairwise_distances = pdist(dat["xyz"])  # Returns condensed distance matrix
    print(f"Number of particle pairs: {len(pairwise_distances)}")
    print(
        f"Distance min={np.min(pairwise_distances):.6f} max={np.max(pairwise_distances):.6f} mean={np.mean(pairwise_distances):.6f}"
    )

    # %%
    # Plot particle distrib

    fig = plt.figure(dpi=120, figsize=(10, 5))
    ax = fig.add_subplot(121, projection="3d")
    ax.set_xlim3d(bmin[0], bmax[0])
    ax.set_ylim3d(bmin[1], bmax[1])
    ax.set_zlim3d(bmin[2], bmax[2])
    ax.set_aspect("equal")

    cm = matplotlib.colormaps["viridis"]
    sc = ax.scatter(
        dat["xyz"][:, 0],
        dat["xyz"][:, 1],
        dat["xyz"][:, 2],
        s=5,
        vmin=scatter_range[0],
        vmax=scatter_range[1],
        c=dat["hpart"],
        cmap=cm,
        edgecolors="black",
        linewidths=0.5,
    )
    cbar = plt.colorbar(sc, ax=ax, orientation="horizontal", pad=0.1, aspect=30)
    cbar.set_label("hpart")
    ax.set_title(f"t = {model.get_time():0.3f}")

    ax = fig.add_subplot(122)

    # Filter distances to the display range
    max_dist = hmean * 3
    filtered_distances = pairwise_distances[pairwise_distances <= max_dist] / hmean

    # Create histogram and normalize by distance^2
    bins = 500
    counts, bin_edges = np.histogram(filtered_distances, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Avoid division by zero for the first bin
    normalized_counts = np.where(bin_centers > 0, counts / (bin_centers**2), 0)

    normalized_counts = normalized_counts / len(dat["xyz"])

    ax.bar(bin_centers, normalized_counts, width=bin_edges[1] - bin_edges[0], align="center")
    ax.set_xlabel("distances / hmean")
    ax.set_ylabel("counts / (r^2 * number of particles)")
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1.5)

    plt.tight_layout()
    plt.savefig(os.path.join(dump_folder, f"{sim_name}_{iplot:04}.png"))
    plt.close(fig)


tcur = 0
for iplot in range(50):
    model.evolve_until(tcur)
    make_plot(model, iplot)
    tcur += 0.5


####################################################
# Convert PNG sequence to Image sequence in mpl
####################################################
import matplotlib.animation as animation

render_gif = True


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


# If the animation is not returned only a static image will be shown in the doc
glob_str = os.path.join(dump_folder, f"{sim_name}_*.png")
ani = show_image_sequence(glob_str)

if render_gif and shamrock.sys.world_rank() == 0:
    # To save the animation using Pillow as a gif
    # writer = animation.PillowWriter(fps=15, metadata=dict(artist="Me"), bitrate=1800)
    # ani.save("scatter.gif", writer=writer)

    # Show the animation
    plt.show()
