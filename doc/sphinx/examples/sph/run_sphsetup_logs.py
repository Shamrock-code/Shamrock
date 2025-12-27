"""
SPH Setup logs
==============

This simple example shows how to run the SPH setup while dumping the logs and render the steps.

In general it boils down to setting ``do_setup_log=True``:

.. code-block:: python

    setup.apply_setup(
        gen,
        insert_step=int(scheduler_split_val / 4),
        msg_count_limit=32, # Maximum number of message send & received per ranks per steps
        msg_size_limit=scheduler_split_val // 4, # Max of the sum of the r&s messages size per steps
        do_setup_log=True, # Dump the logs
    )
"""

# %%
# Run a sedov setup

import json
import os

import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

import shamrock

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

gamma = 5.0 / 3.0
rho_g = 1
target_tot_u = 1

bmin = (-0.6, -0.6, -0.6)
bmax = (0.6, 0.6, 0.6)

N_target = 1e6
scheduler_split_val = int(1e6 / 16)
scheduler_merge_val = int(1)

# render exemple
"""
4 processes (at the end of the file)
"""

xm, ym, zm = bmin
xM, yM, zM = bmax
vol_b = (xM - xm) * (yM - ym) * (zM - zm)

part_vol = vol_b / N_target

# lattice volume
part_vol_lattice = 0.74 * part_vol

dr = (part_vol_lattice / ((4.0 / 3.0) * 3.1416)) ** (1.0 / 3.0)

pmass = -1

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
cfg.set_eos_adiabatic(gamma)
cfg.set_max_neigh_cache_size(int(100e9))
cfg.print_status()
model.set_solver_config(cfg)
model.init_scheduler(scheduler_split_val, scheduler_merge_val)


bmin, bmax = model.get_ideal_hcp_box(dr, bmin, bmax)
xm, ym, zm = bmin
xM, yM, zM = bmax

model.resize_simulation_box(bmin, bmax)

setup = model.get_setup()
gen = setup.make_generator_lattice_hcp(dr, bmin, bmax)

# On aurora /2 was correct to avoid out of memory
setup.apply_setup(
    gen,
    insert_step=int(scheduler_split_val / 4),
    msg_count_limit=32,
    msg_size_limit=scheduler_split_val // 4,
    do_setup_log=True,
)

# %%
# Utility to render the logs
# Copy paste this one if you want to do it outside of this setup

folder = "_to_trash/sph_setup_logs"
os.makedirs(folder, exist_ok=True)


def print_setup_logs(filepath, name_png_prefix):

    with open(filepath, "r") as file:
        data = json.load(file)

    max_count = np.max([np.max(step["count_per_rank"]) for step in data])
    print("Max count: ", max_count)

    max_msg_size = np.max(
        [np.max([indices_size for _, _, indices_size in step["msg_list"]] + [0]) for step in data]
    )
    print("Max msg size: ", max_msg_size)

    for step_idx, step in enumerate(data):
        world_size = len(step["count_per_rank"])

        comm_matrix = np.zeros((world_size, world_size))

        for msg in step["msg_list"]:
            sender_rank, receiver_rank, indices_size = msg
            comm_matrix[sender_rank][receiver_rank] = indices_size

        # Create figure with better layout
        fig = plt.figure(figsize=(14, 10), dpi=125)
        fig.suptitle(f"Setup Step {step_idx}", fontsize=16, fontweight="bold")

        # Create GridSpec layout: counts on left, matrix in middle, colorbar on right
        gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1, 3, 0.15], wspace=0.02)

        # Count per rank subplot (left)
        ax2 = fig.add_subplot(gs[0, 0])
        ax2.barh(
            np.arange(world_size), step["count_per_rank"], height=0.8, color="steelblue", alpha=0.7
        )
        ax2.set_xlabel("Particle Count", fontsize=11)
        ax2.set_ylabel("Sender Rank", fontsize=12)
        ax2.set_xlim(0, max_count * 1.1)
        ax2.grid(True, alpha=0.3, linestyle="--", axis="x")
        ax2.invert_yaxis()  # Match imshow orientation
        ax2.invert_xaxis()  # Bars grow towards the matrix
        ax2.tick_params(axis="x", rotation=45)

        # Communication matrix subplot (middle, shares y-axis)
        ax1 = fig.add_subplot(gs[0, 1], sharey=ax2)

        # Create a masked array to handle zeros
        comm_matrix_masked = np.ma.masked_where(comm_matrix == 0, comm_matrix)

        # Create colormap with black for null values
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color="black")

        # Use logarithmic normalization
        vmin = np.min(comm_matrix[comm_matrix > 0]) if np.any(comm_matrix > 0) else 1
        norm = mcolors.LogNorm(vmin=1, vmax=max(max_msg_size, 2))

        im = ax1.imshow(
            comm_matrix_masked, cmap=cmap, aspect="equal", interpolation="nearest", norm=norm
        )
        ax1.set_xlabel("Receiver Rank", fontsize=12)
        ax1.set_title(
            "Communication Matrix & Particle Count per Rank (log scale)", fontsize=14, pad=10
        )
        ax1.grid(False)
        ax1.tick_params(labelleft=False)  # Hide y-axis labels since they're shared

        # Colorbar on the right
        cax = fig.add_subplot(gs[0, 2])
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label("Message Size (log scale)", rotation=270, labelpad=20, fontsize=11)

        # Add statistics as text
        total_count = np.sum(step["count_per_rank"])
        avg_count = np.mean(step["count_per_rank"])
        std_count = np.std(step["count_per_rank"])
        stats_text = f"Total: {total_count:,}\nAvg: {avg_count:.1f}\nStd: {std_count:.1f}"
        ax2.text(
            0.02,
            0.02,
            stats_text,
            transform=ax2.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            horizontalalignment="left",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()
        plt.savefig(folder + f"/{name_png_prefix}_{step_idx:05d}.png")
        plt.close()


# %%
# Make gif for the doc (plot_to_gif.py)
#
# Convert PNG sequence to Image sequence in mpl

# sphinx_gallery_multi_image = "single"

import matplotlib.animation as animation


def show_image_sequence(glob_str):
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
# Do it for setup logs
print_setup_logs("setup_log_step.json", "setup_log_step")

# %%
# Make a gif out of it
render_gif = True
glob_str = os.path.join(folder, "setup_log_step_*.png")

# If the animation is not returned only a static image will be shown in the doc
ani = show_image_sequence(glob_str)

# To save the animation using Pillow as a gif
writer = animation.PillowWriter(fps=15, metadata=dict(artist="Me"), bitrate=1800)
ani.save(folder + "setup_log_step.gif", writer=writer)

# Show the animation
plt.show()


# %%
# Same as above but on an exemple with 4 processes

# %%
# .. raw:: html
#
#   <details>
#   <summary><a>Exemple data on same setup but 4 processes</a></summary>
#

saved_data = [
    {"count_per_rank": [5580, 15001, 15015, 15015], "msg_list": [], "step_counter": 0},
    {"count_per_rank": [20505, 30016, 30032, 30030], "msg_list": [], "step_counter": 1},
    {"count_per_rank": [35520, 44941, 45047, 45045], "msg_list": [], "step_counter": 2},
    {"count_per_rank": [50536, 59957, 59972, 60060], "msg_list": [], "step_counter": 3},
    {"count_per_rank": [65551, 74972, 74988, 74986], "msg_list": [], "step_counter": 4},
    {"count_per_rank": [80566, 89987, 90003, 90001], "msg_list": [], "step_counter": 5},
    {"count_per_rank": [95492, 105003, 105018, 105016], "msg_list": [], "step_counter": 6},
    {"count_per_rank": [110507, 119928, 120034, 120032], "msg_list": [], "step_counter": 7},
    {"count_per_rank": [125522, 134943, 134959, 135047], "msg_list": [], "step_counter": 8},
    {"count_per_rank": [140538, 149959, 149974, 149972], "msg_list": [], "step_counter": 9},
    {"count_per_rank": [155553, 164974, 164990, 164988], "msg_list": [], "step_counter": 10},
    {"count_per_rank": [170478, 179989, 180005, 180003], "msg_list": [], "step_counter": 11},
    {"count_per_rank": [185493, 194916, 195020, 195018], "msg_list": [], "step_counter": 12},
    {"count_per_rank": [200508, 209931, 209945, 210034], "msg_list": [], "step_counter": 13},
    {"count_per_rank": [215524, 224946, 224960, 224959], "msg_list": [], "step_counter": 14},
    {"count_per_rank": [230539, 239962, 239976, 239974], "msg_list": [], "step_counter": 15},
    {"count_per_rank": [245464, 254977, 254991, 246088], "msg_list": [], "step_counter": 16},
    {"count_per_rank": [260022, 268499, 254991, 246088], "msg_list": [], "step_counter": 17},
    {"count_per_rank": [244397, 268499, 254991, 246088], "msg_list": [], "step_counter": 18},
    {"count_per_rank": [228772, 268499, 254991, 246088], "msg_list": [], "step_counter": 19},
    {"count_per_rank": [213147, 268499, 254991, 246088], "msg_list": [], "step_counter": 20},
    {"count_per_rank": [197522, 268499, 254991, 246088], "msg_list": [], "step_counter": 21},
    {"count_per_rank": [181897, 268499, 254991, 246088], "msg_list": [], "step_counter": 22},
    {"count_per_rank": [150647, 237249, 223741, 214838], "msg_list": [], "step_counter": 23},
    {"count_per_rank": [136436, 205999, 192491, 184599], "msg_list": [], "step_counter": 24},
    {
        "count_per_rank": [136436, 205999, 192491, 184599],
        "msg_list": [[1, 0, 201354], [2, 0, 191172], [3, 0, 184546]],
        "step_counter": 25,
    },
    {
        "count_per_rank": [152061, 201354, 191172, 168921],
        "msg_list": [[1, 0, 201354], [2, 0, 191172], [3, 0, 184546]],
        "step_counter": 26,
    },
    {
        "count_per_rank": [152061, 201354, 191172, 168921],
        "msg_list": [
            [0, 1, 50663],
            [0, 2, 50663],
            [0, 3, 45457],
            [1, 2, 67145],
            [1, 0, 67104],
            [1, 3, 67105],
            [2, 1, 63823],
            [2, 0, 63673],
            [2, 3, 63676],
            [3, 1, 56327],
            [3, 2, 56330],
            [3, 0, 56264],
        ],
        "step_counter": 27,
    },
    {
        "count_per_rank": [146783, 185729, 206797, 168921],
        "msg_list": [
            [0, 1, 50663],
            [0, 2, 50663],
            [0, 3, 45457],
            [1, 2, 67145],
            [1, 0, 67104],
            [1, 3, 67105],
            [2, 1, 63823],
            [2, 0, 63673],
            [2, 3, 63676],
            [3, 1, 56327],
            [3, 2, 56330],
            [3, 0, 56264],
        ],
        "step_counter": 28,
    },
    {
        "count_per_rank": [146783, 185729, 206797, 168921],
        "msg_list": [
            [0, 1, 50663],
            [0, 2, 50663],
            [0, 3, 45457],
            [1, 2, 51520],
            [1, 0, 67104],
            [1, 3, 67105],
            [2, 1, 63823],
            [2, 0, 63673],
            [2, 3, 63676],
            [3, 1, 56327],
            [3, 2, 56330],
            [3, 0, 56264],
        ],
        "step_counter": 29,
    },
    {
        "count_per_rank": [162408, 185729, 175547, 168921],
        "msg_list": [
            [0, 1, 50663],
            [0, 2, 50663],
            [0, 3, 45457],
            [1, 2, 51520],
            [1, 0, 67104],
            [1, 3, 67105],
            [2, 1, 63823],
            [2, 0, 63673],
            [2, 3, 63676],
            [3, 1, 56327],
            [3, 2, 56330],
            [3, 0, 56264],
        ],
        "step_counter": 30,
    },
    {
        "count_per_rank": [162408, 185729, 175547, 168921],
        "msg_list": [
            [0, 1, 50663],
            [0, 2, 50663],
            [0, 3, 45457],
            [1, 2, 51520],
            [1, 0, 67104],
            [1, 3, 67105],
            [2, 0, 48048],
            [2, 1, 63823],
            [2, 3, 63676],
            [3, 1, 56327],
            [3, 2, 56330],
            [3, 0, 56264],
        ],
        "step_counter": 31,
    },
    {
        "count_per_rank": [162408, 170104, 159922, 184546],
        "msg_list": [
            [0, 1, 50663],
            [0, 2, 50663],
            [0, 3, 45457],
            [1, 2, 51520],
            [1, 0, 67104],
            [1, 3, 67105],
            [2, 0, 48048],
            [2, 1, 63823],
            [2, 3, 63676],
            [3, 1, 56327],
            [3, 2, 56330],
            [3, 0, 56264],
        ],
        "step_counter": 32,
    },
    {
        "count_per_rank": [162408, 170104, 159922, 184546],
        "msg_list": [
            [0, 1, 50663],
            [0, 2, 50663],
            [0, 3, 45457],
            [1, 0, 51479],
            [1, 2, 51520],
            [1, 3, 67105],
            [2, 0, 48048],
            [2, 3, 48051],
            [2, 1, 63823],
            [3, 1, 56327],
            [3, 2, 56330],
            [3, 0, 56264],
        ],
        "step_counter": 33,
    },
    {
        "count_per_rank": [162408, 185729, 144297, 153296],
        "msg_list": [
            [0, 1, 50663],
            [0, 2, 50663],
            [0, 3, 45457],
            [1, 0, 51479],
            [1, 2, 51520],
            [1, 3, 67105],
            [2, 0, 48048],
            [2, 3, 48051],
            [2, 1, 63823],
            [3, 1, 56327],
            [3, 2, 56330],
            [3, 0, 56264],
        ],
        "step_counter": 34,
    },
    {
        "count_per_rank": [162408, 185729, 144297, 153296],
        "msg_list": [
            [0, 1, 50663],
            [0, 2, 50663],
            [0, 3, 45457],
            [1, 0, 51479],
            [1, 2, 51520],
            [1, 3, 67105],
            [2, 0, 48048],
            [2, 3, 48051],
            [2, 1, 48198],
            [3, 0, 40639],
            [3, 1, 56327],
            [3, 2, 56330],
        ],
        "step_counter": 35,
    },
    {
        "count_per_rank": [131158, 185729, 128672, 168921],
        "msg_list": [
            [0, 1, 50663],
            [0, 2, 50663],
            [0, 3, 45457],
            [1, 0, 51479],
            [1, 2, 51520],
            [1, 3, 67105],
            [2, 0, 48048],
            [2, 3, 48051],
            [2, 1, 48198],
            [3, 0, 40639],
            [3, 1, 56327],
            [3, 2, 56330],
        ],
        "step_counter": 36,
    },
    {
        "count_per_rank": [131158, 185729, 128672, 168921],
        "msg_list": [
            [0, 3, 29832],
            [0, 1, 50663],
            [0, 2, 50663],
            [1, 0, 51479],
            [1, 2, 51520],
            [1, 3, 67105],
            [2, 1, 32573],
            [2, 0, 48048],
            [2, 3, 48051],
            [3, 0, 40639],
            [3, 1, 56327],
            [3, 2, 56330],
        ],
        "step_counter": 37,
    },
    {
        "count_per_rank": [131158, 185729, 113047, 153296],
        "msg_list": [
            [0, 3, 29832],
            [0, 1, 50663],
            [0, 2, 50663],
            [1, 0, 51479],
            [1, 2, 51520],
            [1, 3, 67105],
            [2, 1, 32573],
            [2, 0, 48048],
            [2, 3, 48051],
            [3, 0, 40639],
            [3, 1, 56327],
            [3, 2, 56330],
        ],
        "step_counter": 38,
    },
    {
        "count_per_rank": [131158, 185729, 113047, 153296],
        "msg_list": [
            [0, 3, 29832],
            [0, 1, 25460],
            [0, 2, 75866],
            [1, 0, 51479],
            [1, 2, 51520],
            [1, 3, 67105],
            [2, 1, 7984],
            [2, 0, 48048],
            [2, 3, 48051],
            [3, 0, 40639],
            [3, 1, 27932],
            [3, 2, 84725],
        ],
        "step_counter": 39,
    },
    {
        "count_per_rank": [146783, 154479, 128672, 137671],
        "msg_list": [
            [0, 3, 29832],
            [0, 1, 25460],
            [0, 2, 75866],
            [1, 0, 51479],
            [1, 2, 51520],
            [1, 3, 67105],
            [2, 1, 7984],
            [2, 0, 48048],
            [2, 3, 48051],
            [3, 0, 40639],
            [3, 1, 27932],
            [3, 2, 84725],
        ],
        "step_counter": 40,
    },
    {
        "count_per_rank": [146783, 154479, 128672, 137671],
        "msg_list": [
            [0, 3, 29832],
            [0, 1, 34741],
            [0, 2, 66585],
            [1, 0, 35854],
            [1, 2, 51520],
            [1, 3, 67105],
            [2, 1, 14682],
            [2, 0, 48048],
            [2, 3, 48051],
            [3, 0, 40639],
            [3, 2, 60762],
            [3, 1, 36270],
        ],
        "step_counter": 41,
    },
    {
        "count_per_rank": [115533, 170104, 126406, 122046],
        "msg_list": [
            [0, 3, 29832],
            [0, 1, 34741],
            [0, 2, 66585],
            [1, 0, 35854],
            [1, 2, 51520],
            [1, 3, 67105],
            [2, 1, 14682],
            [2, 0, 48048],
            [2, 3, 48051],
            [3, 0, 40639],
            [3, 2, 60762],
            [3, 1, 36270],
        ],
        "step_counter": 42,
    },
    {
        "count_per_rank": [115533, 170104, 126406, 122046],
        "msg_list": [
            [0, 2, 31660],
            [0, 3, 49132],
            [0, 1, 34741],
            [1, 0, 31515],
            [1, 2, 25616],
            [1, 3, 93009],
            [2, 1, 18234],
            [2, 0, 44496],
            [2, 3, 48051],
            [3, 1, 23293],
            [3, 0, 37991],
            [3, 2, 38418],
        ],
        "step_counter": 43,
    },
    {
        "count_per_rank": [115533, 150140, 95156, 137671],
        "msg_list": [
            [0, 2, 31660],
            [0, 3, 49132],
            [0, 1, 34741],
            [1, 0, 31515],
            [1, 2, 25616],
            [1, 3, 93009],
            [2, 1, 18234],
            [2, 0, 44496],
            [2, 3, 48051],
            [3, 1, 23293],
            [3, 0, 37991],
            [3, 2, 38418],
        ],
        "step_counter": 44,
    },
    {
        "count_per_rank": [115533, 150140, 95156, 106421],
        "msg_list": [
            [0, 2, 31660],
            [0, 3, 49132],
            [0, 1, 34741],
            [1, 0, 31515],
            [1, 2, 25616],
            [1, 3, 93009],
            [2, 1, 18234],
            [2, 0, 44496],
            [2, 3, 48051],
            [3, 1, 23293],
            [3, 0, 37991],
            [3, 2, 38418],
        ],
        "step_counter": 45,
    },
    {
        "count_per_rank": [115533, 150140, 95156, 106421],
        "msg_list": [
            [0, 3, 29832],
            [0, 2, 41334],
            [0, 1, 40981],
            [1, 0, 31515],
            [1, 2, 51520],
            [1, 3, 67105],
            [2, 1, 13650],
            [2, 3, 32426],
            [2, 0, 49080],
            [3, 0, 42614],
            [3, 1, 30632],
            [3, 2, 31539],
        ],
        "step_counter": 46,
    },
    {
        "count_per_rank": [127772, 150140, 95156, 89160],
        "msg_list": [
            [0, 3, 29832],
            [0, 2, 41334],
            [0, 1, 40981],
            [1, 0, 31515],
            [1, 2, 51520],
            [1, 3, 67105],
            [2, 1, 13650],
            [2, 3, 32426],
            [2, 0, 49080],
            [3, 0, 42614],
            [3, 1, 30632],
            [3, 2, 31539],
        ],
        "step_counter": 47,
    },
    {
        "count_per_rank": [127772, 150140, 95156, 89160],
        "msg_list": [
            [0, 2, 38838],
            [0, 3, 34668],
            [0, 1, 38641],
            [1, 0, 31515],
            [1, 2, 45040],
            [1, 3, 73585],
            [2, 1, 14682],
            [2, 3, 32426],
            [2, 0, 48048],
            [3, 0, 25766],
            [3, 1, 28745],
            [3, 2, 34649],
        ],
        "step_counter": 48,
    },
    {
        "count_per_rank": [96522, 150140, 110781, 89160],
        "msg_list": [
            [0, 2, 38838],
            [0, 3, 34668],
            [0, 1, 38641],
            [1, 0, 31515],
            [1, 2, 45040],
            [1, 3, 73585],
            [2, 1, 14682],
            [2, 3, 32426],
            [2, 0, 48048],
            [3, 0, 25766],
            [3, 1, 28745],
            [3, 2, 34649],
        ],
        "step_counter": 49,
    },
    {
        "count_per_rank": [96522, 150140, 110781, 89160],
        "msg_list": [
            [0, 2, 21747],
            [0, 3, 34668],
            [0, 1, 40107],
            [1, 0, 31515],
            [1, 2, 45040],
            [1, 3, 73585],
            [2, 1, 14682],
            [2, 3, 32426],
            [2, 0, 48048],
            [3, 0, 25766],
            [3, 1, 31855],
            [3, 2, 31539],
        ],
        "step_counter": 50,
    },
    {
        "count_per_rank": [96522, 150140, 79531, 104785],
        "msg_list": [
            [0, 2, 21747],
            [0, 3, 34668],
            [0, 1, 40107],
            [1, 0, 31515],
            [1, 2, 45040],
            [1, 3, 73585],
            [2, 1, 14682],
            [2, 3, 32426],
            [2, 0, 48048],
            [3, 0, 25766],
            [3, 1, 31855],
            [3, 2, 31539],
        ],
        "step_counter": 51,
    },
    {
        "count_per_rank": [96522, 150140, 79531, 104785],
        "msg_list": [
            [0, 3, 33052],
            [0, 2, 23363],
            [0, 1, 40107],
            [1, 0, 31515],
            [1, 2, 48274],
            [1, 3, 70351],
            [2, 3, 16801],
            [2, 1, 14682],
            [2, 0, 48048],
            [3, 0, 25766],
            [3, 1, 31855],
            [3, 2, 31539],
        ],
        "step_counter": 52,
    },
    {
        "count_per_rank": [96522, 150140, 95156, 73535],
        "msg_list": [
            [0, 3, 33052],
            [0, 2, 23363],
            [0, 1, 40107],
            [1, 0, 31515],
            [1, 2, 48274],
            [1, 3, 70351],
            [2, 3, 16801],
            [2, 1, 14682],
            [2, 0, 48048],
            [3, 0, 25766],
            [3, 1, 31855],
            [3, 2, 31539],
        ],
        "step_counter": 53,
    },
    {
        "count_per_rank": [96522, 150140, 95156, 73535],
        "msg_list": [
            [0, 2, 20283],
            [0, 3, 34668],
            [0, 1, 41571],
            [1, 0, 31515],
            [1, 2, 45040],
            [1, 3, 73585],
            [2, 3, 16801],
            [2, 1, 13650],
            [2, 0, 49080],
            [3, 2, 14480],
            [3, 0, 26989],
            [3, 1, 32066],
        ],
        "step_counter": 54,
    },
    {
        "count_per_rank": [96522, 134515, 79531, 89160],
        "msg_list": [
            [0, 2, 20283],
            [0, 3, 34668],
            [0, 1, 41571],
            [1, 0, 31515],
            [1, 2, 45040],
            [1, 3, 73585],
            [2, 3, 16801],
            [2, 1, 13650],
            [2, 0, 49080],
            [3, 2, 14480],
            [3, 0, 26989],
            [3, 1, 32066],
        ],
        "step_counter": 55,
    },
    {
        "count_per_rank": [96522, 134515, 79531, 89160],
        "msg_list": [
            [0, 3, 29832],
            [0, 2, 25119],
            [0, 1, 41571],
            [1, 0, 31515],
            [1, 3, 51507],
            [1, 2, 51493],
            [2, 3, 16801],
            [2, 1, 14682],
            [2, 0, 48048],
            [3, 2, 14480],
            [3, 0, 25766],
            [3, 1, 33289],
        ],
        "step_counter": 56,
    },
    {
        "count_per_rank": [112147, 118890, 79531, 73535],
        "msg_list": [
            [0, 3, 29832],
            [0, 2, 25119],
            [0, 1, 41571],
            [1, 0, 31515],
            [1, 3, 51507],
            [1, 2, 51493],
            [2, 3, 16801],
            [2, 1, 14682],
            [2, 0, 48048],
            [3, 2, 14480],
            [3, 0, 25766],
            [3, 1, 33289],
        ],
        "step_counter": 57,
    },
    {
        "count_per_rank": [112147, 118890, 79531, 73535],
        "msg_list": [
            [0, 3, 29832],
            [0, 2, 25119],
            [0, 1, 41571],
            [1, 0, 15890],
            [1, 3, 51507],
            [1, 2, 51493],
            [2, 3, 16801],
            [2, 1, 14682],
            [2, 0, 48048],
            [3, 2, 14480],
            [3, 0, 25766],
            [3, 1, 33289],
        ],
        "step_counter": 58,
    },
    {
        "count_per_rank": [112147, 118890, 63906, 73535],
        "msg_list": [
            [0, 3, 29832],
            [0, 2, 25119],
            [0, 1, 41571],
            [1, 0, 15890],
            [1, 3, 51507],
            [1, 2, 51493],
            [2, 3, 16801],
            [2, 1, 14682],
            [2, 0, 48048],
            [3, 2, 14480],
            [3, 0, 25766],
            [3, 1, 33289],
        ],
        "step_counter": 59,
    },
    {
        "count_per_rank": [112147, 118890, 63906, 73535],
        "msg_list": [
            [0, 3, 33052],
            [0, 2, 23363],
            [0, 1, 40107],
            [1, 0, 15890],
            [1, 3, 54726],
            [1, 2, 48274],
            [2, 3, 16801],
            [2, 1, 16835],
            [2, 0, 30270],
            [3, 2, 15914],
            [3, 0, 24521],
            [3, 1, 33100],
        ],
        "step_counter": 60,
    },
    {
        "count_per_rank": [80897, 118890, 79531, 73535],
        "msg_list": [
            [0, 3, 33052],
            [0, 2, 23363],
            [0, 1, 40107],
            [1, 0, 15890],
            [1, 3, 54726],
            [1, 2, 48274],
            [2, 3, 16801],
            [2, 1, 16835],
            [2, 0, 30270],
            [3, 2, 15914],
            [3, 0, 24521],
            [3, 1, 33100],
        ],
        "step_counter": 61,
    },
    {
        "count_per_rank": [80897, 118890, 79531, 73535],
        "msg_list": [
            [0, 2, 6934],
            [0, 3, 33361],
            [0, 1, 40602],
            [1, 0, 15890],
            [1, 3, 57960],
            [1, 2, 45040],
            [2, 3, 16801],
            [2, 1, 16835],
            [2, 0, 30270],
            [3, 2, 14480],
            [3, 0, 24521],
            [3, 1, 34534],
        ],
        "step_counter": 62,
    },
    {
        "count_per_rank": [80897, 118890, 48281, 89160],
        "msg_list": [
            [0, 2, 6934],
            [0, 3, 33361],
            [0, 1, 40602],
            [1, 0, 15890],
            [1, 3, 57960],
            [1, 2, 45040],
            [2, 3, 16801],
            [2, 1, 16835],
            [2, 0, 30270],
            [3, 2, 14480],
            [3, 0, 24521],
            [3, 1, 34534],
        ],
        "step_counter": 63,
    },
    {
        "count_per_rank": [80897, 118890, 48281, 89160],
        "msg_list": [
            [0, 3, 29832],
            [0, 2, 10463],
            [0, 1, 40602],
            [1, 0, 15890],
            [1, 3, 51507],
            [1, 2, 51493],
            [2, 3, 1176],
            [2, 1, 14682],
            [2, 0, 32423],
            [3, 2, 14480],
            [3, 0, 25766],
            [3, 1, 33289],
        ],
        "step_counter": 64,
    },
    {
        "count_per_rank": [80897, 103265, 48281, 89160],
        "msg_list": [
            [0, 3, 29832],
            [0, 2, 10463],
            [0, 1, 40602],
            [1, 0, 15890],
            [1, 3, 51507],
            [1, 2, 51493],
            [2, 3, 1176],
            [2, 1, 14682],
            [2, 0, 32423],
            [3, 2, 14480],
            [3, 0, 25766],
            [3, 1, 33289],
        ],
        "step_counter": 65,
    },
    {
        "count_per_rank": [80897, 103265, 48281, 89160],
        "msg_list": [
            [0, 3, 27433],
            [0, 2, 12360],
            [0, 1, 41104],
            [1, 0, 15890],
            [1, 3, 33515],
            [1, 2, 50760],
            [2, 3, 1083],
            [2, 1, 14682],
            [2, 0, 32423],
            [3, 2, 13036],
            [3, 0, 25766],
            [3, 1, 34733],
        ],
        "step_counter": 66,
    },
    {
        "count_per_rank": [65272, 103265, 48281, 89160],
        "msg_list": [
            [0, 3, 27433],
            [0, 2, 12360],
            [0, 1, 41104],
            [1, 0, 15890],
            [1, 3, 33515],
            [1, 2, 50760],
            [2, 3, 1083],
            [2, 1, 14682],
            [2, 0, 32423],
            [3, 2, 13036],
            [3, 0, 25766],
            [3, 1, 34733],
        ],
        "step_counter": 67,
    },
    {
        "count_per_rank": [65272, 103265, 48281, 89160],
        "msg_list": [
            [0, 3, 11808],
            [0, 2, 12360],
            [0, 1, 41104],
            [1, 0, 15890],
            [1, 3, 33515],
            [1, 2, 50760],
            [2, 3, 1083],
            [2, 1, 14682],
            [2, 0, 32423],
            [3, 2, 13036],
            [3, 0, 25766],
            [3, 1, 34733],
        ],
        "step_counter": 68,
    },
    {
        "count_per_rank": [49647, 115790, 48188, 73535],
        "msg_list": [
            [0, 3, 11808],
            [0, 2, 12360],
            [0, 1, 41104],
            [1, 0, 15890],
            [1, 3, 33515],
            [1, 2, 50760],
            [2, 3, 1083],
            [2, 1, 14682],
            [2, 0, 32423],
            [3, 2, 13036],
            [3, 0, 25766],
            [3, 1, 34733],
        ],
        "step_counter": 69,
    },
    {
        "count_per_rank": [49647, 115790, 48188, 73535],
        "msg_list": [
            [0, 3, 11808],
            [0, 1, 25479],
            [0, 2, 12360],
            [1, 0, 15890],
            [1, 3, 33515],
            [1, 2, 50760],
            [2, 3, 1083],
            [2, 1, 13650],
            [2, 0, 33455],
            [3, 2, 13036],
            [3, 0, 26989],
            [3, 1, 33510],
        ],
        "step_counter": 70,
    },
    {
        "count_per_rank": [65272, 84540, 63813, 57910],
        "msg_list": [
            [0, 3, 11808],
            [0, 1, 25479],
            [0, 2, 12360],
            [1, 0, 15890],
            [1, 3, 33515],
            [1, 2, 50760],
            [2, 3, 1083],
            [2, 1, 13650],
            [2, 0, 33455],
            [3, 2, 13036],
            [3, 0, 26989],
            [3, 1, 33510],
        ],
        "step_counter": 71,
    },
    {
        "count_per_rank": [65272, 84540, 63813, 57910],
        "msg_list": [
            [0, 1, 25479],
            [0, 3, 14207],
            [0, 2, 9961],
            [1, 0, 15890],
            [1, 2, 32768],
            [1, 3, 35882],
            [2, 3, 1083],
            [2, 1, 14682],
            [2, 0, 32423],
            [3, 0, 10933],
            [3, 2, 13036],
            [3, 1, 33941],
        ],
        "step_counter": 72,
    },
    {
        "count_per_rank": [34022, 100165, 48188, 57910],
        "msg_list": [
            [0, 1, 25479],
            [0, 3, 14207],
            [0, 2, 9961],
            [1, 0, 15890],
            [1, 2, 32768],
            [1, 3, 35882],
            [2, 3, 1083],
            [2, 1, 14682],
            [2, 0, 32423],
            [3, 0, 10933],
            [3, 2, 13036],
            [3, 1, 33941],
        ],
        "step_counter": 73,
    },
    {
        "count_per_rank": [34022, 100165, 48188, 57910],
        "msg_list": [
            [0, 1, 9352],
            [0, 3, 14207],
            [0, 2, 10463],
            [1, 0, 15890],
            [1, 2, 32768],
            [1, 3, 35882],
            [2, 3, 1083],
            [2, 1, 14682],
            [2, 0, 32423],
            [3, 0, 10933],
            [3, 2, 14480],
            [3, 1, 32497],
        ],
        "step_counter": 74,
    },
    {
        "count_per_rank": [40295, 93892, 47043, 43430],
        "msg_list": [
            [0, 1, 9352],
            [0, 3, 14207],
            [0, 2, 10463],
            [1, 0, 15890],
            [1, 2, 32768],
            [1, 3, 35882],
            [2, 3, 1083],
            [2, 1, 14682],
            [2, 0, 32423],
            [3, 0, 10933],
            [3, 2, 14480],
            [3, 1, 32497],
        ],
        "step_counter": 75,
    },
    {
        "count_per_rank": [40295, 93892, 47043, 43430],
        "msg_list": [
            [0, 3, 14207],
            [0, 2, 10463],
            [1, 0, 15890],
            [1, 2, 32768],
            [1, 3, 35882],
            [2, 3, 1083],
            [2, 0, 16798],
            [2, 1, 14682],
            [3, 0, 10933],
            [3, 1, 32497],
        ],
        "step_counter": 76,
    },
    {
        "count_per_rank": [35603, 68915, 48188, 32497],
        "msg_list": [
            [0, 3, 14207],
            [0, 2, 10463],
            [1, 0, 15890],
            [1, 2, 32768],
            [1, 3, 35882],
            [2, 3, 1083],
            [2, 0, 16798],
            [2, 1, 14682],
            [3, 0, 10933],
            [3, 1, 32497],
        ],
        "step_counter": 77,
    },
    {
        "count_per_rank": [35603, 68915, 48188, 32497],
        "msg_list": [
            [0, 2, 7243],
            [0, 3, 17427],
            [1, 0, 15890],
            [1, 2, 15100],
            [1, 3, 37925],
            [2, 3, 1083],
            [2, 0, 16798],
            [2, 1, 14682],
            [3, 1, 32497],
        ],
        "step_counter": 78,
    },
    {
        "count_per_rank": [24670, 84540, 32563, 16872],
        "msg_list": [
            [0, 2, 7243],
            [0, 3, 17427],
            [1, 0, 15890],
            [1, 2, 15100],
            [1, 3, 37925],
            [2, 3, 1083],
            [2, 0, 16798],
            [2, 1, 14682],
            [3, 1, 32497],
        ],
        "step_counter": 79,
    },
    {
        "count_per_rank": [24670, 84540, 32563, 16872],
        "msg_list": [
            [0, 2, 7243],
            [0, 3, 17427],
            [1, 0, 15890],
            [1, 2, 15100],
            [1, 3, 37925],
            [2, 3, 1083],
            [2, 0, 16798],
            [2, 1, 14682],
            [3, 1, 16872],
        ],
        "step_counter": 80,
    },
    {
        "count_per_rank": [9045, 83597, 16798, 33580],
        "msg_list": [
            [0, 2, 7243],
            [0, 3, 17427],
            [1, 0, 15890],
            [1, 2, 15100],
            [1, 3, 37925],
            [2, 3, 1083],
            [2, 0, 16798],
            [2, 1, 14682],
            [3, 1, 16872],
        ],
        "step_counter": 81,
    },
    {
        "count_per_rank": [9045, 83597, 16798, 33580],
        "msg_list": [
            [0, 3, 1217],
            [0, 2, 7828],
            [1, 0, 15890],
            [1, 2, 17143],
            [1, 3, 35882],
            [2, 0, 16798],
            [3, 1, 16872],
        ],
        "step_counter": 82,
    },
    {
        "count_per_rank": [9045, 53290, 32423, 16872],
        "msg_list": [
            [0, 3, 1217],
            [0, 2, 7828],
            [1, 0, 15890],
            [1, 2, 17143],
            [1, 3, 35882],
            [2, 0, 16798],
            [3, 1, 16872],
        ],
        "step_counter": 83,
    },
    {
        "count_per_rank": [9045, 53290, 32423, 16872],
        "msg_list": [
            [0, 3, 1802],
            [0, 2, 7243],
            [1, 2, 1114],
            [1, 0, 15890],
            [1, 3, 36286],
            [2, 0, 16798],
            [3, 0, 431],
            [3, 1, 16441],
        ],
        "step_counter": 84,
    },
    {
        "count_per_rank": [17858, 36551, 9530, 32066],
        "msg_list": [
            [0, 3, 1802],
            [0, 2, 7243],
            [1, 2, 1114],
            [1, 0, 15890],
            [1, 3, 36286],
            [2, 0, 16798],
            [3, 0, 431],
            [3, 1, 16441],
        ],
        "step_counter": 85,
    },
    {
        "count_per_rank": [17858, 36551, 9530, 32066],
        "msg_list": [[0, 3, 1802], [1, 0, 15890], [1, 3, 20661], [2, 0, 1173], [3, 1, 16441]],
        "step_counter": 86,
    },
    {
        "count_per_rank": [17427, 20926, 1173, 16441],
        "msg_list": [[0, 3, 1802], [1, 0, 15890], [1, 3, 20661], [2, 0, 1173], [3, 1, 16441]],
        "step_counter": 87,
    },
    {
        "count_per_rank": [17427, 20926, 1173, 16441],
        "msg_list": [[0, 3, 1802], [1, 0, 265], [1, 3, 20661], [2, 0, 1173], [3, 1, 16441]],
        "step_counter": 88,
    },
    {
        "count_per_rank": [2975, 5301, 0, 32066],
        "msg_list": [[0, 3, 1802], [1, 0, 265], [1, 3, 20661], [2, 0, 1173], [3, 1, 16441]],
        "step_counter": 89,
    },
    {
        "count_per_rank": [2975, 5301, 0, 32066],
        "msg_list": [
            [0, 2, 585],
            [0, 3, 1217],
            [1, 0, 265],
            [1, 2, 404],
            [1, 3, 4632],
            [3, 1, 16441],
        ],
        "step_counter": 90,
    },
    {
        "count_per_rank": [265, 15625, 989, 6665],
        "msg_list": [
            [0, 2, 585],
            [0, 3, 1217],
            [1, 0, 265],
            [1, 2, 404],
            [1, 3, 4632],
            [3, 1, 16441],
        ],
        "step_counter": 91,
    },
    {"count_per_rank": [265, 15625, 989, 6665], "msg_list": [[3, 1, 816]], "step_counter": 92},
    {"count_per_rank": [0, 816, 0, 0], "msg_list": [[3, 1, 816]], "step_counter": 93},
    {"count_per_rank": [0, 816, 0, 0], "msg_list": [], "step_counter": 94},
    {"count_per_rank": [0, 0, 0, 0], "msg_list": [], "step_counter": 95},
]

with open("setup_log_step.json", "w") as file:
    json.dump(saved_data, file)

# %%
# .. raw:: html
#
#   </details>

# %%
# Make the plots

print_setup_logs("setup_log_step.json", "setup_log_step_exemple")

# %%
# Make a gif out of it
render_gif = True
glob_str = os.path.join(folder, "setup_log_step_exemple_*.png")

# If the animation is not returned only a static image will be shown in the doc
ani = show_image_sequence(glob_str)

# To save the animation using Pillow as a gif
writer = animation.PillowWriter(fps=15, metadata=dict(artist="Me"), bitrate=1800)
ani.save(folder + "setup_log_step_exemple.gif", writer=writer)


# sphinx_gallery_thumbnail_number = 2

# Show the animation
plt.show()
