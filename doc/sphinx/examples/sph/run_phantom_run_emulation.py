"""
Perform a Phantom run in Shamrock
==========================================

Setup from a phantom dump and run according to the input file
"""

import os
import shutil
from urllib.request import urlretrieve

import shamrock

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

dump_folder = "_to_trash/phantom_test_sim"
if shamrock.sys.world_rank() == 0:
    # remove the folder if it exists (ok if it does not exist)
    if os.path.exists(dump_folder):
        shutil.rmtree(dump_folder)
    os.makedirs(dump_folder, exist_ok=True)

shamrock.sys.mpi_barrier()

input_file_name = "disc.in"
dump_file_name = "disc_00000.tmp"

input_file_path = os.path.join(dump_folder, input_file_name)
dump_file_path = os.path.join(dump_folder, dump_file_name)

input_file_url = "https://raw.githubusercontent.com/Shamrock-code/reference-files/refs/heads/main/phantom_disc_simulation/disc.in"
dump_file_url = "https://raw.githubusercontent.com/Shamrock-code/reference-files/refs/heads/main/phantom_disc_simulation/disc_00000.tmp"

shamrock.utils.url.download_file(input_file_url, input_file_path)
shamrock.utils.url.download_file(dump_file_url, dump_file_path)


def plot_that_rippa(ctx, model, dump_number):
    pixel_x = 1920
    pixel_y = 1080
    radius = 5
    center = (0.0, 0.0, 0.0)

    aspect = pixel_x / pixel_y
    pic_range = [-radius * aspect, radius * aspect, -radius, radius]
    delta_x = (radius * 2 * aspect, 0.0, 0.0)
    delta_y = (0.0, radius * 2, 0.0)

    arr_rho = model.render_cartesian_column_integ(
        "rho",
        "f64",
        center=(0.0, 0.0, 0.0),
        delta_x=delta_x,
        delta_y=delta_y,
        nx=pixel_x,
        ny=pixel_y,
    )

    import copy

    import matplotlib
    import matplotlib.pyplot as plt

    my_cmap = copy.copy(matplotlib.colormaps.get_cmap("gist_heat"))  # copy the default cmap
    my_cmap.set_bad(color="black")

    plt.figure(figsize=(16 / 2, 9 / 2))
    res = plt.imshow(arr_rho, cmap=my_cmap, origin="lower", extent=pic_range, norm="log", vmin=1e-9)

    cbar = plt.colorbar(res, extend="both")
    cbar.set_label(r"$\int \rho \, \mathrm{d} z$ [code unit]")
    # or r"$\rho$ [code unit]" for slices

    plt.title("t = {:0.3f} [code unit]".format(model.get_time()))
    plt.xlabel("x")
    plt.ylabel("z")
    plt.show()


ctx, model, in_params = shamrock.utils.phantom.run_phantom_simulation(
    dump_folder, "disc", callback=plot_that_rippa
)
