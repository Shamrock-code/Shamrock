"""
Start a SPH simulation from a phantom dump
==========================================

This simple example shows how to start a SPH simulation from a phantom dump
"""

# %%
# Download a phantom dump
dump_folder = "_to_trash"
import os

os.system("mkdir -p " + dump_folder)

url = "https://raw.githubusercontent.com/Shamrock-code/reference-files/refs/heads/main/blast_00010"

filename = dump_folder + "/blast_00010"

from urllib.request import urlretrieve

urlretrieve(url, filename)

# %%
# Init shamrock
import shamrock

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

ctx, model, in_params = shamrock.utils.phantom.load_simulation(
    dump_folder, dump_file_name="blast_00010", in_file_name=None
)

# %%
# Run a simple timestep just for wasting some computing time :)
#

model.timestep()

# %%
# .. note::
#    Note that shamrock has to update some smoothing lengths that were in the phantom dump
#    I think that since smoothing length is single precision in phantom they are slightly off
#    from the shamrock point of view hence the update

# %%
# Plot the result
import matplotlib.pyplot as plt

pixel_x = 1200
pixel_y = 1080
radius = 0.7
center = (0.0, 0.0, 0.0)

aspect = pixel_x / pixel_y
pic_range = [-radius * aspect, radius * aspect, -radius, radius]
delta_x = (radius * 2 * aspect, 0.0, 0.0)
delta_y = (0.0, radius * 2, 0.0)

arr_rho = model.render_cartesian_column_integ(
    "rho", "f64", center=(0.0, 0.0, 0.0), delta_x=delta_x, delta_y=delta_y, nx=pixel_x, ny=pixel_y
)

import copy

import matplotlib

my_cmap = copy.copy(matplotlib.colormaps.get_cmap("gist_heat"))  # copy the default cmap
my_cmap.set_bad(color="black")

fig_width = 6
fig_height = fig_width / aspect
plt.figure(figsize=(fig_width, fig_height))
res = plt.imshow(arr_rho, cmap=my_cmap, origin="lower", extent=pic_range)

cbar = plt.colorbar(res, extend="both")
cbar.set_label(r"$\int \rho \, \mathrm{d} z$ [code unit]")
# or r"$\rho$ [code unit]" for slices

plt.title("t = {:0.3f} [code unit]".format(model.get_time()))
plt.xlabel("x")
plt.ylabel("z")
plt.show()
