"""
Mirror symmetry of the RAMSES solver AMR
========================================

CI test for the refinement and derefinement of the new AMR (``cfg.set_amr_mode_old(False)``) of
the RAMSES solver.

An overdense and overpressured sphere, centered in a periodic box, expands. The density based
AMR (``cfg.set_amr_mode_density_based``) refines the blocks of the sphere, and merges them back
once the density has dropped. The test checks that both the refinement and the derefinement
happen.

The setup is symmetric under the mirrors :math:`x \\to -x`, :math:`y \\to -y`, :math:`z \\to -z`
about the box center and under the swaps of two axes, so the solution is expected to stay
symmetric to round-off through the refinement and derefinement passes. The asymmetries are
measured and shown as a diagnostic, they are not enforced by this test.

The siblings of a merged block are stored in Morton order (z fastest), while the cells of a block
are indexed x fastest. Filling the merged block in memory order would transpose x and z inside
it, breaking the x and z mirrors and the x/y and y/z swaps at the first derefinement pass, while
leaving the y mirror and the x/z swap intact. This signature, asymmetries of order
:math:`10^{-1}` for the x and z mirrors and the x/y and y/z swaps while the y mirror and the x/z
swap stay at round-off, is the one to look for in the plots below.
"""

# sphinx_gallery_multi_image = "single"

import glob
import itertools
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm

import shamrock

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

# the new AMR is experimental
shamrock.enable_experimental_features()

# %%
# Use shamrock documentation style for matplotlib
shamrock.matplotlib.set_shamrock_mpl_style()

# %%
# Setup parameters
# ^^^^^^^^^^^^^^^^

gamma = 5.0 / 3.0

# sphere of radius R0 centered in the unit box
R0 = 0.2
rho_in, P_in = 1.0, 1.0
rho_out, P_out = 0.1, 0.01

# base grid of base^3 blocks of 2^3 cells, refined at most max_amr_lev times
base = 8
max_amr_lev = 2
sz = 2 << max_amr_lev  # int coords size of a base block (cells of the finest level are 1 wide)
scale_fact = 1.0 / (sz * base)  # a power of two: the cell sizes and positions are exact

# a block is refined if its mass is above 8 * crit_mass and merged if it is below crit_mass:
# the base blocks of the sphere are refined once, and merged back when the sphere has expanded
crit_mass = 1e-4

t_end = 0.3
n_frames = 31

# reference level for the asymmetries, only drawn on the plots (round-off is ~1e-15)
tol = 1e-10

sim_folder = "_to_trash/ramses_amr_mirror_symmetry/"

if shamrock.sys.world_rank() == 0:
    os.makedirs(sim_folder, exist_ok=True)
    # remove the frames of a previous run
    for f in glob.glob(os.path.join(sim_folder, "frame_*.png")):
        os.remove(f)

# %%
# Setup the solver
# ^^^^^^^^^^^^^^^^

ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_Ramses(context=ctx, vector_type="f64_3", grid_repr="i64_3")

cfg = model.gen_default_config()
cfg.set_scale_factor(scale_fact)
cfg.set_eos_gamma(gamma)
cfg.set_slope_lim_minmod()
cfg.set_face_time_interpolation(True)
cfg.set_riemann_solver_hllc()
cfg.set_amr_mode_old(False)
cfg.set_amr_mode_density_based(crit_mass=crit_mass)
model.set_solver_config(cfg)

model.init_scheduler(int(1e7), 1)
model.make_base_grid((0, 0, 0), (sz, sz, sz), (base, base, base))

# %%
# Initial conditions
# ^^^^^^^^^^^^^^^^^^


def is_in_sphere(rmin, rmax):
    center = 0.5 * (np.array(rmin) + np.array(rmax))
    return np.linalg.norm(center - 0.5) < R0


def rho_map(rmin, rmax):
    return rho_in if is_in_sphere(rmin, rmax) else rho_out


def rhoetot_map(rmin, rmax):
    return (P_in if is_in_sphere(rmin, rmax) else P_out) / (gamma - 1.0)


model.set_field_value_lambda_f64("rho", rho_map)
model.set_field_value_lambda_f64("rhoetot", rhoetot_map)
model.set_field_value_lambda_f64_3("rhovel", lambda rmin, rmax: (0.0, 0.0, 0.0))

# %%
# Symmetry diagnostics
# ^^^^^^^^^^^^^^^^^^^^
# The fields are sampled on a :math:`32^3` lattice symmetric about the box center. Mirroring the
# lattice is reversing an axis of the sampled array, swapping two axes of the box is transposing
# it. The asymmetry is the maximum of :math:`|f - S(f)|` over the lattice, relative to the maximum
# of :math:`|f|`, for each symmetry :math:`S`.

n_lattice = 32
x_lattice = (np.arange(n_lattice) + 0.5) / n_lattice
xx, yy, zz = np.meshgrid(x_lattice, x_lattice, x_lattice, indexing="ij")
lattice = list(zip(xx.ravel().tolist(), yy.ravel().tolist(), zz.ravel().tolist()))

symmetries = {
    "mirror_x": lambda r: r[::-1, :, :],
    "mirror_y": lambda r: r[:, ::-1, :],
    "mirror_z": lambda r: r[:, :, ::-1],
    "swap_xy": lambda r: r.transpose(1, 0, 2),
    "swap_xz": lambda r: r.transpose(2, 1, 0),
    "swap_yz": lambda r: r.transpose(0, 2, 1),
}
fields = ["rho", "rhoetot"]


def get_asymmetries():
    ret = {}
    for field in fields:
        r = np.array(model.render_slice(field, "f64", lattice))
        r = r.reshape(n_lattice, n_lattice, n_lattice)
        for name, sym in symmetries.items():
            ret[(field, name)] = np.max(np.abs(r - sym(r))) / np.max(np.abs(r))
    return ret


def get_blocks():
    """Block bounds in int coords and number of blocks per level (0 is the base grid)"""
    dic = ctx.collect_data()
    block_min = np.array(dic["cell_min"], dtype=float)
    block_max = np.array(dic["cell_max"], dtype=float)
    levels = np.rint(np.log2(sz / (block_max[:, 0] - block_min[:, 0]))).astype(int)
    count = {int(lev): int(np.sum(levels == lev)) for lev in np.unique(levels)}
    return block_min, block_max, count


# %%
# Plotting utilities
# ^^^^^^^^^^^^^^^^^^
# Each frame shows the density in a :math:`z` slice through the centers of the finest cells just
# below the mid-plane, with the edges of the AMR blocks, the deviation of this slice from its
# mirror :math:`x \to -x`, and the history of the asymmetries of the density on the lattice.

n_slice = 256
z_slice_int = sz * base / 2 - 0.5
x_slice = (np.arange(n_slice) + 0.5) / n_slice
sx, sy = np.meshgrid(x_slice, x_slice, indexing="ij")
slice_positions = [
    (x, y, z_slice_int * scale_fact) for x, y in zip(sx.ravel().tolist(), sy.ravel().tolist())
]


def block_edges_in_slice(block_min, block_max):
    sel = (block_min[:, 2] <= z_slice_int) & (z_slice_int < block_max[:, 2])
    segs = []
    for lo, hi in zip(block_min[sel] * scale_fact, block_max[sel] * scale_fact):
        x0, y0, x1, y1 = lo[0], lo[1], hi[0], hi[1]
        segs += [[(x0, y0), (x1, y0)], [(x1, y0), (x1, y1)]]
        segs += [[(x1, y1), (x0, y1)], [(x0, y1), (x0, y0)]]
    return segs


def plot_frame(iframe, t, hist_t, hist_asym, block_min, block_max, count):
    rho = np.array(model.render_slice("rho", "f64", slice_positions)).reshape(n_slice, n_slice)
    dev = np.abs(rho - rho[::-1, :]) / np.max(rho)

    fig, axs = plt.subplots(1, 3, figsize=(16, 5), dpi=90)

    res = axs[0].imshow(
        rho.T,
        origin="lower",
        extent=[0, 1, 0, 1],
        cmap=matplotlib.colormaps["magma"],
        vmin=0,
        vmax=rho_in,
    )
    axs[0].add_collection(
        LineCollection(block_edges_in_slice(block_min, block_max), colors="white", linewidths=0.5)
    )
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].set_title(f"density and AMR blocks, t = {t:0.3f}\nblocks per level {count}")
    fig.colorbar(res, ax=axs[0], fraction=0.046).set_label(r"$\rho$")

    res = axs[1].imshow(
        np.maximum(dev, 1e-17).T,
        origin="lower",
        extent=[0, 1, 0, 1],
        cmap=matplotlib.colormaps["viridis"],
        norm=LogNorm(vmin=1e-16, vmax=1e-1),
    )
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("y")
    axs[1].set_title(r"$|\rho - \rho_{\rm mirror\ x}| / \max \rho$ on the slice")
    fig.colorbar(res, ax=axs[1], fraction=0.046)

    for name in symmetries:
        axs[2].semilogy(hist_t, np.maximum(hist_asym["rho", name], 1e-17), "-o", ms=3, label=name)
    axs[2].axhline(tol, color="k", ls="--", lw=1, label="round-off level")
    axs[2].set_xlim(0, t_end)
    axs[2].set_ylim(1e-17, 1)
    axs[2].set_xlabel("t")
    axs[2].set_title(r"asymmetry of $\rho$ on the lattice")
    axs[2].legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(os.path.join(sim_folder, f"frame_{iframe:04d}.png"))
    plt.close(fig)


# %%
# Run the simulation
# ^^^^^^^^^^^^^^^^^^

hist_t = []
hist_asym = {(field, name): [] for field in fields for name in symmetries}
hist_count = []

for i, t in enumerate(np.linspace(0, t_end, n_frames)):
    model.evolve_until(t)

    block_min, block_max, count = get_blocks()
    hist_t.append(t)
    hist_count.append(count)
    for k, v in get_asymmetries().items():
        hist_asym[k].append(v)

    plot_frame(i, t, hist_t, hist_asym, block_min, block_max, count)

# %%
# Evolution of the density and of the AMR blocks
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

from shamrock.utils.plot import show_image_sequence

# If the animation is not returned only a static image will be shown in the doc
ani = show_image_sequence(os.path.join(sim_folder, "frame_*.png"), render_gif=True)

if shamrock.sys.world_rank() == 0:
    # To save the animation using Pillow as a gif
    writer = animation.PillowWriter(fps=5, metadata=dict(artist="Me"), bitrate=1800)
    ani.save(os.path.join(sim_folder, "rho_amr_mirror_symmetry.gif"), writer=writer)

plt.show()

# %%
# Asymmetries
# ^^^^^^^^^^^
# The maximum asymmetry of the density and of the total energy on the lattice, for each symmetry,
# with the number of blocks of the refined level.

n_fine = [count.get(1, 0) for count in hist_count]

fig, axs = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
for field, ls in zip(fields, ["-", "--"]):
    for icol, name in enumerate(symmetries):
        axs[0].semilogy(
            hist_t,
            np.maximum(hist_asym[field, name], 1e-17),
            ls,
            color=f"C{icol}",
            label=f"{field} {name}",
        )
axs[0].axhline(tol, color="k", ls=":", lw=1, label="round-off level")
axs[0].set_ylim(1e-17, 1)
axs[0].set_ylabel("asymmetry")
axs[0].legend(ncol=2, fontsize=7)
axs[1].plot(hist_t, n_fine, "-o")
axs[1].set_xlabel("t")
axs[1].set_ylabel("level 1 blocks")
plt.tight_layout()
plt.show()

# %%
# Checks
# ^^^^^^
# Both refinement and derefinement must have happened. The maximum asymmetry of each symmetry is
# printed as a diagnostic: it is expected to stay at round-off, but it is not enforced here.

refined = max(n_fine) > 0
derefined = any(b < a for a, b in itertools.pairwise(n_fine))
print(f"number of level 1 blocks: {n_fine}")

errors = []
if not refined:
    errors.append("no block has been refined, the test setup is not exercising the AMR")
if not derefined:
    errors.append("no block has been merged, the test setup is not exercising the derefinement")

for (field, name), hist in hist_asym.items():
    print(f"max {field} {name} asymmetry: {max(hist):.2e}")

if errors:
    raise RuntimeError("AMR refine/derefine test failed:\n" + "\n".join(errors))

print("AMR refine/derefine test passed")
