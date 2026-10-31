"""
Testing the mirror symmetry of the RAMSES solver AMR
====================================================

CI test for the refinement and derefinement of the new AMR (``cfg.set_amr_mode_old(False)``) of
the RAMSES solver.

An overdense and overpressured sphere, centered in a periodic box, expands. The density based
AMR refines the blocks of the sphere and of the shock, and merges them back once the density has
dropped. The setup is symmetric under the mirrors :math:`x \\to -x`, :math:`y \\to -y`,
:math:`z \\to -z` about the box center and under the swaps of two axes, so the density must stay
symmetric to round-off through the refinement and derefinement passes.

The siblings of a merged block are stored in Morton order (z fastest), while the cells of a block
are indexed x fastest: filling the merged block in memory order transposes x and z inside it,
and breaks the x and z mirrors and the x/y swap while leaving the y mirror and the x/z swap
intact.
"""

import itertools

import numpy as np

import shamrock

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

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
# Symmetry measurement
# ^^^^^^^^^^^^^^^^^^^^
# The fields are sampled on a lattice symmetric about the box center. Mirroring the lattice is
# reversing an axis of the sampled array, swapping two axes of the box is transposing it.

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


def render(field):
    r = np.array(model.render_slice(field, "f64", lattice))
    return r.reshape(n_lattice, n_lattice, n_lattice)


def get_asymmetries():
    ret = {}
    for field in ["rho", "rhoetot"]:
        r = render(field)
        for name, sym in symmetries.items():
            ret[f"{field} {name}"] = np.max(np.abs(r - sym(r))) / np.max(np.abs(r))
    return ret


def get_block_count_per_level():
    dic = ctx.collect_data()
    block_sizes = np.array(dic["cell_max"])[:, 0] - np.array(dic["cell_min"])[:, 0]
    # level 0 is the base grid, negative levels are base blocks merged together
    levels = np.rint(np.log2(sz / block_sizes)).astype(int)
    return {int(lev): int(np.sum(levels == lev)) for lev in np.unique(levels)}


# %%
# Run
# ^^^

t_outputs = np.linspace(0.0, 0.3, 7)[1:]

tol = 1e-10
asym_history = []
level_history = [get_block_count_per_level()]
asym_history.append(get_asymmetries())

for t in t_outputs:
    model.evolve_until(t)
    level_history.append(get_block_count_per_level())
    asym_history.append(get_asymmetries())

# %%
# Results
# ^^^^^^^

if shamrock.sys.world_rank() == 0:
    for i, (lev, asym) in enumerate(zip(level_history, asym_history)):
        t = 0.0 if i == 0 else t_outputs[i - 1]
        print(f"t = {t:.3f}, blocks per level: {lev}")
        print("    max asymmetry: " + ", ".join(f"{k} {v:.2e}" for k, v in asym.items()))

# the test only makes sense if both refinement and derefinement have happened
n_fine = [lev.get(1, 0) for lev in level_history]
refined = max(n_fine) > 0
derefined = any(b < a for a, b in itertools.pairwise(n_fine))
print(f"number of level 1 blocks: {n_fine}")

errors = []
if not refined:
    errors.append("no block has been refined, the test setup is not exercising the AMR")
if not derefined:
    errors.append("no block has been merged, the test setup is not exercising the derefinement")

for i, asym in enumerate(asym_history):
    for k, v in asym.items():
        if not v < tol:
            t = 0.0 if i == 0 else t_outputs[i - 1]
            errors.append(f"t = {t:.3f}: {k} asymmetry {v:.2e} >= {tol:.0e}")

if errors:
    raise RuntimeError("AMR mirror symmetry test failed:\n" + "\n".join(errors))
