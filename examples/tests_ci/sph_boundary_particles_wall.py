"""
Testing SPH frozen boundary particles
=====================================

CI test for the frozen boundary particles of the SPH solver. A Sod tube is closed by two walls
made of boundary particles (``freeze_id != 0``) at rest, carrying the left and right states. The
test checks that:

- the boundary particles keep their position, velocity and internal energy bit for bit,
- the Sod solution between the walls is not perturbed by the walls.

The run stops before the waves reach the walls: frozen boundary particles only push back on the
gas through their summed density, so a strong compression against them (e.g. a reflected shock)
makes the gas leak into them. They are not a reflective wall.
"""

import numpy as np

import shamrock

# particle tracking is experimental
shamrock.enable_experimental_features()

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")

gamma = 1.4

rho_g = 1
rho_d = 0.125

fact = (rho_g / rho_d) ** (1.0 / 3.0)

P_g = 1
P_d = 0.1

u_g = P_g / ((gamma - 1) * rho_g)
u_d = P_d / ((gamma - 1) * rho_d)

resol = 64

# the gas lives in |x| < x_wall, the walls fill x_wall < |x| < xs (about 1)
x_wall = 0.75

ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M6")

cfg = model.gen_default_config()
cfg.set_artif_viscosity_VaryingCD10(
    alpha_min=0.0, alpha_max=1, sigma_decay=0.1, alpha_u=1, beta_AV=2
)
# periodic in y and z, in x the periodic seam lies between the two walls
cfg.set_boundary_periodic()
cfg.set_boundary_particles(True)
cfg.set_particle_tracking(True)
cfg.set_eos_adiabatic(gamma)
cfg.print_status()
model.set_solver_config(cfg)

model.init_scheduler(int(1e8), 1)

(xs, ys, zs) = model.get_box_dim_fcc_3d(1, resol, 12, 12)
dr = 1 / xs
(xs, ys, zs) = model.get_box_dim_fcc_3d(dr, resol, 12, 12)

model.resize_simulation_box((-xs, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))

setup = model.get_setup()
gen1 = setup.make_generator_lattice_hcp(dr, (-xs, -ys / 2, -zs / 2), (0, ys / 2, zs / 2))
gen2 = setup.make_generator_lattice_hcp(dr * fact, (0, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))
comb = setup.make_combiner_add(gen1, gen2)
setup.apply_setup(comb)

model.set_value_in_a_box("uint", "f64", u_g, (-xs, -ys / 2, -zs / 2), (0, ys / 2, zs / 2))
model.set_value_in_a_box("uint", "f64", u_d, (0, -ys / 2, -zs / 2), (xs, ys / 2, zs / 2))

# the walls: frozen boundary particles at rest
model.set_field_value_lambda_u32("freeze_id", lambda r: 1 if abs(r[0]) > x_wall else 0)

vol_b = xs * ys * zs
totmass = (rho_d * vol_b) + (rho_g * vol_b)
pmass = model.total_mass_to_part_mass(totmass)
model.set_particle_mass(pmass)

model.set_cfl_cour(0.1)
model.set_cfl_force(0.1)


def get_walls(data):
    """Return the wall particles sorted by particle id, so that two states can be compared"""
    is_wall = data["freeze_id"] != 0
    order = np.argsort(data["part_id"][is_wall])
    return {
        "part_id": data["part_id"][is_wall][order],
        "xyz": data["xyz"][is_wall][order],
        "vxyz": data["vxyz"][is_wall][order],
        "uint": data["uint"][is_wall][order],
    }


data_init = ctx.collect_data()
n_walls = int(np.sum(data_init["freeze_id"] != 0))
n_gas = len(data_init["xyz"]) - n_walls
print(f"wall particles: {n_walls}, gas particles: {n_gas}")
assert n_walls > 0 and n_gas > 0

# the walls must be thicker than the kernel support
assert xs - x_wall > 3 * 1.2 * dr * fact

walls_setup = get_walls(data_init)

# the first step wraps every position in the periodic box, which may change the last bit of the
model.timestep()
walls_init = get_walls(ctx.collect_data())

# boundary particles at rest stay at rest
if not np.all(walls_init["vxyz"] == 0):
    exit("the wall particles have gained some velocity")
if not np.array_equal(walls_init["part_id"], walls_setup["part_id"]):
    exit("the set of wall particles changed during the first step")
if not np.array_equal(walls_init["uint"], walls_setup["uint"]):
    exit("the wall particles internal energy changed during the first step")
if not np.allclose(walls_init["xyz"], walls_setup["xyz"], rtol=0, atol=1e-12):
    exit("the wall particles moved during the first step")

# from now on the walls must stay bit for bit identical to their state after the first step

err_log = ""


def check_walls_unchanged(label):
    """Append to err_log if the wall particles differ from their state after the first step"""
    global err_log
    walls = get_walls(ctx.collect_data())
    for key in ["part_id", "xyz", "vxyz", "uint"]:
        if walls[key].shape != walls_init[key].shape or not np.array_equal(
            walls[key], walls_init[key]
        ):
            err_log += f"{label}: the wall particles field {key} has changed\n"


#################
### Sod solution between the walls (waves have not reached the walls yet)
#################
t_sod = 0.245
model.evolve_until(t_sod)

check_walls_unchanged(f"t = {t_sod}")

sod = shamrock.phys.SodTube(gamma=gamma, rho_1=1, P_1=1, rho_5=0.125, P_5=0.1)
sodanalysis = model.make_analysis_sodtube(sod, (1, 0, 0), t_sod, 0.0, -0.5, 0.5)
rho, v, P = sodanalysis.compute_L2_dist()
vx, vy, vz = v

print("current errors :")
print(f"err_rho = {rho}")
print(f"err_vx = {vx}")
print(f"err_vy = {vy}")
print(f"err_vz = {vz}")
print(f"err_P = {P}")

# loose bounds (twice the errors obtained when the test was written): the walls must not degrade
# the solution, the exact values are a regression check of the SPH solver itself, already covered
# by sod_tube_sph.py
max_err = {"rho": 7e-4, "vx": 6e-3, "P": 7e-4}
for name, val in [("rho", rho), ("vx", vx), ("P", P)]:
    if not val < max_err[name]:
        err_log += f"L2 error on {name} = {val} is above {max_err[name]}\n"

if err_log != "":
    exit("Test did not pass : \n" + err_log)
