import os

import matplotlib.pyplot as plt
import numpy as np

import shamrock

ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_Ramses(context=ctx, vector_type="f64_3", grid_repr="i64_3")


multx = 1
multy = 1
multz = 1
max_amr_lev = 3
cell_size = 2 << max_amr_lev  # refinement is limited to cell_size = 2
base = 16

cfg = model.gen_default_config()
scale_fact = 1 / (cell_size * base * multx)
cfg.set_scale_factor(scale_fact)

center = (0.5 * base * scale_fact, 0.5 * base * scale_fact, 0.5 * base * scale_fact)
Rstart = 1.0 / (2.0 * base) + 1e-4
gamma = 5.0 / 3.0

cfg.set_eos_gamma(gamma)
cfg.set_Csafe(0.3)
cfg.set_boundary_condition("x", "periodic")
cfg.set_boundary_condition("y", "periodic")
cfg.set_boundary_condition("z", "periodic")
cfg.set_riemann_solver_hllc()
cfg.set_slope_lim_minmod()
cfg.set_face_time_interpolation(True)

err_min = 0.30
err_max = 0.10

cfg.set_amr_mode_pseudo_gradient_based(error_min=err_min, error_max=err_max)

model.set_solver_config(cfg)


model.init_scheduler(int(1e7), 1)
model.make_base_grid(
    (0, 0, 0), (cell_size, cell_size, cell_size), (base * multx, base * multy, base * multz)
)


def rho_map(rmin, rmax):
    return 1.0


def rhoe_map(rmin, rmax):
    x_min, y_min, z_min = rmin
    x_max, y_max, z_max = rmax
    x = (x_min + x_max) * 0.5 - 0.5
    y = (y_min + y_max) * 0.5 - 0.5
    z = (z_min + z_max) * 0.5 - 0.5
    r = np.sqrt(x * x + y * y + z * z)
    if r < Rstart:
        return 10.0 / (gamma - 1.0)
    else:
        return 1e-5 / (gamma - 1.0)


def rhovel_map(rmin, rmax):
    return (0.0, 0.0, 0.0)


model.set_field_value_lambda_f64("rho", rho_map)
model.set_field_value_lambda_f64("rhoetot", rhoe_map)
model.set_field_value_lambda_f64_3("rhovel", rhovel_map)

tmax = 0.2


dt = 0
t = 0
freq = 1
dX0 = []
for i in range(10000):
    next_dt = model.evolve_once_override_time(t, dt)

    t += dt
    dt = next_dt

    if i % freq == 0:
        model.dump_vtk(f"test{i:04d}_ref_new.vtk")

    if tmax < t + next_dt:
        dt = tmax - t
    if t == tmax:
        model.dump_vtk(f"test{i:04d}_ref_new.vtk")
        break
