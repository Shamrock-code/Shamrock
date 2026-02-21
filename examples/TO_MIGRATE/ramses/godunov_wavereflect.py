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

sz = 1 << 1
base = 32

cfg = model.gen_default_config()
scale_fact = 1 / (sz * base * multx)
cfg.set_scale_factor(scale_fact)
cfg.set_boundary_condition("x", "periodic")
cfg.set_boundary_condition("y", "periodic")
cfg.set_boundary_condition("z", "periodic")
cfg.set_eos_gamma(5.0 / 3.0)
model.set_solver_config(cfg)

name = "test_periodic"


model.init_scheduler(int(1e7), 1)
model.make_base_grid((0, 0, 0), (sz, sz, sz), (base * multx, base * multy, base * multz))

gamma = 5.0 / 3.0

u_cs1 = 1 / (gamma * (gamma - 1))

kx, ky, kz = 4 * np.pi, 0, 0
delta_rho = 0
delta_v = 1e-4


def rho_map(rmin, rmax):
    x, y, z = rmin

    return 1.0
    # return 1.0 + delta_rho * np.cos(kx * x + ky * y + kz * z)


def rhoetot_map(rmin, rmax):
    rho = rho_map(rmin, rmax)

    x, y, z = rmin
    rsq = (x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2
    # return x
    # return (u_cs1 + u_cs1 * delta_rho * np.cos(kx * x + ky * y + kz * z)) * rho
    # return u_cs1 + 0.001 * np.exp(-rsq / 0.01)
    return u_cs1


def rhovel_map(rmin, rmax):
    rho = rho_map(rmin, rmax)

    x, y, z = rmin
    rsq = (x - 0.2) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2
    return (0 + delta_v * np.exp(-rsq / 0.01) * rho, 0, 0)
    # return (0, 0, 0)  # eturn (0 + delta_v * np.cos(kx * x + ky * y + kz * z) * rho, 0, 0)


model.set_field_value_lambda_f64("rho", rho_map)
model.set_field_value_lambda_f64("rhoetot", rhoetot_map)
model.set_field_value_lambda_f64_3("rhovel", rhovel_map)

# model.evolve_once(0,0.1)
tmax = 0.127 * 1
all_t = np.linspace(0, tmax, 10)

for i, t in enumerate(all_t):
    model.dump_vtk(name + "_" + str(i) + ".vtk")
    model.evolve_until(t)


def make_cartesian_coords(nx, ny, z_val, min_x, max_x, min_y, max_y):
    """
    Generate a list of positions in cylindrical coordinates (r, theta)
    spanning [0, ext*2] x [-pi, pi] for use with the rendering module.

    Returns:
        list: List of [x, y, z] coordinate lists
    """

    # Create the cylindrical coordinate grid
    x_vals = np.linspace(min_x, max_x, nx)
    y_vals = np.linspace(min_y, max_y, ny)

    # Create meshgrid
    x_grid, y_grid = np.meshgrid(x_vals, y_vals)

    # Convert to Cartesian coordinates (z = 0 for a disc in the xy-plane)
    z_grid = z_val * np.ones_like(x_grid)

    # Flatten and stack to create list of positions
    positions = np.column_stack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()])

    return [tuple(pos) for pos in positions]


nx, ny = 512, 512
positions = make_cartesian_coords(nx, ny, 0.5, 0, 1 - 1e-6, 0, 1 - 1e-6)
arr_rho_pos = model.render_slice("rho", "f64", positions)

import matplotlib


def plot_rho_slice_cylindrical(metadata, arr_rho_pos):
    ext = metadata["extent"]

    my_cmap = matplotlib.colormaps["gist_heat"].copy()  # copy the default cmap
    my_cmap.set_bad(color="black")

    arr_rho_pos = np.array(arr_rho_pos).reshape(nx, ny)

    ampl = 3e-5

    res = plt.imshow(
        arr_rho_pos,
        cmap=my_cmap,
        origin="lower",
        extent=ext,
        vmin=1 - ampl,
        vmax=1 + ampl,
        aspect="auto",
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"t = {metadata['time']:0.3f} [seconds]")
    cbar = plt.colorbar(res, extend="both")
    cbar.set_label(r"$\rho$ [code unit]")


dpi = 200
metadata = {"extent": [0, 1, 0, 1], "time": tmax}

plt.figure(dpi=dpi)
plot_rho_slice_cylindrical(metadata, arr_rho_pos)

plt.show()


def convert_to_cell_coords(dic):
    cmin = dic["cell_min"]
    cmax = dic["cell_max"]

    xmin = []
    ymin = []
    zmin = []
    xmax = []
    ymax = []
    zmax = []

    for i in range(len(cmin)):
        m, M = cmin[i], cmax[i]

        mx, my, mz = m
        Mx, My, Mz = M

        for j in range(8):
            a, b = model.get_cell_coords(((mx, my, mz), (Mx, My, Mz)), j)

            x, y, z = a
            xmin.append(x)
            ymin.append(y)
            zmin.append(z)

            x, y, z = b
            xmax.append(x)
            ymax.append(y)
            zmax.append(z)

    dic["xmin"] = np.array(xmin)
    dic["ymin"] = np.array(ymin)
    dic["zmin"] = np.array(zmin)
    dic["xmax"] = np.array(xmax)
    dic["ymax"] = np.array(ymax)
    dic["zmax"] = np.array(zmax)

    return dic


dic = convert_to_cell_coords(ctx.collect_data())


X = []
rho = []
velx = []
rhoe = []

for i in range(len(dic["xmin"])):
    X.append(dic["xmin"][i])
    rho.append(dic["rho"][i])
    velx.append(dic["rhovel"][i][0])
    rhoe.append(dic["rhoetot"][i])


fig, axs = plt.subplots(3, 1, sharex=True)
axs[0].plot(X, rho, ".", label="rho")
axs[1].plot(X, velx, ".", label="vx")
axs[2].plot(X, rhoe, ".", label="rhoe")


plt.show()
