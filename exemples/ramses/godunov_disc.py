import os

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

import shamrock

### Parameters

## Grid setup
resolution = 128

## Initial conditions
h_over_r = 0.1
contrast_factor = 10
ref_density = 1
ref_sound_speed = 1
cs_expo = 0.5
point_mass_Gmass = 1
gravity_softening = 0.1
radius = 1
gamma0 = 1.6

center = 0.5


## Initial conditions


def get_rc_soft(rmin, rmax):
    rmin = np.array(rmin) - center
    rmax = np.array(rmax) - center
    x, y, z = (rmin + rmax) / 2
    rc_soft = np.sqrt(x**2 + y**2 + gravity_softening**2)
    return rc_soft


def get_cs(rc_soft):
    cs = ref_sound_speed * (rc_soft / radius) ** -cs_expo
    return cs


def get_omega(rs_soft, rc_soft, cs_expo):

    cs = get_cs(rc_soft)
    if cs_expo == 0.5:
        omega = point_mass_Gmass / (rc_soft * rc_soft * rs_soft)
        omega -= (4.0 - cs_expo) * (cs * cs / rc_soft * rc_soft)
        omega = max(omega, 0)
        omega = np.sqrt(omega)
    else:
        omega = np.sqrt(
            max(
                point_mass_Gmass / pow(rs_soft, 3)
                - (3.0 - cs_expo) * cs * cs / (rc_soft * rc_soft),
                0.0,
            )
        )

    return omega


def rho_map(rmin, rmax):
    """
    Initial density
    ----
    rmin : tuple
        Coordinates of the lower corner of the cell
    rmax : tuple
        Coordinates of the upper corner of the cell
    """
    rc_soft = get_rc_soft(rmin, rmax)
    dens = ref_density * (radius / rc_soft) ** ((5 - 2 * cs_expo) / 2.0)

    return dens


def rhovel_map(rmin, rmax):
    rc_soft = get_rc_soft(rmin, rmax)
    rho = rho_map(rmin, rmax)

    rmin = np.array(rmin) - center
    rmax = np.array(rmax) - center
    x, y, z = (rmin + rmax) / 2
    rs = np.sqrt(x**2 + y**2 + z**2)
    rs_soft = np.sqrt(x**2 + y**2 + z**2 + gravity_softening**2)
    x_soft = x * (rs_soft / rs)
    y_soft = y * (rs_soft / rs)
    omega = get_omega(rs_soft, rc_soft, cs_expo)

    vx = -rho * omega * y_soft
    vy = rho * omega * x_soft
    vz = 0.0

    return (vx * rho, vy * rho, vz * rho)


def rhoe_map(rmin, rmax):
    """
    Initial total energy
    """
    rho = rho_map(rmin, rmax)
    rho_vx, rho_vy, rho_vz = rhovel_map(rmin, rmax)

    rc_soft = get_rc_soft(rmin, rmax)
    cs = get_cs(rc_soft)

    eint = rho * cs * cs / (gamma0 - 1)
    ekin = 0.5 * (rho_vx**2 + rho_vy**2 + rho_vz**2) / rho

    return eint + ekin


def make_cartesian_coords(nx, ny, z_val, min_x, max_x, min_y, max_y):
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


nx, ny = resolution, resolution


positions = make_cartesian_coords(nx, ny, 0.5, 0, 1 - 1e-6, 0, 1 - 1e-6)


def plot_rho_slice_cartesian(metadata, arr_rho_pos, iplot, case_name, dpi=200):
    ext = metadata["extent"]

    my_cmap = matplotlib.colormaps["gist_heat"].copy()  # copy the default cmap
    my_cmap.set_bad(color="black")

    arr_rho_pos = np.array(arr_rho_pos).reshape(nx, ny)

    plt.figure(dpi=dpi)
    res = plt.imshow(
        arr_rho_pos,
        cmap=my_cmap,
        origin="lower",
        extent=ext,
        aspect="auto",
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"t = {metadata['time']:0.3f} [seconds]")
    cbar = plt.colorbar(res, extend="both")
    cbar.set_label(r"$\rho$ [code unit]")
    plt.savefig(os.path.join(".", f"rho_{case_name}_{iplot:04d}.png"))
    plt.close()


def plot(t, iplot):
    metadata = {"extent": [0, 1, 0, 1], "time": t}
    arr_rho_pos = model.render_slice("rho", "f64", positions)
    plot_rho_slice_cartesian(metadata, arr_rho_pos, iplot, "disk")


### Initialisation

ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_Ramses(context=ctx, vector_type="f64_3", grid_repr="i64_3")


cell_size = 2
base = resolution

cfg = model.gen_default_config()

scale_fact = 1 / (cell_size * base)
cfg.set_scale_factor(scale_fact)
cfg.set_eos_gamma(1.4)  # Can we have something which is NOT an EOS?

cfg.set_riemann_solver_hll()
cfg.set_slope_lim_minmod()
cfg.set_face_time_interpolation(True)

model.set_solver_config(cfg)

model.init_scheduler(int(1e7), 1)  # (crit_split - patches, crit_merge - patches)
model.make_base_grid(
    (0, 0, 0), (cell_size, cell_size, cell_size), (base, base, base)
)  # What is this doing?


model.set_field_value_lambda_f64("rho", rho_map)
model.set_field_value_lambda_f64("rhoetot", rhoe_map)
model.set_field_value_lambda_f64_3("rhovel", rhovel_map)

dt = 0.01
for i in range(3):
    model.dump_vtk("disk" + str(i) + ".vtk")
    plot(dt * i, i)
    model.evolve_until(dt)

plot(dt * (i + 1), i)
