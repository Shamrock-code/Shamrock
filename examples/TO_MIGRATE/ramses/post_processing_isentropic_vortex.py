import os
import glob
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize



# -------------------------------------------------------
# Read all files
# -------------------------------------------------------

# files = sorted(glob.glob("amr_athena_kelvin_helmhotz*.vtk"))
files = sorted(glob.glob("amr_isentropic_vortex_period_*.vtk"))

if len(files) == 0:
    raise RuntimeError("No vtk files found.")

# -------------------------------------------------------
# Compute global limits
# -------------------------------------------------------

rho_min = np.inf
rho_max = -np.inf
level_max = 0

for file in files:

    mesh = pv.read(file)

    mesh = mesh.compute_cell_sizes()

    dx = np.cbrt(mesh["Volume"])
    dx0 = dx.max()

    level = np.rint(np.log2(dx0 / dx)).astype(int)

    rho_min = min(rho_min, mesh["rho"].min())
    rho_max = max(rho_max, mesh["rho"].max())

    level_max = max(level_max, level.max())

print("Global rho :", rho_min, rho_max)
print("Maximum level :", level_max)



# -------------------------------------------------------
# Configuration
# -------------------------------------------------------

WINDOW_SIZE = (1000, 1000)
AMR_CMAP = "Paired"
RHO_CMAP = "gist_rainbow"


CAMERA_HEIGHT = 1000.0

EDGE_COLOR = "black"
EDGE_WIDTH = 0.25

ZOOM = 1.0


# -------------------------------------------------------
# Fixed camera
# -------------------------------------------------------

mesh0 = pv.read(files[0])

xmin, xmax, ymin, ymax, zmin, zmax = mesh0.bounds

xc = 0.5 * (xmin + xmax)
yc = 0.5 * (ymin + ymax)
zc = 0.5 * (zmin + zmax)

parallel_scale = max(
    xmax - xmin,
    ymax - ymin
) / 2

# -------------------------------------------------------
# Rendering function to plot separately AMR level and 
# -------------------------------------------------------
def render_plot(
        sl,
):
    
    
    plotter = pv.Plotter(
        shape=(1,1),
        off_screen=True,
        window_size=WINDOW_SIZE,
    )

    plotter.set_background("white")

    plotter.subplot(0, 0)

    plotter.add_mesh(
    sl,
    scalars="rho",
    preference="cell",
    cmap=RHO_CMAP,
    clim=(rho_min, rho_max),
    show_edges=True,
    edge_color=EDGE_COLOR,
    line_width=EDGE_WIDTH,
    interpolate_before_map=False,
    lighting=False,
    show_scalar_bar=False,
    )

    plotter.enable_parallel_projection()

    plotter.camera.position = (
        xc,
        yc,
        CAMERA_HEIGHT,
    )

    plotter.camera.focal_point = (
        xc,
        yc,
        zc,
    )

    plotter.camera.up = (0, 1, 0)

    plotter.camera.parallel_scale = parallel_scale / ZOOM
    plotter.disable_anti_aliasing()

    image = plotter.screenshot(
            return_img=True,
                scale=2,
            )

    plotter.close()

    return image


#--------plot figure for each dump --------------

for file in files:

    print(file)

    mesh = pv.read(file)

    mesh = mesh.compute_cell_sizes()

    dx = np.cbrt(mesh["Volume"])
    dx0 = dx.max()

    mesh["level"] = np.rint(np.log2(dx0 / dx)).astype(int)

    zmid = 0.5 * (mesh.bounds[4] + mesh.bounds[5])



    sl = mesh.slice(
    normal="z",
    origin=(xc, yc, zmid),
    generate_triangles=False,
    )

    basename = os.path.splitext(os.path.basename(file))[0]

    image = render_plot(
        sl
    )


    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_axes([0.06, 0.05, 0.82, 0.90])

    ax.imshow(image)
    ax.set_axis_off()

    norm_rho = mpl.colors.Normalize(
    vmin=rho_min,
    vmax=rho_max,
    )

    sm_rho = mpl.cm.ScalarMappable(
        norm=norm_rho,
        cmap=RHO_CMAP,
    )

    sm_rho.set_array([])
    # Density colorbar (left)
    cax1 = fig.add_axes([0.90, 0.1, 0.018, 0.80])

    cb1 = fig.colorbar(
        sm_rho,
        cax=cax1,
         shrink=0.99,
        pad=0.03,
    )

    cb1.set_label(r"$\rho$", fontsize=16)
    cb1.ax.tick_params(labelsize=12)

     #-------- Save figure
    plt.savefig(
        basename + "amr_isentropic_density.png",
        dpi=500,
        bbox_inches="tight",
    )

    plt.close(fig)