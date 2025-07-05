import matplotlib.pyplot as plt
import numpy as np

import shamrock

# If we use the shamrock executable to run this script instead of the python interpreter,
# we should not initialize the system as the shamrock executable needs to handle specific MPI logic
if not shamrock.sys.is_initialized():
    shamrock.change_loglevel(1)
    shamrock.sys.init("0:0")


def is_in_sphere(pt):
    x, y, z = pt
    return (x**2 + y**2 + z**2) < 1


####################################################
# Setup parameters
####################################################
dr = 0.05
pmass = 1

C_cour = 0.3
C_force = 0.25

bsize = 4

do_plots = True

####################################################
####################################################
####################################################

ctx = shamrock.Context()
ctx.pdata_layout_new()

model = shamrock.get_Model_SPH(context=ctx, vector_type="f64_3", sph_kernel="M4")

cfg = model.gen_default_config()
# cfg.set_artif_viscosity_VaryingMM97(alpha_min = 0.1,alpha_max = 1,sigma_decay = 0.1, alpha_u = 1, beta_AV = 2)
# cfg.set_artif_viscosity_ConstantDisc(alpha_u=alpha_u, alpha_AV=alpha_AV, beta_AV=beta_AV)
# cfg.set_eos_locally_isothermalLP07(cs0=cs0, q=q, r0=r0)
cfg.set_artif_viscosity_VaryingCD10(
    alpha_min=0.0, alpha_max=1, sigma_decay=0.1, alpha_u=1, beta_AV=2
)
cfg.set_particle_tracking(True)
cfg.set_boundary_periodic()
cfg.set_eos_adiabatic(1.00001)
cfg.print_status()
model.set_solver_config(cfg)

model.init_scheduler(int(300), 50)

bmin = (-bsize, -bsize, -bsize)
bmax = (bsize, bsize, bsize)
model.resize_simulation_box(bmin, bmax)

model.set_particle_mass(pmass)

setup = model.get_setup()
lat = setup.make_generator_lattice_hcp(dr, (-bsize, -bsize, -bsize), (bsize, bsize, bsize))

thesphere = setup.make_modifier_filter(parent=lat, filter=is_in_sphere)

offset_sphere = setup.make_modifier_offset(
    parent=thesphere, offset_position=(3.0, 3.0, 3.0), offset_velocity=(-1.0, -1.0, -1.0)
)

setup.apply_setup(offset_sphere)

model.set_value_in_a_box("uint", "f64", 1, bmin, bmax)

model.set_cfl_cour(C_cour)
model.set_cfl_force(C_force)

model.change_htolerance(1.3)
model.timestep()
model.change_htolerance(1.1)


def draw_aabb(ax, aabb, color, alpha):
    """
    Draw a 3D AABB in matplotlib

    Parameters
    ----------
    ax : matplotlib.Axes3D
        The axis to draw the AABB on
    aabb : shamrock.math.AABB_f64_3
        The AABB to draw
    color : str
        The color of the AABB
    alpha : float
        The transparency of the AABB
    """

    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    xmin, ymin, zmin = aabb.lower()
    xmax, ymax, zmax = aabb.upper()

    points = [
        aabb.lower(),
        (aabb.lower()[0], aabb.lower()[1], aabb.upper()[2]),
        (aabb.lower()[0], aabb.upper()[1], aabb.lower()[2]),
        (aabb.lower()[0], aabb.upper()[1], aabb.upper()[2]),
        (aabb.upper()[0], aabb.lower()[1], aabb.lower()[2]),
        (aabb.upper()[0], aabb.lower()[1], aabb.upper()[2]),
        (aabb.upper()[0], aabb.upper()[1], aabb.lower()[2]),
        aabb.upper(),
    ]

    faces = [
        [points[0], points[1], points[3], points[2]],
        [points[4], points[5], points[7], points[6]],
        [points[0], points[1], points[5], points[4]],
        [points[2], points[3], points[7], points[6]],
        [points[0], points[2], points[6], points[4]],
        [points[1], points[3], points[7], points[5]],
    ]

    edges = [
        [points[0], points[1]],
        [points[0], points[2]],
        [points[0], points[4]],
        [points[1], points[3]],
        [points[1], points[5]],
        [points[2], points[3]],
        [points[2], points[6]],
        [points[3], points[7]],
        [points[4], points[5]],
        [points[4], points[6]],
        [points[5], points[7]],
        [points[6], points[7]],
    ]

    collection = Poly3DCollection(faces, alpha=alpha, color=color)
    ax.add_collection3d(collection)

    edge_collection = Line3DCollection(edges, color="k", alpha=alpha)
    ax.add_collection3d(edge_collection)


def plot_state(iplot):

    patch_list = ctx.get_patch_list_global()

    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlim3d(bmin[0], bmax[0])
    ax.set_ylim3d(bmin[1], bmax[1])
    ax.set_zlim3d(bmin[2], bmax[2])

    ptransf = model.get_patch_transform()

    for p in patch_list:
        draw_aabb(ax, ptransf.to_obj_coord(p), "blue", 0.1)

    parts = ctx.collect_data()

    ids = parts["part_id"]

    for i in range(len(ids)):
        if not i in ids:
            print("missing {} : {}".format(i, ids[i]))

    print("------")
    for i in range(len(ids)):
        print("part {} : {}".format(i, ids[i]))

    exit()

    pos = parts["xyz"]
    X = pos[:, 0]
    Y = pos[:, 1]
    Z = pos[:, 2]

    ax.scatter(X, Y, Z, c="red", s=1)

    plt.savefig("test_{}.png".format(iplot))
    plt.cla()
    plt.clf()

    print(patch_list)


nstop = 100
dt_stop = 0.1

t_stop = [i * dt_stop for i in range(nstop + 1)]

iplot = 0
istop = 0
for ttarg in t_stop:

    model.evolve_until(ttarg)

    if do_plots:
        plot_state(iplot)

    iplot += 1
    istop += 1
