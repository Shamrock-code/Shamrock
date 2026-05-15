"""
Testing Ray AABB intersection
=============================

This example shows how to use Ray AABB intersection in matplotlib
"""

# %%

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

import shamrock

# %%


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
    xmin, ymin, zmin = aabb.lower
    xmax, ymax, zmax = aabb.upper

    points = [
        aabb.lower,
        (aabb.lower[0], aabb.lower[1], aabb.upper[2]),
        (aabb.lower[0], aabb.upper[1], aabb.lower[2]),
        (aabb.lower[0], aabb.upper[1], aabb.upper[2]),
        (aabb.upper[0], aabb.lower[1], aabb.lower[2]),
        (aabb.upper[0], aabb.lower[1], aabb.upper[2]),
        (aabb.upper[0], aabb.upper[1], aabb.lower[2]),
        aabb.upper,
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


# %%


def draw_ray(ax, ray, color):
    """
    Draw a 3D Ray in matplotlib

    Parameters
    ----------
    ax : matplotlib.Axes3D
        The axis to draw the Ray on
    ray : shamrock.Ray_f64_3
        The Ray to draw
    color : str
        The color of the Ray
    """
    xmin, ymin, zmin = ray.origin()
    nx, ny, nz = ray.direction()
    inx, iny, inz = ray.inv_direction()
    print(ray.direction(), ray.inv_direction())
    print(nx * inx, ny * iny, nz * inz)

    ax.plot3D([xmin, nx + xmin], [ymin, ny + ymin], [zmin, nz + zmin], c=color)


# %%

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

aabb1 = shamrock.math.AABB_f64_3((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))

draw_aabb(ax, aabb1, "b", 0.1)


def add_ray(ray):
    cd = aabb1.intersect_ray(ray)

    print(cd)

    if cd:
        draw_ray(ax, ray, "g")
    else:
        draw_ray(ax, ray, "r")


add_ray(shamrock.math.Ray_f64_3((-2.0, -2.0, -2.0), (2.0, 2.0, 2.0)))
add_ray(shamrock.math.Ray_f64_3((-2.0, -2.0, -2.0), (1.0, 2.0, 2.0)))
add_ray(shamrock.math.Ray_f64_3((-2.0, -2.0, -2.0), (0.7, 2.0, 2.0)))
add_ray(shamrock.math.Ray_f64_3((-2.0, -2.0, -2.0), (0.6, 2.0, 2.0)))
add_ray(shamrock.math.Ray_f64_3((-2.0, -2.0, -2.0), (0.5, 2.0, 2.0)))
add_ray(shamrock.math.Ray_f64_3((-2.0, -2.0, -2.0), (0.1, 2.0, 2.0)))
add_ray(shamrock.math.Ray_f64_3((-2.0, -2.0, -2.0), (0.0, 2.0, 2.0)))
add_ray(shamrock.math.Ray_f64_3((-2.0, -2.0, -2.0), (0.0, 0.0, 2.0)))

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-2, 2)

plt.show()
