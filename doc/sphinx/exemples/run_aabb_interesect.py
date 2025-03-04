"""
Testing AABB intersection routine
=================================

This example shows how to use AABB intersection and plot it in matplotlib
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


# %%

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

aabb1 = shamrock.math.AABB_f64_3((-1.0, -1.0, -1.0), (2.0, 2.0, 2.0))
aabb2 = shamrock.math.AABB_f64_3((-2.0, -2.0, -2.0), (1.0, 1.0, 1.0))

draw_aabb(ax, aabb1, "b", 0.1)
draw_aabb(ax, aabb2, "r", 0.1)
draw_aabb(ax, aabb1.get_intersect(aabb2), "g", 0.5)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-2, 2)

plt.show()
