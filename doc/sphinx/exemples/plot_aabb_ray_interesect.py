"""
Testing Ray AABB intersection
=============================

This example shows how to use Ray AABB intersection in matplotlib
"""

# %%
# This is a section header
# ------------------------
# This is the first section!
# The `#%%` signifies to Sphinx-Gallery that this text should be rendered as
# reST and if using one of the above IDE/plugin's, also signifies the start of a
# 'code block'.

import matplotlib.pyplot as plt
import shamrock


# draw cube
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
'''
points = [[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1], [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]]

faces = [[points[0], points[1], points[3], points[2]],
         [points[4], points[5], points[7], points[6]],
         [points[0], points[1], points[5], points[4]],
         [points[2], points[3], points[7], points[6]],
         [points[0], points[2], points[6], points[4]],
         [points[1], points[3], points[7], points[5]]]

edges = [[points[0], points[1]], [points[0], points[2]], [points[0], points[4]], [points[1], points[3]], [points[1], points[5]], [points[2], points[3]], [points[2], points[6]], [points[3], points[7]], [points[4], points[5]], [points[4], points[6]], [points[5], points[7]], [points[6], points[7]]]

collection = Poly3DCollection(faces, alpha=0.5)
ax.add_collection3d(collection)

edge_collection = Line3DCollection(edges, color='k')
ax.add_collection3d(edge_collection)
'''


def draw_aabb(ax,aabb, color, alpha):
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
        aabb.upper(), ]

    faces = [[points[0], points[1], points[3], points[2]],
            [points[4], points[5], points[7], points[6]],
            [points[0], points[1], points[5], points[4]],
            [points[2], points[3], points[7], points[6]],
            [points[0], points[2], points[6], points[4]],
            [points[1], points[3], points[7], points[5]]]

    edges = [[points[0], points[1]], [points[0], points[2]], [points[0], points[4]], [points[1], points[3]], [points[1], points[5]], [points[2], points[3]], [points[2], points[6]], [points[3], points[7]], [points[4], points[5]], [points[4], points[6]], [points[5], points[7]], [points[6], points[7]]]

    collection = Poly3DCollection(faces, alpha=alpha, color=color)
    ax.add_collection3d(collection)

    edge_collection = Line3DCollection(edges, color='k', alpha=alpha)
    ax.add_collection3d(edge_collection)

def draw_ray(ax,ray, color):

    xmin, ymin, zmin = ray.origin()
    nx, ny, nz = ray.direction()
    inx, iny, inz = ray.inv_direction()
    print(ray.direction(),ray.inv_direction())
    print(nx*inx,ny*iny,nz*inz)

    ax.plot3D([xmin, nx+xmin], [ymin, ny+ymin], [zmin, nz+zmin], c=color)



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

aabb1 = shamrock.AABB_f64_3((-1.,-1.,-1.),(2.,2.,2.))
aabb2 = shamrock.AABB_f64_3((-2.,-2.,-2.),(1.,1.,1.))

draw_aabb(ax,aabb1, 'b',0.1)
draw_aabb(ax,aabb2, 'r',0.1)
draw_aabb(ax,aabb1.get_intersect(aabb2), 'g',0.5)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-2, 2)

plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

aabb1 = shamrock.AABB_f64_3((-1.,-1.,-1.),(1.,1.,1.))

draw_aabb(ax,aabb1, 'b',0.1)

def add_ray(ray):
    cd = aabb1.intersect_ray(ray)

    print(cd)

    if cd:
        draw_ray(ax,ray, 'g')
    else:
        draw_ray(ax,ray, 'r')

add_ray(shamrock.Ray_f64_3((-2.,-2.,-2.),(2.0,2.,2.)))
add_ray(shamrock.Ray_f64_3((-2.,-2.,-2.),(1.0,2.,2.)))
add_ray(shamrock.Ray_f64_3((-2.,-2.,-2.),(0.7,2.,2.)))
add_ray(shamrock.Ray_f64_3((-2.,-2.,-2.),(0.6,2.,2.)))
add_ray(shamrock.Ray_f64_3((-2.,-2.,-2.),(0.5,2.,2.)))
add_ray(shamrock.Ray_f64_3((-2.,-2.,-2.),(0.1,2.,2.)))
add_ray(shamrock.Ray_f64_3((-2.,-2.,-2.),(0.0,2.,2.)))
add_ray(shamrock.Ray_f64_3((-2.,-2.,-2.),(0.0,0.0,2.)))

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-2, 2)

plt.show()


#draw_ray(0, 0.5, 0, 0.5, 0, 1, 'g')
