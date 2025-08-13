"""
Ghost layer generation using paving functions
=============================================

This example showcase how to use the paving functions to generate ghost layers.

The complex thing is that we want to intersect the current data with the ghost layer,
but we do not want to modify the original buffer. As a result we perform the intersection
in a space transformed by the paving function, and then map the result back to the original space.

Formally speaking for a paving function f:

Ghost layer = f(patch) V box
            = f( f_inv(f(patch) V box) )
            = f( patch V f_inv(box) ),

where V denote a ghost layer intersection.
"""

# sphinx_gallery_multi_image = "single"

import matplotlib.pyplot as plt
import numpy as np

import shamrock

# %%
# Set box size

box_size_x = 2.0
box_size_y = 2.0

# %%
# Particle set

color_set = [
    "blue",
    "red",
    "green",
    "black",
    "orange",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
    "cyan",
    "magenta",
    "yellow",
    "teal",
    "navy",
    "gold",
    "lime",
    "indigo",
    "maroon",
    "turquoise",
]

x_set = np.linspace(0.2, box_size_x - 0.2, 4)
y_set = np.linspace(0.2, box_size_y - 0.2, 4)

parts = []
i = 0
for x in x_set:
    for y in y_set:
        parts.append({"x": x, "y": y, "color": color_set[i % len(color_set)]})
        i += 1


# %%
# Utility to plot the paving function

margin = 0.1


def add_rect(x, y, w, h):
    plt.gca().add_patch(
        plt.Rectangle(
            (x + margin, y + margin),
            w - 2 * margin,
            h - 2 * margin,
            alpha=0.5,
            fill=True,
            facecolor="grey",
            linewidth=2,
        )
    )


def add_rect_aabb(aabb):
    add_rect(
        aabb.lower[0], aabb.lower[1], aabb.upper[0] - aabb.lower[0], aabb.upper[1] - aabb.lower[1]
    )


def ghost_intersect(part, box_to_intersect):
    """
    dummy ghost layer intersection to showcase how it works
    """
    x, y, z = part["x"], part["y"], 0.0

    part_aabb = shamrock.math.AABB_f64_3((x - 0.3, y - 0.3, 0.0), (x + 0.3, y + 0.3, 0.0))

    intersect = box_to_intersect.get_intersect(part_aabb)

    _x, _y, _z = intersect.upper
    _z = 1.0  # we want to be sure that the volume is not null but the z is not used
    intersect.upper = (_x, _y, _z)

    return intersect.is_volume_not_null()


def plot_paving_function(pav_func, pav_func_name):

    box_to_intersect = shamrock.math.AABB_f64_3((0.0, 0.0, 0.0), (box_size_x, box_size_y, 0.0))

    def get_indices():
        for i in range(-2, 3):
            for j in range(-3, 4):
                if i == 0 and j == 0:
                    continue
                yield i, j

    plt.figure()

    for i, j in get_indices():

        box_to_intersect_inv_mapped = pav_func.f_aabb_inv(box_to_intersect, i, j, 0)

        add_rect_aabb(box_to_intersect_inv_mapped)

    for part in parts:
        x, y, z = (part["x"], part["y"], 0.0)
        plt.scatter(x, y, color=part["color"])

    plt.title(f"Paving function: {pav_func_name}\n1. Inverse map current sim box")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.figure()

    for i, j in get_indices():

        box_to_intersect_inv_mapped = pav_func.f_aabb_inv(box_to_intersect, i, j, 0)

        add_rect_aabb(box_to_intersect_inv_mapped)

        for part in parts:
            x, y, z = (part["x"], part["y"], 0.0)
            if ghost_intersect(part, box_to_intersect_inv_mapped):
                plt.scatter(x, y, color=part["color"])

    plt.title(f"Paving function: {pav_func_name}\n2. Inverse ghost layer with inverse map")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.figure()

    for i, j in get_indices():

        box_to_intersect_inv_mapped = pav_func.f_aabb_inv(box_to_intersect, i, j, 0)
        box_to_intersect_mapped = pav_func.f_aabb(box_to_intersect, i, j, 0)

        add_rect_aabb(box_to_intersect_mapped)

        for part in parts:
            x, y, z = (part["x"], part["y"], 0.0)

            if ghost_intersect(part, box_to_intersect_inv_mapped):
                x, y, z = pav_func.f((x, y, 0.0), i, j, 0)
                plt.scatter(x, y, color=part["color"])

    for part in parts:
        x, y, z = (part["x"], part["y"], 0.0)
        plt.scatter(x, y, color=part["color"])

    plt.title(f"Paving function: {pav_func_name}\n3. Map back the resulting ghost layer")
    plt.xlabel("x")
    plt.ylabel("y")


# %%
# Testing the paving functions
# ----------------------------

# %%
# Periodic paving function

plot_paving_function(
    shamrock.math.paving_function_general_3d(
        (box_size_x, box_size_y, 0.0), (box_size_x / 2.0, box_size_y / 2.0, 0.0), True, True, True
    ),
    "Periodic box",
)


# %%
# Periodic & reflective paving function

plot_paving_function(
    shamrock.math.paving_function_general_3d(
        (box_size_x, box_size_y, 0.0), (box_size_x / 2.0, box_size_y / 2.0, 0.0), False, True, True
    ),
    "reflective in x periodic in y",
)


# %%
# Fully reflective paving function

plot_paving_function(
    shamrock.math.paving_function_general_3d(
        (box_size_x, box_size_y, 0.0), (box_size_x / 2.0, box_size_y / 2.0, 0.0), False, False, True
    ),
    "Fully reflective",
)


# %%
# Periodic & reflective paving function with shear

plot_paving_function(
    shamrock.math.paving_function_general_3d_shear_x(
        (box_size_x, box_size_y, 0.0),
        (box_size_x / 2.0, box_size_y / 2.0, 0.0),
        False,
        True,
        True,
        0.3,
    ),
    "reflective in x periodic in y with shear",
)

plt.show()
