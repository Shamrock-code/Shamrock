"""
Paving functions
============================

This simple example shows how paving functions in Shamrock works
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

parts = [
    {"x": 0.1 * box_size_x, "y": 0.1 * box_size_y, "color": "blue"},
    {"x": 0.7 * box_size_x, "y": 0.1 * box_size_y, "color": "red"},
    {"x": 0.1 * box_size_x, "y": 0.7 * box_size_y, "color": "green"},
    {"x": 0.7 * box_size_x, "y": 0.7 * box_size_y, "color": "black"},
]


# %%
# Utility to plot the paving function


def add_rect(x, y, w, h):
    plt.gca().add_patch(
        plt.Rectangle((x, y), w, h, alpha=0.5, fill=True, facecolor="grey", linewidth=2)
    )


def plot_paving_function(pav_func, pav_func_name, shear_x=0.0):

    plt.figure()
    for i in range(-2, 3):
        for j in range(-3, 4):
            # j = i
            add_rect(
                0.05 + i * box_size_x + shear_x * j,
                0.05 + j * box_size_y,
                box_size_x - 0.1,
                box_size_y - 0.1,
            )

            for part in parts:
                x, y, z = pav_func.f((part["x"], part["y"], 0.0), i, j, 0)

                plt.scatter(x, y, color=part["color"])
    plt.title(f"Paving function: {pav_func_name}")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.figure()
    for i in range(-2, 3):
        for j in range(-3, 4):
            add_rect(
                0.05 + i * box_size_x + shear_x * j,
                0.05 + j * box_size_y,
                box_size_x - 0.1,
                box_size_y - 0.1,
            )

            for part in parts:
                x, y, z = pav_func.f((part["x"], part["y"], 0.0), i, j, 0)

                x_2, y_2, z_2 = pav_func.f_inv((x, y, 0.0), i, j, 0)

                delta_x = x_2 - part["x"]
                delta_y = y_2 - part["y"]

                if abs(delta_x) > 1e-4 or abs(delta_y) > 1e-4:
                    print("error")

                plt.scatter(x_2 + 0.1 * i, y_2 + 0.1 * j, color=part["color"])
    plt.title(f"Paving function inverse: {pav_func_name}")
    plt.xlabel("x")
    plt.ylabel("y")


# %%
# Testing the paving functions
# ------------------------

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
    shear_x=0.3,
)

plt.show()
