"""
Binary orbit functions
=======================================

This example shows how to use binary orbit functions
"""

import numpy as np

import shamrock

# %%
# Define the unit system
si = shamrock.UnitSystem()
sicte = shamrock.Constants(si)
codeu = shamrock.UnitSystem(
    unit_time=sicte.year(),
    unit_length=sicte.au(),
    unit_mass=sicte.sol_mass(),
)
ucte = shamrock.Constants(codeu)
G = ucte.G()


# %%
# Utility to plot the resulting orbits
def plot_orbits(m1, m2, a, e, roll, pitch, yaw):
    """
    Plot the orbit of a binary system by varying the true anomaly
    """
    import matplotlib.pyplot as plt

    ax = plt.figure().add_subplot(projection="3d")

    x1, x2, y1, y2, z1, z2 = [], [], [], [], [], []

    vx1, vx2, vy1, vy2, vz1, vz2 = [], [], [], [], [], []

    max_nu = np.pi
    min_nu = -np.pi
    if e >= 1:  # if parabolic do not exceed pi
        max_nu = 0.75 * np.pi
        min_nu = -0.75 * np.pi

    for nu in np.linspace(min_nu, max_nu, 200, endpoint=False):
        # To see the orbit start
        if nu > 1.8 * np.pi:
            break

        _x1, _x2, _v1, _v2 = shamrock.phys.get_binary_rotated(
            m1=m1, m2=m2, a=a, e=e, nu=nu, G=G, roll=roll, pitch=pitch, yaw=yaw
        )

        # print(_x1, _x2, _v1, _v2)
        x1.append(_x1[0])
        x2.append(_x2[0])
        y1.append(_x1[1])
        y2.append(_x2[1])
        z1.append(_x1[2])
        z2.append(_x2[2])

        vx1.append(_v1[0])
        vx2.append(_v2[0])
        vy1.append(_v1[1])
        vy2.append(_v2[1])
        vz1.append(_v1[2])
        vz2.append(_v2[2])

    ax.plot(x1, y1, z1, "-o", markevery=[0, 50, 100, 150])
    ax.plot(x2, y2, z2, "-o", markevery=[0, 50, 100, 150])

    for i in range(0, len(x1), 50):
        vnorm = np.sqrt(vx1[i] ** 2 + vy1[i] ** 2 + vz1[i] ** 2) * 0.03
        ax.quiver(x1[i], y1[i], z1[i], vx1[i], vy1[i], vz1[i], color="r", length=vnorm)
    for i in range(0, len(x2), 50):
        vnorm = np.sqrt(vx1[i] ** 2 + vy1[i] ** 2 + vz1[i] ** 2) * 0.03
        ax.quiver(x2[i], y2[i], z2[i], vx2[i], vy2[i], vz2[i], color="b", length=vnorm)

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()


# %%
# Orbit 1
plot_orbits(0.7, 0.3, 1.0, 0.3, 0.0, 0.0, 0.0)

# %%
# Orbit 2
plot_orbits(0.5, 0.5, 1.0, 0.3, 1.0, 0.0, 0.0)

# %%
# Orbit 3
plot_orbits(0.5, 0.5, 1.0, 0.0, 1.0, 0.0, 0.0)

# %%
# Orbit 4
plot_orbits(0.9, 0.1, 1.0, 0.0, 0.0, 1.0, 0.0)

# %%
# Orbit 5 (hyperbolic)
plot_orbits(0.9, 0.1, 1.0, 1.2, 0.0, 1.0, 0.0)
