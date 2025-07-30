"""
Shamrock 3D Gaussian generator
=======================================

This example shows how to use the mock gaussian function
"""

# %%

import matplotlib.pyplot as plt  # plots
import numpy as np  # sqrt & arctan2

import shamrock

# %%
# Pseudo random number generator seed

eng = shamrock.algs.gen_seed(111)

# %%
# Generate positions

list_pos = []
for i in range(1000000):
    list_pos.append(shamrock.algs.mock_gaussian_f64_3(eng))


# %%
# Compute r and theta
r_val = []
for x, y, z in list_pos:
    r = np.sqrt(x**2 + y**2 + z**2)
    r_val.append(r)

theta_val = []
for x, y, z in list_pos:
    theta = np.arctan2(y, x)
    theta_val.append(theta)


# %%
# Radial distribution


hist_r, bins_r = np.histogram(r_val, bins=1000, density=True)
r = np.linspace(bins_r[0], bins_r[-1], 1000)

maxwell_b = (4 * np.pi * r * r) * np.exp(-(r**2) / 2) / (np.sqrt(2 * np.pi)) ** 3

plt.figure()
plt.plot(r, maxwell_b, "r--", lw=2)
plt.bar(bins_r[:-1], hist_r, np.diff(bins_r), alpha=0.5)
plt.xlabel("$r$")
plt.ylabel("$f(r)$")
plt.show()

# %%
# Angular distribution

hist_theta, bins_theta = np.histogram(theta_val, bins=1000, density=True)
theta = np.linspace(bins_theta[0], bins_theta[-1], 1000)

plt.figure()
plt.plot(theta, [1 / (2 * np.pi) for _ in theta], "r--", lw=2)
plt.bar(bins_theta[:-1], hist_theta, np.diff(bins_theta), alpha=0.5)
plt.xlabel(r"$\theta$")
plt.ylabel(r"$f(\theta)$")

plt.show()
