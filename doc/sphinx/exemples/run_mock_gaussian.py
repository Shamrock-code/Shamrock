"""
Shamrock Gaussian generator
=======================================

This example shows how to use the mock gaussian function
"""

# %%

import shamrock

eng = shamrock.algs.gen_seed(111)

import matplotlib.pyplot as plt
import numpy as np

hist, bins = np.histogram(
    [shamrock.algs.mock_gaussian(eng) for _ in range(100000)], bins=1000, density=True
)
x = np.linspace(bins[0], bins[-1], 1000)
plt.plot(x, np.exp(-(x**2) / 2) / np.sqrt(2 * np.pi), "r--", lw=2)
plt.bar(bins[:-1], hist, np.diff(bins), alpha=0.5)
plt.show()
