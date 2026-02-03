"""
Shamrock 3D unit vector generator
=======================================

This example shows how to use the unit vector generator
"""

# %%

import matplotlib.pyplot as plt  # plots
import numpy as np  # sqrt & arctan2

import shamrock
import sympy as sp
from scipy.special import erfinv, erf


#random set of points between 0 and 1
np.random.seed(111)
points = np.random.rand(1000)[:]

range_start = (0,3)
range_end = (0,1) # must be between 0 and 1 because of the normalization

#define the function exp(-x^2) using sympy
x = sp.symbols('x')
f = sp.Abs(sp.sin(-x**2))

# Numerical integration of f over range_start
primitive = []
primitive_x = []
accum = 0
dx = (range_start[1] - range_start[0]) / 100
for x_val in np.linspace(range_start[0], range_start[1], 100):
    primitive.append(accum)
    primitive_x.append(x_val)
    accum += f.subs(x, x_val).evalf() * dx

primitive = np.array(primitive)
primitive_x = np.array(primitive_x)

# normalize f so that primitive[-1] = 1
norm = primitive[-1]
primitive = primitive / norm
f = f / norm

print(f"primitive = {primitive}")
print(f"primitive_x = {primitive_x}")

# plot f
plt.figure()
x_plot = np.linspace(range_start[0], range_start[1], 100)
f_plot = [f.subs(x, x_val).evalf() for x_val in x_plot]
plt.plot(x_plot, f_plot)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("f(x) = exp(-x^2)")

# plot primitive
plt.figure()
plt.plot(primitive_x, primitive)
plt.xlabel("x")
plt.ylabel("primitive(x)")
plt.title("primitive(x) = integral(f(x))")

# plot finv
plt.figure()
plt.plot(primitive, primitive_x)
plt.xlabel("x")
plt.ylabel("finv(x)")
plt.title("finv(x) = inverse(primitive(x))")

#interpolate primitive using scipy.interpolate.interp1d
from scipy.interpolate import interp1d
mapping_interp = interp1d(primitive, primitive_x, kind='linear')

points_mapped = [mapping_interp(point) for point in points]

print(f"points = {points}")
print(f"points_mapped = {points_mapped}")

plt.figure()
hist_r, bins_r = np.histogram(points, bins=101, density=True, range=range_end)
r = np.linspace(bins_r[0], bins_r[-1], 101)

plt.bar(bins_r[:-1], hist_r, np.diff(bins_r), alpha=0.5)
plt.xlabel("$r$")
plt.ylabel("$f(r)$")

plt.figure()
hist_r, bins_r = np.histogram(points_mapped, bins=101, density=True, range=range_start)
r = np.linspace(bins_r[0], bins_r[-1], 101)

plt.bar(bins_r[:-1], hist_r, np.diff(bins_r), alpha=0.5)
plt.plot(r, [f.subs(x, x_val).evalf() for x_val in r])
plt.xlabel("$r$")
plt.ylabel("$f(r)$")

plt.show()