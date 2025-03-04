"""
Sedov blast solution
=================================

This example shows how to plot the analytical solution to the sedov blast
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

import shamrock

# %%
sedov_sol = shamrock.phys.SedovTaylor()

# %%
x = np.linspace(0, 1, 300)
p = []
vr = []
rho = []
for i in range(len(x)):
    _rho, _vr, _p = sedov_sol.get_value(x[i])
    p.append(_p)
    vr.append(_vr)
    rho.append(_rho)

# %%
plt.plot(x, rho, label="rho")
plt.plot(x, vr, label="vr")
plt.plot(x, p, label="p")
plt.xlabel("r")
plt.legend()
plt.tight_layout()
plt.show()
