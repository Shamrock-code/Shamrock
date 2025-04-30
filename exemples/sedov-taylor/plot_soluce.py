import os
import sys

import matplotlib.pyplot as plt
import numpy as np

import shamrock

sedov_sol = shamrock.phys.SedovTaylor()

r_theo = np.linspace(0, 1, 300)
p_theo = []
vr_theo = []
rho_theo = []
for i in range(len(r_theo)):
    _rho_theo, _vr_theo, _p_theo = sedov_sol.get_value(r_theo[i])
    p_theo.append(_p_theo)
    vr_theo.append(_vr_theo)
    rho_theo.append(_rho_theo)

r_theo = np.array(r_theo)
p_theo = np.array(p_theo)
vr_theo = np.array(vr_theo)
rho_theo = np.array(rho_theo)

gamma = 5.0 / 3.0

plt.style.use("custom_style.mplstyle")
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(9, 6), dpi=125)

axs[0, 0].plot(r_theo, vr_theo, c="red", label="v (theory)")
axs[1, 0].plot(r_theo, p_theo / ((gamma - 1) * rho_theo), c="red", label="u (theory)")
axs[0, 1].plot(r_theo, rho_theo, c="red", label="rho (theory)")
axs[1, 1].plot(r_theo, p_theo, c="red", label="P (theory)")

print("double r_theo[] ={")
for a in r_theo:
    print(f"    {a:.17e},")
print("};")


print("double vr_theo[] ={")
for a in vr_theo:
    print(f"    {a:.17e},")
print("};")


print("double rho_theo[] ={")
for a in rho_theo:
    print(f"    {a:.17e},")
print("};")


print("double p_theo[] ={")
for a in p_theo:
    print(f"    {a:.17e},")
print("};")

axs[0, 0].set_ylabel(r"$v$")
axs[1, 0].set_ylabel(r"$u$")
axs[0, 1].set_ylabel(r"$\rho$")
axs[1, 1].set_ylabel(r"$P$")

axs[0, 0].set_xlabel("$r$")
axs[1, 0].set_xlabel("$r$")
axs[0, 1].set_xlabel("$r$")
axs[1, 1].set_xlabel("$r$")

axs[0, 0].set_xlim(0, 0.55)
axs[1, 0].set_xlim(0, 0.55)
axs[0, 1].set_xlim(0, 0.55)
axs[1, 1].set_xlim(0, 0.55)

plt.tight_layout()
plt.show()
