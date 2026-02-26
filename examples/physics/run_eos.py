"""
Equations of state functions
=======================================

"""

# %%
# Machida06 EoS
# ^^^^^^^^^^^^^

import matplotlib.pyplot as plt
import numpy as np

import shamrock

cs = 190.0
rho_c1 = 1.92e-13 * 1000  # g/cm^3 -> kg/m^3
rho_c2 = 3.84e-8 * 1000  # g/cm^3 -> kg/m^3
rho_c3 = 1.92e-3 * 1000  # g/cm^3 -> kg/m^3


si = shamrock.UnitSystem()
sicte = shamrock.Constants(si)
kb = sicte.kb()
print(kb)
mu = 2.375
mh = 1.00784 * sicte.dalton()
print(mu * mh * kb)

rho_plot = np.logspace(-15, 5, 1000)
P_plot = []
cs_plot = []
T_plot = []
for rho in rho_plot:
    P, _cs, T = shamrock.phys.eos.eos_Machida06(
        cs=cs, rho=rho, rho_c1=rho_c1, rho_c2=rho_c2, rho_c3=rho_c3, mu=mu, mh=mh, kb=kb
    )
    P_plot.append(P)
    cs_plot.append(_cs)
    T_plot.append(T)

plt.figure()
plt.plot(rho_plot, P_plot, label="P")
plt.yscale("log")
plt.xscale("log")
plt.xlabel("$\\rho$ [kg.m^-3]")
plt.ylabel("$P$ [Pa]")
plt.axvspan(rho_c1, rho_c2, color="grey", alpha=0.5)
plt.axvspan(rho_c3, rho_plot[-1], color="grey", alpha=0.5)

plt.figure()
plt.plot(rho_plot, cs_plot, label="cs")
plt.yscale("log")
plt.xlabel("$\\rho$ [kg.m^-3]")
plt.xscale("log")
plt.ylabel("$c_s$ [m/s]")
plt.axvspan(rho_c1, rho_c2, color="grey", alpha=0.5)
plt.axvspan(rho_c3, rho_plot[-1], color="grey", alpha=0.5)

plt.figure()
plt.plot(rho_plot, T_plot, label="T")
plt.yscale("log")
plt.xscale("log")
plt.xlabel("$\\rho$ [kg.m^-3]")
plt.ylabel("$T$ [K]")
plt.axvspan(rho_c1, rho_c2, color="grey", alpha=0.5)
plt.axvspan(rho_c3, rho_plot[-1], color="grey", alpha=0.5)


plt.tight_layout()

# %%
# Fermi gas EoS
# ^^^^^^^^^^^^^

rho_plot = np.logspace(1, 20, 1000)
P_plot = []
cs_plot = []

for rho in rho_plot:
    P, _cs = shamrock.phys.eos.eos_Fermi(mu_e=2, rho=rho)
    P_plot.append(P)
    cs_plot.append(_cs)

plt.figure()
plt.suptitle("Fermi Gas EoS")
plt.plot(rho_plot, P_plot, label="P", color="blue")
plt.yscale("log")
plt.xscale("log")
plt.xlabel("$\\rho$ [kg.m^-3]")
plt.ylabel("$P$ [Pa]", color="blue")
plt.legend()


ax = plt.twinx()
ax.plot(rho_plot, cs_plot, label="cs", color="orange")
ax.set_yscale("log")
ax.set_ylabel("$c_s$ [m/s]", color="orange")
ax.legend(loc="lower right")
