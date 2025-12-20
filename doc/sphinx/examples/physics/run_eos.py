"""
Equations of state functions
=======================================

"""

# %%
# Machida06 EoS
# ^^^^^^^^^

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
# Tillotson EoS
# ^^^^^^^^^
# With Tillotson parameters for


rho_plot = 0.5 * np.logspace(1, 5)

kwargs_tillotson = {
    "rho0": 7.8e3,
    "E0": 0.095e8,
    "a": 0.5,
    "b": 1.5,
    "A": 1.279e11,
    "B": 1.05e11,
    "alpha": 5.0,
    "beta": 5.0,
    "u_iv": 0.024e8,
    "u_cv": 0.0867e8,
}

fig, axs = plt.subplots(1, 2)
fig.suptitle("Tillotson EoS")

for u in [1e5, 5e6, 1e8]:
    P_plot = []
    cs_plot = []
    for rho in rho_plot:
        P, _cs = shamrock.phys.eos.eos_Tillotson(rho=rho, u=u, **kwargs_tillotson)
        P_plot.append(P)
        cs_plot.append(_cs)
    axs[0].plot(rho_plot, P_plot, label=f"$u={u:.0e}$ J/kg")
    axs[1].plot(rho_plot, cs_plot, label=f"$u={u:.0e}$ J/kg")

axs[0].set_ylabel("$P$ [Pa]")
axs[0].set_title("$P(\\rho)$")
axs[1].set_ylabel("$c_s$ [m/s]")
axs[1].set_title("$c_s(\\rho)$")

for ax in axs:
    ax.set_xscale("log")
    ax.set_xlabel("$\\rho$ [kg.m^-3]")
    ax.axvline(x=kwargs_tillotson["rho0"], color="black", ls="--", lw="1", alpha=0.4)
    ax.legend()
fig.tight_layout()


plt.show()
