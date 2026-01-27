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

# %%
# Tillotson EoS
# ^^^^^^^^^^^^^
# With Tillotson parameters for Granite (Benz et al. 1986)

kwargs_tillotson = {
    "rho0": 2.7e3,
    "E0": 1.6e7,
    "a": 0.5,
    "b": 1.3,
    "A": 1.8e10,
    "B": 1.8e10,
    "alpha": 5.0,
    "beta": 5.0,
    "u_iv": 3.5e6,
    "u_cv": 1.8e7,
}
cv = 790
rho0 = kwargs_tillotson["rho0"]
u_iv = kwargs_tillotson["u_iv"]
u_cv = kwargs_tillotson["u_cv"]

rho_plot = np.linspace(0.5 * rho0, 1.5 * rho0)
T_list = [0, 4000, 30000]
u_T_list = cv * np.array(T_list)
P_plot = [[] for _ in u_T_list]
cs_plot = []
uc_plot = []

# Compute the data
for rho in rho_plot:
    _uc = shamrock.phys.eos.cold_energy_Tillotson(rho=rho, **kwargs_tillotson)
    for i, u_T in enumerate(u_T_list):
        P, _cs = shamrock.phys.eos.eos_Tillotson(rho=rho, u=u_T, **kwargs_tillotson)
        P_plot[i].append(P)
    cs_plot.append(_cs)
    uc_plot.append(_uc)

P_plot = np.array(P_plot)
cs_plot = np.array(cs_plot)
uc_plot = np.array(uc_plot)

fig, axs = plt.subplots(2, figsize=(8, 8))
fig.suptitle("Tillotson EoS")


axs[0].set_title("Pressure")
axs[0].set_xlabel("$\\rho$ [kg/m³]")
axs[0].set_ylabel("$P$ [Pa]")
axs[1].set_title("Internal energy")
axs[1].set_xlabel("$\\rho$ [kg/m³]")
axs[1].set_ylabel("$u$ [J/kg]")

axs[1].axhline(y=kwargs_tillotson["u_iv"], color="black", ls="--", lw=1, alpha=0.5)
axs[1].axhline(y=kwargs_tillotson["u_cv"], color="black", ls="--", lw=1, alpha=0.5)
axs[0].annotate(
    text=r"$\rho_0$",
    xy=(rho0, np.max(P_plot)),
    ha="left",
    va="top",
)
axs[1].annotate(
    text=r"$u_{\rm iv}$",
    xy=(rho_plot[-1], u_iv),
    ha="right",
    va="top",
)
axs[1].annotate(
    text=r"$u_{\rm cv}$",
    xy=(rho_plot[-1], u_cv),
    ha="right",
    va="top",
)

for i, u_T in enumerate(u_T_list):
    axs[0].plot(rho_plot, P_plot[i], label=f"T={T_list[i]:.0e}K")
    axs[1].plot(
        rho_plot, uc_plot + np.full_like(uc_plot, u_T), label=f"T={T_list[i]:.0e}K"
    )
# Notice that at T=0K, u=u_{cold}.

for ax in axs:
    ax.set_xlim(rho_plot[0], rho_plot[-1])
    ax.axvspan(rho_plot[0], rho0, color="grey", alpha=0.1)
    ax.axvline(x=rho0, color="black", ls="--")
    ax.legend()

fig.tight_layout()
