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

# rho_plot = 0.5 * np.logspace(1, 5)
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
rho0 = kwargs_tillotson["rho0"]
u_iv = kwargs_tillotson["u_iv"]
u_cv = kwargs_tillotson["u_cv"]

rho_plot = np.linspace(0.5 * rho0, 1.5 * rho0)
P_plot = []
cs_plot = []
uc_plot = []

for rho in rho_plot:
    P, _cs = shamrock.phys.eos.eos_Tillotson(rho=rho, u=0, **kwargs_tillotson)
    _uc = shamrock.phys.eos.cold_energy_Tillotson(rho=rho, **kwargs_tillotson)
    P_plot.append(P)
    cs_plot.append(_cs)
    uc_plot.append(_uc)

P_plot = np.array(P_plot)
cs_plot = np.array(cs_plot)
uc_plot = np.array(uc_plot)

# rho_low = rho_plot[uc_plot > kwargs_tillotson["u_iv"]][0]
rho_low = rho0
rho_high = 2 * rho0
# rho_high = rho_plot[uc_plot > kwargs_tillotson["u_cv"]][0]

fig, axs = plt.subplots(2)
fig.suptitle("Fermi Gas EoS")
axs[0].plot(rho_plot, P_plot, color="blue")
axs[0].axvline(x=rho0, color="black", ls="--")
axs[0].set_xlabel("$\\rho$ [kg/m³]")
axs[0].set_ylabel("$P$ [Pa]", color="blue")
axs[0].legend()


ax_twin = axs[0].twinx()
ax_twin.set_ylabel("$c_s$ [m/s]", color="orange")
ax_twin.legend(loc="lower right")

axs[1].set_title("Cold energy")
axs[1].set_xlabel("$\\rho$ [kg/m³]")
axs[1].set_ylabel("$u_c$ [J/kg]", color="tab:blue")
axs[1].set_ylim(uc_plot[0], uc_plot[-1])
axs[1].plot(rho_plot, uc_plot, color="tab:blue")

axs[1].axhline(y=kwargs_tillotson["u_iv"], color="black", ls="--", lw=0.5, alpha=0.5)
axs[1].axhline(y=kwargs_tillotson["u_cv"], color="black", ls="--", lw=0.5, alpha=0.5)
axs[1].annotate(
    text=r"$u_{\rm iv}$",
    xy=(1, u_iv / uc_plot[-1]),
    xycoords="axes fraction",
    ha="right",
    va="bottom",
)
axs[1].annotate(
    text=r"$u_{\rm cv}$",
    xy=(1, u_cv / uc_plot[-1]),
    xycoords="axes fraction",
    ha="right",
    va="bottom",
)

for ax in axs:
    ax.set_xlim(rho_plot[0], rho_plot[-1])
    ax.axvspan(rho_plot[0], rho_low, color="grey", alpha=0.5)
    ax.axvspan(rho_high, rho_plot[-1], color="grey", alpha=0.5)

fig.tight_layout()

# fig, axs = plt.subplots(1, 2)
# fig.suptitle("Tillotson EoS")

# for u in [1e5, 5e6, 1e8]:
#     P_plot = []
#     cs_plot = []
#     for rho in rho_plot:
#         P, _cs = shamrock.phys.eos.eos_Tillotson(rho=rho, u=u, **kwargs_tillotson)
#         P_plot.append(P)
#         cs_plot.append(_cs)
#     axs[0].plot(rho_plot, P_plot, label=f"$u={u:.0e}$ J/kg")
#     axs[1].plot(rho_plot, cs_plot, label=f"$u={u:.0e}$ J/kg")

# axs[0].set_ylabel("$P$ [Pa]")
# axs[0].set_title("$P(\\rho)$")
# axs[1].set_ylabel("$c_s$ [m/s]")
# axs[1].set_title("$c_s(\\rho)$")

# for ax in axs:
#     ax.set_xscale("log")
#     ax.set_xlabel("$\\rho$ [kg.m^-3]")
#     ax.axvline(x=kwargs_tillotson["rho0"], color="black", ls="--", lw="1", alpha=0.4)
#     ax.legend()
# fig.tight_layout()

# %%
