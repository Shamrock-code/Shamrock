import os

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

import pyvista as pv




def get_mass(R, rho):
    return rho * (4.0 * np.pi / 3.0) * (R**3)





####
T0 = 10.0  # [K]
R0 = 7.07e16 * 1e-2  # [cm -> m]

rho0 = 1.38e-18 * 1e3  # [g/cm^3 -> kg/m^3]
M0 = get_mass(R0, rho0)  # [kg]
mu = 2.3  # molecular gas
m_H = 1.6735e-27  # [kg]
kb = 1.380649e-23
G =6.6743015e-11  

print(f"proton-mass = {m_H} \n")
E_th0 = (3.0 * M0 * kb * T0) / (2 * mu * m_H)  # [J]
E_grav0 = (-3.0 * G * M0**2) / (5.0 * R0)  # [J]
alpha0 = E_th0 / np.abs(E_grav0)

t_ff = np.sqrt((3.0 * np.pi) / (32.0 * G * rho0))  # [s]
cs_sqr = (kb * T0) / (mu * m_H)
lamb_J = np.sqrt((cs_sqr * np.pi) / (G * rho0))  # [m]
print(f"kb = {kb}\n")
print(f"G value from set-up = {G}\n")
print(f"Jeans length = {lamb_J}\n")
print(f"sound speed  = {np.sqrt(cs_sqr)}\n")
print(f"alpha = {alpha0}\n")
print(f"ss = {(3600 * 24 * 365)}\n")
print(f"free fall time = {t_ff / (3600 * 24 * 365)} years \n")
N_J = 16  # N_J points per Jeans length
L0 = 4 * R0  # [m]
min_reso = (L0 * N_J) / (lamb_J)
print(f"min reso = {min_reso}\n")
gamma = 5.0 / 3.0







#####






mesh = pv.read("_iso_collapse_0000.vtk")

print(mesh.array_names)

centers = mesh.cell_centers().points

x = centers[:,0]
y = centers[:,1]
z = centers[:,2]

rho = mesh["rho"]

rhovel = mesh["rhovel"]

vx = rhovel[:,0]
vy = rhovel[:,1]
vz = rhovel[:,2]


dz = L0/100

mask = np.abs(z) < dz

x = x[mask]
y = y[mask]

rho = rho[mask]

vx = vx[mask]
vy = vy[mask]
vz = vz[mask]

rhoe = mesh["rhoetot"][mask]

P = (gamma - 1.0) * rhoe

T = P * mu * m_H / (rho * kb)

R = np.sqrt(x*x + y*y)

vr = (x*vx + y*vy)/R

vr[R == 0] = 0

vol = mesh.compute_cell_sizes()["Volume"][mask]

rmin = R[R>0].min()
rmax = R.max()

rbins = np.logspace(
    np.log10(rmin),
    np.log10(rmax),
    100
)

rmid = 0.5*(rbins[1:] + rbins[:-1])

rho_prof = np.zeros(len(rmid))

for i in range(len(rmid)):
    m = (R >= rbins[i]) & (R < rbins[i+1])

    if np.any(m):
        rho_prof[i] = np.mean(rho[m])
    else:
        rho_prof[i] = np.nan

vr_prof = np.zeros(len(rmid))

for i in range(len(rmid)):
    m = (R >= rbins[i]) & (R < rbins[i+1])

    if np.any(m):
        vr_prof[i] = np.mean(vr[m])
    else:
        vr_prof[i] = np.nan

T_prof = np.zeros(len(rmid))

for i in range(len(rmid)):
    m = (R >= rbins[i]) & (R < rbins[i+1])

    if np.any(m):
        T_prof[i] = np.mean(T[m])
    else:
        T_prof[i] = np.nan

centers3d = mesh.cell_centers().points

x3 = centers3d[:,0]
y3 = centers3d[:,1]
z3 = centers3d[:,2]

r3 = np.sqrt(x3*x3 + y3*y3 + z3*z3)

rho3 = mesh["rho"]

vol3 = mesh.compute_cell_sizes()["Volume"]

mass_cell = rho3 * vol3

Menc = np.zeros(len(rmid))

for i,r in enumerate(rmid):
    Menc[i] = np.sum(mass_cell[r3 <= r])


plt.figure()

plt.loglog(rmid, rho_prof)

plt.xlabel("Radius [m]")
plt.ylabel(r"$\rho$ [kg m$^{-3}$]")

plt.tight_layout()
plt.show()

plt.figure()

plt.semilogx(rmid, vr_prof)

plt.xlabel("Radius [m]")
plt.ylabel(r"$v_r$ [m s$^{-1}$]")

plt.tight_layout()
plt.show()

plt.figure()

plt.loglog(rmid, T_prof)

plt.xlabel("Radius [m]")
plt.ylabel("T [K]")

plt.tight_layout()
plt.show()

plt.figure()

plt.loglog(rmid, Menc)

plt.xlabel("Radius [m]")
plt.ylabel("M(<r) [kg]")

plt.tight_layout()
plt.show()


plt.figure()

plt.loglog(rho, T, ".")

plt.xlabel(r"$\rho$ [kg m$^{-3}$]")
plt.ylabel("T [K]")

plt.tight_layout()
plt.show()