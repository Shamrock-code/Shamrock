from math import *

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

#####============================== matplot config start ===============================

lw, ms = 3, 5  # linewidth  #markersize
elw, cs = 0.75, 0.75  # elinewidth and capthick #capsize for errorbar specifically
fontsize = 15
tickwidth, ticksize = 1.5, 4
mpl.rcParams["axes.titlesize"] = fontsize * 1.5
mpl.rcParams["axes.labelsize"] = fontsize * 1.5
mpl.rcParams["xtick.major.size"] = ticksize
mpl.rcParams["ytick.major.size"] = ticksize
mpl.rcParams["xtick.major.width"] = tickwidth
mpl.rcParams["ytick.major.width"] = tickwidth
mpl.rcParams["xtick.minor.size"] = ticksize
mpl.rcParams["ytick.minor.size"] = ticksize
mpl.rcParams["xtick.minor.width"] = tickwidth
mpl.rcParams["ytick.minor.width"] = tickwidth
mpl.rcParams["lines.linewidth"] = lw
mpl.rcParams["lines.markersize"] = ms
mpl.rcParams["lines.markeredgewidth"] = 1.15
mpl.rcParams["lines.dash_joinstyle"] = "bevel"
mpl.rcParams["markers.fillstyle"] = "top"
mpl.rcParams["lines.dashed_pattern"] = 6.4, 1.6, 1, 1.6
mpl.rcParams["xtick.labelsize"] = fontsize
mpl.rcParams["ytick.labelsize"] = fontsize
mpl.rcParams["legend.fontsize"] = fontsize
mpl.rcParams["grid.linewidth"] = 5
mpl.rcParams["font.weight"] = "normal"
mpl.rcParams["font.serif"] = "latex"


from scipy.interpolate import interp1d


def get_collapse_time_scales(data, A, rho0):
    times = data[:, 0]
    rho = data[:, 1]
    rho_start = A * rho0
    rho_target = A * rho0 * np.cosh(1)
    time_of_rho = interp1d(rho, times, kind="linear", fill_value="extrapolate")
    t_start = float(time_of_rho(rho_start))
    t_target = float(time_of_rho(rho_target))
    delta_t = t_target - t_start

    return delta_t


def all_collapse_time_scales(A, rho0, Lambda, cs_list, len_list):
    abs_path = "/Users/lsewanou/code_workshop/Shamrock/build/"
    T_list = []
    for i in range(len(cs_list)):
        cs = cs_list[i]
        lsz = len_list[i]
        name_prefix = f"Jeans-instablity-test-datas-times-for-1-amp-{A}-cs-{cs}-rho0-{1}-lambda-{Lambda}-{64}--{lsz}"

        path = abs_path + name_prefix

        print(path)
        rho_data = np.loadtxt(path)
        delta_t = get_collapse_time_scales(rho_data, A, rho0)
        T_list.append(delta_t)

        print(f" cs = {cs}     :   Tcol = {delta_t}  \n")

    print(f"cs = {cs_list}\n\n")
    print(f"T_col = {T_list}\n\n")


A = 1e-4
rho0 = 1.0
Lambda = 0.5
cs_list = [
    0.105,
    0.1164,
    0.1256,
    0.128,
    0.1329,
    0.1384,
    0.1439,
    0.1493,
    0.1548,
    0.164,
    0.1676,
    0.1713,
    0.1768,
    0.1823,
    0.1878,
    0.1932,
    0.1987,
    0.2042,
    0.2189,
    0.2262,
    0.2335,
    0.2426,
    0.2518,
    0.2591,
    0.2664,
    0.2737,
    0.2792,
]

len_list = [
    83,
    92,
    99,
    101,
    104,
    108,
    113,
    117,
    121,
    128,
    131,
    133,
    138,
    142,
    146,
    150,
    154,
    159,
    170,
    175,
    181,
    188,
    195,
    201,
    206,
    212,
    216,
]

all_collapse_time_scales(A, rho0, Lambda, cs_list, len_list)


# interpolate time as a function of density


# # --- Step 1: Create mock data ---
# # Suppose time goes from 0 to 10 seconds
# t = np.linspace(0, 10, 100)
# # Simulate a density curve (increasing with time)
# rho = 2.0 * np.sinh(0.2 * t) + 5.0  # arbitrary model

# # --- Step 2: Define parameters ---
# A = 1.0
# rho0 = rho[0]
# rho_start = A * rho0
# rho_target = A * rho0 * np.cosh(1)

# # --- Step 3: Interpolate time as a function of density ---
# time_of_rho = interp1d(rho, t, kind='linear', fill_value='extrapolate')

# t_start = float(time_of_rho(rho_start))
# t_target = float(time_of_rho(rho_target))
# delta_t = t_target - t_start

# print(f"ρ₀ = {rho0:.4f}")
# print(f"Target density = {rho_target:.4f}")
# print(f"Time to reach target = {delta_t:.4f} seconds")

# # --- Step 4: Plot for visualization ---
# plt.figure(figsize=(7,5))
# plt.plot(t, rho, label='Density vs. Time')
# plt.axhline(rho_start, color='green', linestyle='--', label=r'$A\rho_0$')
# plt.axhline(rho_target, color='red', linestyle='--', label=r'$A\rho_0\cosh(1)$')
# plt.scatter([t_start, t_target], [rho_start, rho_target], color='black')
# plt.xlabel('Time')
# plt.ylabel('Density')
# plt.title('Finding time for density increase')
# plt.legend()
# plt.grid(True)
# plt.show()
