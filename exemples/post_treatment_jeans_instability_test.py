from math import *

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

#####============================== matplot config start ===============================

lw, ms = 3.5, 14  # linewidth  #markersize
elw, cs = 0.75, 0.75  # elinewidth and capthick #capsize for errorbar specifically
fontsize = 35
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
mpl.rcParams["markers.fillstyle"] = "full"
mpl.rcParams["lines.dashed_pattern"] = 6.4, 1.6, 1, 1.6
mpl.rcParams["xtick.labelsize"] = fontsize
mpl.rcParams["ytick.labelsize"] = fontsize
mpl.rcParams["legend.fontsize"] = fontsize
mpl.rcParams["grid.linewidth"] = 5
mpl.rcParams["font.weight"] = "normal"
mpl.rcParams["font.serif"] = "latex"

from scipy.fft import rfft, rfftfreq
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

G = 1
rho0 = 1.0
Lambd = 0.5
A = 1e-4
x0 = 7.812500000000000000e-03


cc = np.sqrt((G * rho0 * Lambd * Lambd) / (np.pi))
print(cc)


cs_osc = np.linspace(0.04 + cc, cc + 4, 100)

print(f"cs_osc = {cs_osc}\n")
cs_col = np.linspace(0.1, cc - 0.001, 100)
print(f"cs_col = {cs_col}\n")


# cs_array = np.stack((cs_osc, cs_col)).T
# np.savetxt(f"CS-for-Jeans-instability-test-lambda-{Lambd}-cs-critique-{cc}-amp-{A}",cs_array)


def compute_jeans_lenght(cs, rho0=1.0, G=1.0):
    return np.sqrt((np.pi * cs * cs) / (G * rho0))


def compute_t_osc(lambd, lambd_jeans, rho0=1.0, G=1.0):
    return np.sqrt(np.pi / (G * rho0)) * (lambd / np.sqrt(lambd_jeans**2 - lambd**2))


def compute_t_col(lambd, lambd_jeans, rho0=1.0, G=1.0):
    return np.sqrt(1.0 / (4 * np.pi * G * rho0)) * (lambd / np.sqrt(-(lambd_jeans**2) + lambd**2))


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
    return T_list


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


cs_list_1 = [
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
]

len_list_1 = [
    347,
    386,
    424,
    462,
    501,
    539,
    578,
    616,
    654,
    693,
    731,
    770,
    1154,
    1538,
    1922,
    2306,
    2690,
    3074,
]


def zero_crossings(t, rho):
    """Return the interpolated zero-crossing times of rho(t)."""
    zt = []
    for i in range(len(rho) - 1):
        if rho[i] == 0:
            zt.append(t[i])
        elif rho[i] * rho[i + 1] < 0:  # sign change
            # Linear interpolation for zero: rho[i] + α(rho[i+1]-rho[i]) = 0
            alpha = -rho[i] / (rho[i + 1] - rho[i])
            zt.append(t[i] + alpha * (t[i + 1] - t[i]))
    return np.array(zt)


def mean_period_from_zero_crossings(t, rho):
    zt = zero_crossings(t, rho)

    # full periods: difference between every second zero crossing
    full_periods = zt[2:] - zt[:-2]

    return np.mean(full_periods), full_periods, zt


def all_oscillation_time_scales_zeros_crossing(A, Lambda, cs_list, len_list):
    abs_path = "/Users/lsewanou/code_workshop/Shamrock/build/"

    T_list = []

    for i in range(len(cs_list)):
        cs = cs_list[i]
        lsz = len_list[i]
        name_prefix = f"Jeans-instablity-test-datas-times-for-1-amp-{A}-cs-{cs}-rho0-{1}-lambda-{Lambda}-{64}--{lsz}"

        path = abs_path + name_prefix
        print(path)
        datas = np.loadtxt(path)

        rho_num = np.array(datas[:, 1])
        times = np.array(datas[:, 0])

        T, all_T, Zeros_T = mean_period_from_zero_crossings(times, rho_num)
        print()

        # print(f"Estimated period T = {T:.5f} - all_t = {all_T}")
        T_list.append(float(T))
    return T_list


def get_plot_ana(lambd, cs_osc, cs_col, rho0=1.0, G=1.0):
    # compute free fall time
    tff = np.sqrt((3 * np.pi) / (32 * rho0 * G))
    lambd_jeans_osc = compute_jeans_lenght(cs_osc, rho0, G)
    # print(lambd_jeans_osc)
    lambd_jeans_col = compute_jeans_lenght(cs_col, rho0, G)
    # print(lambd_jeans_col)
    t_osc = compute_t_osc(lambd, lambd_jeans_osc, rho0, G)
    t_col = compute_t_col(lambd, lambd_jeans_col, rho0, G)

    X_osc_ana = lambd / lambd_jeans_osc
    X_col_ana = lambd / lambd_jeans_col
    Y_osc_ana = t_osc / tff
    Y_col_ana = t_col / tff

    cs_np_arr = np.array(cs_list)
    discrete_lambd_jeans_col = compute_jeans_lenght(cs_np_arr, rho0, G)
    t_col_discrete = compute_t_col(lambd, discrete_lambd_jeans_col, rho0, G)
    X_col_num = lambd / discrete_lambd_jeans_col
    Y_col_ana_dis = t_col_discrete / tff
    delt_t_col = all_collapse_time_scales(A, rho0, lambd, cs_list, len_list)
    Y_col_num = np.array(delt_t_col) / tff
    err_col = np.abs(Y_col_num - Y_col_ana_dis) / Y_col_ana_dis

    cs_np_arr_1 = np.array(cs_list_1)
    discrete_lambd_jeans_osc = compute_jeans_lenght(cs_np_arr_1, rho0, G)
    t_osc_discrete = compute_t_osc(lambd, discrete_lambd_jeans_osc, rho0, G)
    X_osc_num = lambd / discrete_lambd_jeans_osc
    Y_osc_ana_dis = t_osc_discrete / tff
    delt_t_osc = all_oscillation_time_scales_zeros_crossing(A, lambd, cs_list_1, len_list_1)
    Y_osc_num = np.array(delt_t_osc) / tff

    err_osc = np.abs(Y_osc_num - Y_osc_ana_dis) / Y_osc_ana_dis

    fig, axs = plt.subplots(1, 2, figsize=(25, 10))
    plt.subplots_adjust(wspace=0.35)

    axs[0].plot(X_osc_ana, Y_osc_ana, "b")
    axs[0].plot(X_col_ana, Y_col_ana, "b")
    axs[0].axvline(1, 0, 8, color="black", linestyle="--")
    axs[0].scatter(X_col_num, Y_col_num, c="r", marker="o", label="Collapsing modes")
    axs[0].scatter(X_osc_num, Y_osc_num, c="g", marker="*", label="Oscillating modes")
    axs[0].set_xlabel("$ \\lambda / \\lambda_{J}$")
    axs[0].set_ylabel("$ T/t_{ff}$")
    axs[0].legend(prop={"size": fontsize})

    axs[1].scatter(X_osc_num, err_osc, c="g", marker="*", label="Oscillating modes")
    axs[1].scatter(X_col_num, err_col, c="r", marker="o", label="Collapsing modes")
    # axs[1].set_xscale('log')
    axs[1].set_yscale("log")
    axs[1].axvline(1, 0, 8, color="black", linestyle="--")
    axs[1].set_xlabel("$ \\lambda / \\lambda_{J}$")
    axs[1].set_ylabel("$ |\; \\Delta T \;| / T_{ana}$")
    axs[1].legend(prop={"size": fontsize})

    plt.tight_layout()
    plt.savefig(f"Characteristics-timescales-lambda-{Lambd}-amp-{A}.pdf")


get_plot_ana(Lambd, cs_osc, cs_col)
