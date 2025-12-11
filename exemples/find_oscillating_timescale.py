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

from scipy.fft import rfft, rfftfreq
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


def all_oscillation_time_scales(A, Lambda, cs_list, len_list):
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
        peaks, _ = find_peaks(rho_num)
        T_values = np.diff(times[peaks])
        T_mean = np.mean(T_values)
        print(f"Successive periods:  {T_values}   \t Average period: {T_mean} \n")

        if np.isnan(T_mean):
            T_list.append(5 + cs)
        else:
            T_list.append(float(T_mean))
    return T_list


def all_oscillation_time_scales_fft(A, Lambda, cs_list, len_list):
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
        dt = times[1] - times[0]
        yf = np.abs(rfft(rho_num))
        xf = rfftfreq(times.shape[0], dt)

        freq = xf[np.argmax(yf)]
        T = 1 / freq

        print("Dominant frequency:", freq)
        print("Estimated period:", T)


A = 1e-4
Lambda = 0.5
rho0 = 1.0
x0 = 7.812500000000000000e-03

# cs_list = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
#             1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

# len_list = [309, 347, 386, 424, 462, 501, 539, 578, 616, 654, 693, 731,
#            770,  1154, 1538, 1922, 2306,2690, 3074]

cs_list = [
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

len_list = [
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

# all_oscillation_time_scales_fft(A, Lambda, cs_list, len_list)

# TT = all_oscillation_time_scales(A, Lambda, cs_list, len_list)

# print(f" T_osc : {TT} \n")


def model_fixed(t, omega):
    return A * rho0 * np.cos(2 * np.pi * x0 / Lambda) * np.cos(omega * t)


# def model(t, C,omega,phi,D):
#     return C * np.cos(omega * t + phi) + D

# def all_oscillation_time_scales_1 (A, Lambda, cs_list, len_list):
#     abs_path="/Users/lsewanou/code_workshop/Shamrock/build/"

#     T_list = []

#     for  i in range(len(cs_list)):
#         cs = cs_list[i]
#         lsz = len_list[i]
#         name_prefix =f"Jeans-instablity-test-datas-times-for-1-amp-{A}-cs-{cs}-rho0-{1}-lambda-{Lambda}-{64}--{lsz}"
#         # name_prefix_space =f"Jeans-instablity-test-datas-spaces-for-1-amp-{A}-cs-{cs}-rho0-{1}-lambda-{Lambda}-{64}--{lsz}"

#         path = abs_path+name_prefix
#         # path_space = abs_path+name_prefix_space
#         print(path)
#         # print(path_space)
#         # datas_space = np.loadtxt(path_space)
#         datas = np.loadtxt(path)

#         rho_num = np.array(datas[:,1])
#         times = np.array(datas[:,0])
#         omega_guess = 2 * np.pi / (times[-1] - times[0])
#         p0 = [omega_guess]

#         # Fit
#         popt, pcov = curve_fit(model_fixed, times, rho_num, p0=p0)
#         omega_fit = popt[0]
#         omega_err = np.sqrt(np.diag(pcov))[0]

#         print(f"Fitted omega = {omega_fit:.5f} ± {omega_err:.5f}")

#         # Compute the period
#         T = 2 * np.pi / omega_fit
#         print(f"Estimated period T = {T:.5f}")
#         T_list.append(float(T))
#     return T_list

# T_list =  all_oscillation_time_scales_1(A, Lambda, cs_list, len_list)

# print(f" T_osc : {T_list} \n")


# def model(t, C,omega,phi,D):
#     return C * np.sin(omega * t + phi) + D

# def all_oscillation_time_scales_2 (A, Lambda, cs_list, len_list):
#     abs_path="/Users/lsewanou/code_workshop/Shamrock/build/"

#     T_list = []

#     for  i in range(len(cs_list)):
#         cs = cs_list[i]
#         lsz = len_list[i]
#         name_prefix =f"Jeans-instablity-test-datas-times-for-1-amp-{A}-cs-{cs}-rho0-{1}-lambda-{Lambda}-{64}--{lsz}"
#         # name_prefix_space =f"Jeans-instablity-test-datas-spaces-for-1-amp-{A}-cs-{cs}-rho0-{1}-lambda-{Lambda}-{64}--{lsz}"

#         path = abs_path+name_prefix
#         # path_space = abs_path+name_prefix_space
#         print(path)
#         # print(path_space)
#         # datas_space = np.loadtxt(path_space)
#         datas = np.loadtxt(path)

#         rho_num = np.array(datas[:,1])
#         times = np.array(datas[:,0])
#         omega_guess = 2 * np.pi / (times[-1] - times[0])
#         p0 = [omega_guess]
#         params, pcov = curve_fit(model,times,rho_num,p0=[1,1,0,0])
#         C, omega, phi, D = params

#         # Fit

#         omega_fit = omega
#         omega_err = np.sqrt(np.diag(pcov))[0]

#         print(f"Fitted omega = {omega_fit:.5f} ± {omega_err:.5f}")

#         # Compute the period
#         T = 2 * np.pi / omega_fit
#         print(f"Estimated period T = {T:.5f}")
#         T_list.append(float(T))
#     return T_list

# T_list =  all_oscillation_time_scales_2(A, Lambda, cs_list, len_list)

# print(f" T_osc : {T_list} \n")


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

        print(f"Estimated period T = {T:.5f} - all_t = {all_T}")
        T_list.append(float(T))
    return T_list


T_list = all_oscillation_time_scales_zeros_crossing(A, Lambda, cs_list, len_list)

print(f" T_osc : {T_list} \n")
