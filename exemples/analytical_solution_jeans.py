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


# from cmath import *
# def analytical_1(t,w,x,A,rho0,k,sign=1):
#     res = 0.0 + 0.0j
#     res = A*rho0 *exp (k*x + sign*w*t)
#     return res.real, res.imag


def plot_analytical_solution_osc_times(A, k, rho0, Lambd, w_lambd, times, x_pos):
    density = [A * rho0 * np.cos((2 * pi * x_pos) / Lambd) * np.cos(w_lambd * t) for t in times]
    velocity = [
        ((A * w_lambd) / k) * np.sin((2 * np.pi * x_pos) / Lambd) * np.sin(w_lambd * t)
        for t in times
    ]
    return density, velocity


def plot_analytical_solution_osc_snapshots(A, k, rho0, Lambd, w_lambd, positions, t_last):
    density = [
        A * rho0 * np.cos((2 * pi * x) / Lambd) * np.cos(w_lambd * t_last) for x in positions
    ]
    velocity = [
        ((A * w_lambd) / k) * np.sin((2 * np.pi * x) / Lambd) * np.sin(w_lambd * t_last)
        for x in positions
    ]
    return density, velocity


def plot_analytical_solution_col_times(A, k, rho0, Lambd, gam_lambd, times, x_pos):
    density = [A * rho0 * np.cos((2 * pi * x_pos) / Lambd) * np.cosh(gam_lambd * t) for t in times]
    velocity = [
        ((A * gam_lambd) / k) * np.sin((2 * np.pi * x_pos) / Lambd) * np.sinh(gam_lambd * t)
        for t in times
    ]
    return density, velocity


def plot_analytical_solution_col_snapshots(A, k, rho0, Lambd, gam_lambd, positions, t_last):
    density = [
        A * rho0 * np.cos((2 * pi * x) / Lambd) * np.cosh(gam_lambd * t_last) for x in positions
    ]
    velocity = [
        ((A * gam_lambd) / k) * np.sin((2 * np.pi * x) / Lambd) * np.sinh(gam_lambd * t_last)
        for x in positions
    ]
    return density, velocity


def get_analytical_osc_component(A, L, rho0, cs, Lambd, G, times, positions):
    x0 = positions[0]
    t_last = times[-1]
    k = (2 * np.pi) / (L * Lambd)
    Lambd_jeans = np.sqrt((np.pi * cs * cs) / (G * rho0))
    w_lambd = 2 * np.pi * cs * np.sqrt((1.0 / (Lambd**2) - 1.0 / (Lambd_jeans**2)))
    dens_in_time, vel_in_time = plot_analytical_solution_osc_times(
        A, k, rho0, Lambd, w_lambd, times, x0
    )
    dens_fix_time, vel_fix_time = plot_analytical_solution_osc_snapshots(
        A, k, rho0, Lambd, w_lambd, positions, t_last
    )

    fig, axs = plt.subplots(2, 2, figsize=(35, 20))
    plt.subplots_adjust(wspace=0.25)
    axs[0][0].plot(times, dens_in_time, "bo", lw=2, label="gas-num")
    axs[0][0].set_xlabel("Time", fontsize=15, fontweight="bold")
    axs[0][0].set_ylabel("Density", fontsize=15, fontweight="bold")

    axs[0][1].plot(times, vel_in_time, "bo", lw=2, label="gas-num")
    axs[0][1].set_xlabel("Time", fontsize=15, fontweight="bold")
    axs[0][1].set_ylabel("Velocity", fontsize=15, fontweight="bold")

    axs[1][0].plot(positions, dens_fix_time, "bo", lw=2, label="gas-num")
    axs[1][0].set_xlabel("x", fontsize=15, fontweight="bold")
    axs[1][0].set_ylabel("Density", fontsize=15, fontweight="bold")

    axs[1][1].plot(positions, vel_fix_time, "bo", lw=2, label="gas-num")
    axs[1][1].set_xlabel("x", fontsize=15, fontweight="bold")
    axs[1][1].set_ylabel("Velocity", fontsize=15, fontweight="bold")

    plt.legend(prop={"weight": "bold"})
    plt.savefig(f"Jeans-analytical-Lambda[{Lambd}]-cs[{cs}]-A[{A}]-tfinal-[{times[-1]}].pdf")


"""
G=1.
L=1.
cs = 1.0
Lambd = 0.5
A = 1e-6
rho0=1
X = [0.015625 ,0.046875 ,0.078125 ,0.109375 ,0.140625 ,0.171875 ,0.203125 ,0.234375,
    0.265625  ,0.296875 ,0.328125 ,0.359375 ,0.390625 ,0.421875 ,0.453125 ,0.484375,
    0.515625  ,0.546875 ,0.578125 ,0.609375 ,0.640625 ,0.671875 ,0.703125 ,0.734375,
    0.765625  ,0.796875 ,0.828125 ,0.859375 ,0.890625 ,0.921875 ,0.953125 ,0.984375]

times = [0.0, 0.005208333333333333, 0.01041663539065743, 0.015624905915498286, 0.020833144593192972,
        0.026041351119933373, 0.031249525256449015, 0.036457666870007094, 0.04166577596668926, 0.0468738527121521,
        0.05208189744313566, 0.057289910668692906, 0.06249789306476213, 0.0677058454638124, 0.07291376884051927,
        0.07812166429276544, 0.08332953302937608, 0.08853737636197295, 0.09374519568214347, 0.09895299244956265,
        0.10416076818891064, 0.109368524476186, 0.11457626293288214, 0.11978398522530602, 0.1249916930630202,
        0.13019938819165122, 0.13540707247448125, 0.14061474756980546, 0.145822411905032, 0.15103006646998876,
        0.15623771274682025, 0.1614453522611995, 0.16665298656891545, 0.17186061725352533, 0.17706824592460638,
        0.18227587421427188, 0.18748350377185333, 0.1926911362564222, 0.19789877332840097, 0.20310641664030757,
        0.2083140678254482, 0.2135217284947187, 0.2187294002228816, 0.2239370845377892, 0.2291447829223204,
        0.23435249119833523, 0.23956020804076308, 0.2447679346051384, 0.24997567202550236, 0.2551834143138035,
        0.2603911566671876, 0.2655988978055568, 0.27080663695281637, 0.27601437390318273, 0.28122210907257555,
        0.2864298413897398, 0.29163756059635515, 0.2968452676877302, 0.302052963734725, 0.3072606498669534,
        0.3124683272705862, 0.3176759971923063, 0.3228836595392839, 0.3280913091998801, 0.3332989478015778,
        0.338506577014315, 0.34371419855957724, 0.34892181419304463, 0.3541294257010156, 0.35933703489186375,
        0.3645446435872802, 0.3697522536140412, 0.3749598667970096, 0.3801674849498939, 0.3853751099420963,
        0.39058274356949296, 0.3957903875371838, 0.40099804088411295, 0.4062057021269666, 0.4114133736662209,
        0.4166210563599888, 0.42182875074248977, 0.42703645823028363, 0.4322441803429894, 0.4374519194063518,
        0.4426596774470996, 0.44786745633544045, 0.45307525782010005, 0.4582830835333544, 0.4634909349905357,
        0.4686988135832262, 0.47390672057357947, 0.47911465708799683, 0.4843226241107941, 0.4895306198705094,
        0.49473864364400527, 0.4999466964678675, 0.5051547792439093, 0.5103628927307453, 0.5155710375392147,
        0.5207792141334825, 0.5259874228485913, 0.53119565623169, 0.5364038940154083, 0.5416121360586059,
        0.5468203820568234, 0.5520286106201243, 0.557236814268456, 0.5624449925094132, 0.5676531451186974,
        0.5728612719215403, 0.5780693709561456, 0.5832774419368435, 0.5884854854330082, 0.593693502132252,
        0.5989014928098448, 0.6041094582381761, 0.6093173988448234, 0.6145253153471337, 0.6197332087652079,
        0.6249410801584145, 0.6301489306160057, 0.6353567612611585, 0.6405645732543909, 0.6457723677949856,
        0.6509801451809349, 0.6561879052133487, 0.6613956492973818, 0.6666033788594438, 0.6718110953374237,
        0.6770188003591361, 0.6822264954516148, 0.6874341821222851, 0.6926418614167894, 0.6978495345815129,
        0.7030572029985338, 0.70826486813645, 0.7134725315136581, 0.7186801946726741, 0.7238878591617919,
        0.7290955265207311, 0.7343031982717435, 0.7395108759130857, 0.7447185609135557, 0.7499262547104542,
        0.7551339587040702, 0.7603416742556562, 0.7655494026860233, 0.7707571452877133, 0.7759649033456907,
        0.7811726674104928, 0.786380431980716, 0.7915881909051914, 0.7967959407087979, 0.8020036822548221,
        0.8072114166889288, 0.8124191423228762, 0.8176268517908019, 0.8228345464504424, 0.8280422280922826,
        0.8332498976461422, 0.8384575560364798, 0.8436652044448743, 0.8488728443208658, 0.8540804770848893,
        0.8592881041692946, 0.8644957270408311, 0.8697033423923122, 0.8749109482864456, 0.8801185464121829,
        0.8853261384945746, 0.8905337262885, 0.8957413115730464, 0.9009488961465623, 0.9061564818171235,
        0.9113640703953854, 0.9165716636708486, 0.921779263584488, 0.926986872040211, 0.9321944908440318,
        0.9374021218316535, 0.9426097614263643, 0.9478174105191747, 0.9530250699561443, 0.9582327402010593,
        0.9634404220459732, 0.9686481167094182, 0.9738558262174898, 0.9790635532149852, 0.9842713000217034,
        0.9894790686560855, 0.9946868609599653, 0.9998946786419405, 1.0]

# get_analytical_osc_component(A,L,rho0,cs,Lambd,G,times,X)

"""
