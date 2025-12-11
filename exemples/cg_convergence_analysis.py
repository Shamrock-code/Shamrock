import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, LogLocator

lw, ms = 4, 7  # linewidth  #markersize
elw, cs = 0.75, 0.75  # elinewidth and capthick #capsize for errorbar specifically
fontsize = 30
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
mpl.rcParams["grid.linewidth"] = 10
mpl.rcParams["font.weight"] = "normal"
mpl.rcParams["font.serif"] = "latex"


nb_cubes = [8, 16, 32, 64, 128, 256]
nb_cells = [512, 4096, 32768, 262144, 2097152, 16777216]
nb_iterations_stg1 = [4, 4, 4, 6, 13, 28]
nb_iterations = [25, 52, 105, 214, 434, 892]
l1_errors = [
    0.07088543540954591,
    0.017349621579817582,
    0.004229263363853544,
    0.001054769336729399,
    0.0002633896159460123,
    6.582817689760645e-05,
]
l2_errors = [
    0.07202988829914829,
    0.017036782726443923,
    0.0042040647458346705,
    0.001047648588988026,
    0.000261702861780769,
    6.541265357007495e-05,
]
linf_errors = [
    0.07819204147907248,
    0.02135206612624718,
    0.005438239115740109,
    0.0013656010248497062,
    0.00034177428388310803,
    8.546689196645526e-05,
]

N_h = [16, 128]
resol_er = [8e-1 * (1.0 / i**2) for i in N_h]
resol_it = [2 * i for i in N_h]


fig, axs = plt.subplots(1, 2, figsize=(25, 12))
plt.subplots_adjust(wspace=0.25, hspace=0.3, top=0.94, bottom=0.7, left=0.7, right=0.94)

axs[0].plot(nb_cubes, l1_errors, linestyle="-", marker="X", color="green", lw=5, label="$L1$")
axs[0].plot(nb_cubes, l2_errors, linestyle=":", marker="D", color="purple", lw=5, label="$L2$")
axs[0].plot(N_h, resol_er, "k--")
axs[0].text(
    32, 8e-1 * (1.0 / 32**2), "$\\rm{\\propto O\\left(\\frac{1}{h^{2}} \\right)}$", size=fontsize
)
axs[0].plot(
    nb_cubes, linf_errors, linestyle="-.", marker="8", color="orange", lw=5, label="$L_{\\infty}$"
)

axs[0].set_title("Errror to analytical solution ")
axs[0].set_xlabel("$N$", fontsize=fontsize)
axs[0].set_ylabel(
    " $\\frac{||\\phi_{num} - \\phi_{ana} ||}{|| \\phi_{ana} ||} $ ", fontsize=fontsize
)
# axs[0].set_ylim(5e-11,2.5e-5)
# Set log-log scale
axs[0].set_yscale("log")
axs[0].set_xscale("log")

# Format tick labels as integers
axs[0].xaxis.set_major_formatter(FormatStrFormatter("%d"))


# Set major and minor tick positions on log scale
axs[0].xaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0], numticks=10))
axs[0].yaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0], numticks=10))
# Customize tick appearance
axs[0].tick_params(which="major", length=10, width=3)
# Custom x-tick labels
label_ticks = nb_cubes
labels = [str(val) for val in nb_cubes]
axs[0].set_xticks(nb_cubes)
axs[0].set_xticklabels(labels, fontsize=fontsize)

# Y-axis tick size
axs[0].tick_params(axis="y", labelsize=fontsize)
axs[0].tick_params(axis="x", labelsize=fontsize)

# Frame style
axs[0].patch.set_edgecolor("black")
axs[0].patch.set_linewidth(5)

# Legend
axs[0].legend(prop={"size": fontsize})


# axs[1].plot(nb_cubes, nb_iterations_stg1, marker="X", color="green", lw=5, label="$1st$-stg")
axs[1].plot(nb_cubes, nb_iterations, marker="D", color="purple", lw=5)
axs[1].plot(N_h, resol_it, "k--")
axs[1].text(33.4, 2 * 32, "$\\rm{\\propto O\\left(h \\right)}$", size=fontsize)


axs[1].set_title("Iteration scaling ")
axs[1].set_xlabel("$N$", fontsize=fontsize)
axs[1].set_ylabel(" Number of iterations ", fontsize=fontsize)
# axs[0].set_ylim(5e-11,2.5e-5)
# Set log-log scale
axs[1].set_yscale("log")
axs[1].set_xscale("log")

# Format tick labels as integers
axs[1].xaxis.set_major_formatter(FormatStrFormatter("%d"))
axs[1].yaxis.set_major_formatter(FormatStrFormatter("%d"))


# Set major and minor tick positions on log scale
axs[1].xaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0], numticks=10))
axs[1].yaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0], numticks=10))
# Customize tick appearance
axs[1].tick_params(which="major", length=10, width=3)
# Custom x-tick labels
label_ticks = nb_cubes
labels = [str(val) for val in nb_cubes]
axs[1].set_xticks(nb_cubes)
axs[1].set_xticklabels(labels, fontsize=fontsize)

# Y-axis tick size
label_ticks = [25, 52, 105, 214, 434, 892]
labels = [str(val) for val in label_ticks]
axs[1].set_yticks(label_ticks)
axs[1].set_yticklabels(labels, fontsize=fontsize)


axs[1].tick_params(axis="y", labelsize=fontsize)
axs[1].tick_params(axis="x", labelsize=fontsize)

# Frame style
axs[1].patch.set_edgecolor("black")
axs[1].patch.set_linewidth(5)

# Legend
axs[1].legend(prop={"size": fontsize})

plt.tight_layout()
plt.savefig("CG-convergence-test-v2.pdf")
