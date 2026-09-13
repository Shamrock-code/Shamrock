"""
Showcase smoothing length iteration algorithlm
==============================================
"""

import matplotlib.pyplot as plt
import numpy as np

import shamrock


def compute_sums(pmass, id_a, h_a, W, dhW, positions: np.ndarray):
    rho_sum = 0
    sumdWdh = 0

    for j in range(positions.shape[0]):
        dr = positions[id_a, :] - positions[j, :]
        rab2 = dr.dot(dr)

        rab = np.sqrt(rab2)
        rho_sum += pmass * W(rab, h_a)
        sumdWdh += pmass * dhW(rab, h_a)

    return rho_sum, sumdWdh


def W(r, h):
    return shamrock.math.sphkernel.M4_W3d(r, h)


def dhW(r, h):
    return shamrock.math.sphkernel.M4_dhW3d(r, h)


def rho_h(m, h, hfact):
    return m * (hfact / h) * (hfact / h) * (hfact / h)


hfact = 1.2  # shamrock.math.sphkernel.hfactd

# SolverConfig.hpp defaults (src/shammodels/sph/include/shammodels/sph/SolverConfig.hpp:708-714)
epsilon_h = 1e-6  # convergence threshold on eps = |new_h - h_a| / h_old
h_evol_iter_max = 1.1  # htol_up_fine_cycle: per Newton-step clamp on new_h/h_a
h_evol_max = 1.1  # htol_up_coarse_cycle: per subcycle clamp on new_h/ha_0 (h at subcycle start)
h_iter_per_subcycles = 50  # LoopSmoothingLengthIter's Newton sweep count per subcycle
h_max_subcycles_count = 100  # sph_prestep's ghost-zone-rebuild subcycle count


def f_df(rho_ha, rho_sum, sumdWdh, h_a):
    f_iter = rho_sum - rho_ha
    df_iter = sumdWdh + 3 * rho_ha / h_a
    return f_iter, df_iter


def f_kernel(q):
    return shamrock.math.sphkernel.M4_f(q)


def df_kernel(q):
    return shamrock.math.sphkernel.M4_df(q)


def plot_f_df_kernel():
    q = np.linspace(0, 4, 1000)

    f_values = np.array([f_kernel(x) for x in q])
    df_values = np.array([df_kernel(x) for x in q])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(q, f_values, label=r"$f(q)$")
    ax.plot(q, df_values, label=r"$df(q)$")
    ax.plot(q, f_values + df_values * q / 3, label=r"$f(q) + df(q) \cdot q / 3$")
    ax.set_xlabel(r"$q$")
    ax.legend()
    plt.show()


def newton_iterate_new_h(h_a, positions, state_vars: dict):
    """One Newton-Raphson sweep, reproducing the per-particle branch of
    IterateSmoothingLengthDensity.cpp (src/shammodels/sph/src/modules/
    IterateSmoothingLengthDensity.cpp:52-119).

    state_vars["ha_0"] is h_old: the h value at the start of the current
    subcycle (reset by the caller each time sph_prestep would rebuild the
    ghost zone), NOT the previous Newton iterate.

    Returns (new_h, eps). eps == -1 is the sentinel the real kernel uses to
    mean "new_h would exceed ha_0 * h_evol_max (htol_up_coarse_cycle)": the
    caller must treat this as sph_prestep does and start a fresh subcycle.
    """
    ha_0 = state_vars["ha_0"]

    rho_ha = rho_h(pmass, h_a, hfact)
    rho_sum, sumdWdh = compute_sums(pmass, id_a, h_a, W, dhW, positions)
    f_iter, df_iter = f_df(rho_ha, rho_sum, sumdWdh, h_a)
    new_h = h_a - f_iter / df_iter

    # per-iteration clamp (htol_up_fine_cycle), relative to the previous iterate h_a
    h_max_evol_m = 1.0 / h_evol_iter_max
    h_max_evol_p = h_evol_iter_max
    new_h = max(new_h, h_a * h_max_evol_m)
    new_h = min(new_h, h_a * h_max_evol_p)

    # per-subcycle clamp (htol_up_coarse_cycle), relative to ha_0 (h at subcycle start)
    if new_h < ha_0 * h_evol_max:
        eps = abs(new_h - h_a) / ha_0
    else:
        new_h = ha_0 * h_evol_max
        eps = -1.0

    return new_h, eps


def analyse_h_convergence(
    positions: np.ndarray, id_a: int, pmass: float, iterate_new_h, ax, test_h_values: np.ndarray
):

    found_h_a = None
    histories = []
    for init_h_a in test_h_values:
        h_a = init_h_a
        history_h_a = [h_a]
        subcycle_end_indices = []
        converged = False

        # outer loop: sph_prestep's ghost-zone-rebuild subcycle
        # (src/shammodels/sph/src/Solver.cpp:1235, hstep_cnt < h_max_subcycles_count)
        for hstep_cnt in range(h_max_subcycles_count):
            # each subcycle resets h_old to the current h (Solver.cpp:1245)
            state_vars = {"ha_0": h_a}

            # inner loop: LoopSmoothingLengthIter's Newton sweep count
            # (LoopSmoothingLengthIter.cpp:31, iter_h < h_iter_per_subcycles)
            for iter_h in range(h_iter_per_subcycles):
                h_a_prev = h_a
                h_a, eps = iterate_new_h(h_a, positions, state_vars)
                assert h_a <= h_evol_iter_max * h_a_prev, (
                    f"h_a = {h_a} is larger than h_evol_iter_max * h_a_prev = {h_evol_iter_max * h_a_prev}"
                )
                history_h_a.append(h_a)

                if eps < 0:
                    # stuck: h wants to exceed ha_0 * h_evol_max this subcycle.
                    # sph_prestep would rebuild a wider ghost zone here and retry;
                    # break the inner loop to start a fresh subcycle anchored at
                    # the (clamped) current h.
                    break
                if eps < epsilon_h:
                    converged = True
                    break

            # mark where this iter_h subcycle ended, whichever way it ended
            subcycle_end_indices.append(len(history_h_a) - 1)

            if converged:
                found_h_a = h_a
                break
        histories.append((init_h_a, history_h_a, converged, subcycle_end_indices))

    for init_h_a, history_h_a, converged, subcycle_end_indices in histories:
        (line,) = ax.plot(np.array(history_h_a) - found_h_a, label=f"init_h_a = {init_h_a}")
        end_idx = np.array(subcycle_end_indices)
        ax.plot(
            end_idx,
            np.array(history_h_a)[end_idx] - found_h_a,
            marker="x",
            linestyle="none",
            color=line.get_color(),
        )

    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_xlabel("iteration count")
    ax.set_ylabel(r"$\delta h_a$")
    ax.legend()

    # plt.show()

    iteration_counts = [
        (len(history_h_a) - 1 if converged else np.nan)
        for _, history_h_a, converged, _ in histories
    ]

    final_f_values = []
    for _, history_h_a, converged, _ in histories:
        final_h_a = history_h_a[-1]
        rho_ha = rho_h(pmass, final_h_a, hfact)
        rho_sum, sumdWdh = compute_sums(pmass, id_a, final_h_a, W, dhW, positions)
        final_f, _ = f_df(rho_ha, rho_sum, sumdWdh, final_h_a)
        final_f_values.append(final_f)

    return iteration_counts, final_f_values


positions = []

id_a = 0
Nside = 10
for ix in range(Nside):
    for iy in range(Nside):
        for iz in range(Nside):
            positions.append((ix, iy, iz))
            # positions.append(np.random.rand(3))

            if ix == 10 and iy == 10 and iz == 10:
                id_a = len(positions) - 1

pmass = 1.0 / 1000.0

positions = np.array(positions)

plot_f_df_kernel()

h_a_test = np.logspace(-3, 2, 1000)

f_values = np.zeros(h_a_test.shape)
df_values = np.zeros(h_a_test.shape)

rho_sum_values = np.zeros(h_a_test.shape)
rho_h_values = np.zeros(h_a_test.shape)

for i in range(h_a_test.shape[0]):
    rho_ha = rho_h(pmass, h_a_test[i], hfact)
    rho_sum, sumdWdh = compute_sums(pmass, id_a, h_a_test[i], W, dhW, positions)
    rho_sum_values[i] = rho_sum
    rho_h_values[i] = rho_ha
    f_values[i], df_values[i] = f_df(rho_ha, rho_sum, sumdWdh, h_a_test[i])

fig_rho, ax_rho = plt.subplots(figsize=(10, 5))
ax_rho.plot(h_a_test, f_values, label=r"$f(h_a) = \sum_b m_b W(r_{ab}, h_a) - \rho_h(m_a, h_a)$")
ax_rho.plot(
    h_a_test,
    df_values,
    label=r"$f'(h_a) = \sum_b m_b \frac{\partial W}{\partial h}(r_{ab}, h_a) + 3 \rho_h(m_a, h_a) / h_a$",
)
ax_rho.plot(h_a_test, rho_h_values, label=r"$\rho_h(m_a, h_a)$")
ax_rho.plot(h_a_test, rho_sum_values, label=r"$\rho_sum(m_a, h_a)$")

ax_rho.set_yscale("symlog", linthresh=1e-4)
ax_rho.set_xscale("log")
ax_rho.set_xlabel("h_a")
ax_rho.legend()

# sample 10 equally spaced values in h_a_test indexes
test_h_values = np.append(
    h_a_test[np.linspace(0, h_a_test.shape[0] - 1, 10).astype(int)], 1.7039887744498599
)
test_h_values = np.sort(test_h_values)

algs = {
    "Newton": newton_iterate_new_h,
    #"Newton (lim)": newton_iterate_new_h_lim,
    #"Bisection": bisect_iterate_new_h,
    #"Bisection + NR": bisect_NR_iterate_new_h,
}

x = np.arange(len(test_h_values))

for name, alg in algs.items():
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle(name)

    iteration_counts, final_f_values = analyse_h_convergence(
        positions, id_a, pmass, alg, axs[0], test_h_values
    )

    axs[1].bar(x, iteration_counts)
    axs[1].set_yscale("log")
    axs[1].set_xticks(x)
    axs[1].set_xticklabels([f"{v:.3g}" for v in test_h_values])
    axs[1].set_xlabel("init_h_a")
    axs[1].set_ylabel("iteration count")
    axs[1].set_title("Convergence speed")

    axs[2].bar(x, final_f_values)
    axs[2].set_yscale("symlog", linthresh=1e-14)
    axs[2].set_xticks(x)
    axs[2].set_xticklabels([f"{v:.3g}" for v in test_h_values])
    axs[2].set_xlabel("init_h_a")
    axs[2].set_ylabel(r"$f(h_a)$")
    axs[2].set_title("Residual at convergence")

    plt.tight_layout()

plt.show()
