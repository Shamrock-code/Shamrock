"""
Showcase smoothing length iteration algorithlm
==============================================
"""

# sphinx_gallery_thumbnail_number = 2
# sphinx_gallery_multi_image = "single"

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


def count_neighbors(id_a, h_a, W, positions: np.ndarray):
    count = 0

    for j in range(positions.shape[0]):
        dr = positions[id_a, :] - positions[j, :]
        rab2 = dr.dot(dr)

        rab = np.sqrt(rab2)
        if W(rab, h_a) > 0:
            count += 1

    return count


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


def simulate_h_iter(init_h_a, positions: np.ndarray, id_a: int, pmass: float, iterate_new_h):
    """Run the full h iteration (outer ghost-zone subcycles + inner Newton
    sweeps) starting from init_h_a, and return its history.

    Returns (history_h_a, history_f, history_df, history_neigh_count, converged,
    subcycle_end_indices).
    """
    h_a = init_h_a
    history_h_a = [h_a]
    history_f = []
    history_df = []
    history_neigh_count = []
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
            h_a, eps = iterate_new_h(h_a, positions, state_vars)
            history_h_a.append(h_a)

            rho_ha = rho_h(pmass, h_a, hfact)
            rho_sum, sumdWdh = compute_sums(pmass, id_a, h_a, W, dhW, positions)
            f_iter, df_iter = f_df(rho_ha, rho_sum, sumdWdh, h_a)
            history_f.append(f_iter)
            history_df.append(df_iter)
            history_neigh_count.append(count_neighbors(id_a, h_a, W, positions))

            if eps < 0:
                # stuck: h wants to exceed ha_0 * h_evol_max this subcycle.
                # sph_prestep would rebuild a wider ghost zone here and retry;
                # break the inner loop to start a fresh subcycle anchored at
                # the (clamped) current h.
                break
            if eps < epsilon_h:
                converged = True
                break

        # per-subcycle clamp (htol_up_coarse_cycle): whatever the inner Newton
        # loop did, h_a can never end a subcycle above ha_0 * h_evol_max
        # (Solver.cpp:1235-1425, ha_0 is h_old reset at the top of each hstep_cnt).
        ha_0 = state_vars["ha_0"]
        assert h_a <= h_evol_max * ha_0, (
            f"h_a = {h_a} is larger than h_evol_max * ha_0 = {h_evol_max * ha_0}"
        )

        # mark where this iter_h subcycle ended, whichever way it ended
        subcycle_end_indices.append(len(history_h_a) - 1)

        if converged:
            break

    return (
        history_h_a,
        history_f,
        history_df,
        history_neigh_count,
        converged,
        subcycle_end_indices,
    )


def analyse_h_convergence(
    positions: np.ndarray, id_a: int, pmass: float, iterate_new_h, test_h_values: np.ndarray
):

    histories = []
    for init_h_a in test_h_values:
        (
            history_h_a,
            history_f,
            history_df,
            history_neigh_count,
            converged,
            subcycle_end_indices,
        ) = simulate_h_iter(init_h_a, positions, id_a, pmass, iterate_new_h)
        histories.append(
            (init_h_a, history_h_a, history_f, history_neigh_count, converged, subcycle_end_indices)
        )

    best_entry = min(histories, key=lambda entry: np.abs(entry[2][-1]))
    found_h_a = best_entry[1][-1]

    # run one more simulation starting exactly at the found fixed point, and
    # insert it in sorted order (by init_h_a) alongside the other traces
    (
        history_h_a,
        history_f,
        history_df,
        history_neigh_count,
        converged,
        subcycle_end_indices,
    ) = simulate_h_iter(found_h_a, positions, id_a, pmass, iterate_new_h)
    insert_pos = np.searchsorted([entry[0] for entry in histories], found_h_a)
    histories.insert(
        insert_pos,
        (found_h_a, history_h_a, history_f, history_neigh_count, converged, subcycle_end_indices),
    )

    iteration_counts = [
        (len(history_h_a) - 1 if converged else np.nan)
        for _, history_h_a, _, _, converged, _ in histories
    ]

    final_f_values = [history_f[-1] for _, _, history_f, _, _, _ in histories]

    return histories, found_h_a, iteration_counts, final_f_values


def plot_h_convergence(histories, found_h_a, axs):
    ax_h, ax_neigh = axs

    for (
        init_h_a,
        history_h_a,
        history_f,
        history_neigh_count,
        converged,
        subcycle_end_indices,
    ) in histories:
        end_idx = np.array(subcycle_end_indices)

        (line,) = ax_h.plot(np.array(history_h_a) - found_h_a, label=f"init_h_a = {init_h_a}")
        ax_h.plot(
            end_idx,
            np.array(history_h_a)[end_idx] - found_h_a,
            marker="x",
            linestyle="none",
            color=line.get_color(),
        )

        # history_neigh_count has no entry for the initial h_a, so its index i
        # lines up with history_h_a's index i + 1 on the shared x-axis
        neigh_x = np.arange(1, len(history_neigh_count) + 1)
        ax_neigh.plot(neigh_x, history_neigh_count, color=line.get_color())
        ax_neigh.plot(
            end_idx,
            np.array(history_neigh_count)[end_idx - 1],
            marker="x",
            linestyle="none",
            color=line.get_color(),
        )

    ax_h.set_yscale("symlog", linthresh=1e-3)
    ax_h.set_ylabel(r"$\delta h_a$")
    ax_h.legend()

    ax_neigh.set_xlabel("iteration count")
    ax_neigh.set_ylabel("neighbor count")
    ax_neigh.set_yscale("log")


def plot_rho_f_df(h_a_test):
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
    ax_rho.plot(
        h_a_test, f_values, label=r"$f(h_a) = \sum_b m_b W(r_{ab}, h_a) - \rho_h(m_a, h_a)$"
    )
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


# %%

plot_f_df_kernel()

# %%


def generate_cubic_distrib(Nside):
    positions = []

    id_a = 0
    for ix in range(Nside):
        for iy in range(Nside):
            for iz in range(Nside):
                positions.append((ix, iy, iz))
                # positions.append(np.random.rand(3))

                if ix == 10 and iy == 10 and iz == 10:
                    id_a = len(positions) - 1

    positions = np.array(positions)

    return positions, id_a


positions, id_a = generate_cubic_distrib(Nside=10)
pmass = 1.0 / 1000.0

# %%

h_a_test = np.logspace(-3, 2, 1000)

plot_rho_f_df(h_a_test)

# %%

# sample 10 equally spaced values in h_a_test indexes
test_h_values = np.logspace(-3, 2, 10)
algs = {
    "Newton": newton_iterate_new_h,
    # "Newton (lim)": newton_iterate_new_h_lim,
    # "Bisection": bisect_iterate_new_h,
    # "Bisection + NR": bisect_NR_iterate_new_h,
}

results = {}
for name, alg in algs.items():
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(name)

    histories, found_h_a, iteration_counts, final_f_values = analyse_h_convergence(
        positions, id_a, pmass, alg, test_h_values
    )

    plot_h_convergence(histories, found_h_a, axs)

    # histories may hold one more entry than test_h_values (the extra run
    # seeded at found_h_a), so derive the x-axis from histories itself
    init_h_a_values = [entry[0] for entry in histories]
    results[name] = (init_h_a_values, iteration_counts, final_f_values)

    plt.tight_layout()

fig, axs = plt.subplots(2, 1, figsize=(10, 8))
fig.suptitle("Algorithm comparison")

bar_width = 0.8 / len(results)
for i, (name, (init_h_a_values, iteration_counts, final_f_values)) in enumerate(results.items()):
    x = np.arange(len(init_h_a_values)) + i * bar_width
    axs[0].bar(x, iteration_counts, width=bar_width, label=name)
    axs[1].bar(x, final_f_values, width=bar_width, label=name)

first_init_h_a_values = next(iter(results.values()))[0]
xticks = np.arange(len(first_init_h_a_values)) + bar_width * (len(results) - 1) / 2
xticklabels = [f"{v:.3g}" for v in first_init_h_a_values]

axs[0].set_yscale("log")
axs[0].set_xticks(xticks)
axs[0].set_xticklabels(xticklabels)
axs[0].set_xlabel("init_h_a")
axs[0].set_ylabel("iteration count")
axs[0].set_title("Convergence speed")
axs[0].legend()

axs[1].set_yscale("symlog", linthresh=1e-14)
axs[1].set_xticks(xticks)
axs[1].set_xticklabels(xticklabels)
axs[1].set_xlabel("init_h_a")
axs[1].set_ylabel(r"$f(h_a)$")
axs[1].set_title("Residual at convergence")
axs[1].legend()

plt.tight_layout()

plt.show()
