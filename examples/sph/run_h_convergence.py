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

H_EVOL_MPI_MAX = 1.1


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


def newton_iterate_new_h(h_a, positions, state_vars: dict, h_max_evol_m=0.9, h_max_evol_p=1.1):
    if "ha_0" not in state_vars:
        state_vars["ha_0"] = h_a

    rho_ha = rho_h(pmass, h_a, hfact)
    rho_sum, sumdWdh = compute_sums(pmass, id_a, h_a, W, dhW, positions)
    f_iter, df_iter = f_df(rho_ha, rho_sum, sumdWdh, h_a)
    new_h = h_a - f_iter / df_iter

    print(
        f"new_h = {new_h}, h_a = {h_a}, f_iter = {f_iter}, df_iter = {df_iter}, lim m = {h_a * h_max_evol_m}, lim p = {h_a * h_max_evol_p}"
    )
    new_h = max(new_h, h_a * h_max_evol_m)

    new_h = min(new_h, h_a * h_max_evol_p)

    eps = abs(new_h - h_a) / state_vars["ha_0"]
    is_done = eps < 1e-6
    return new_h, is_done


def newton_iterate_new_h_lim(h_a, positions, state_vars: dict, h_max_evol_m=0.9, h_max_evol_p=1.1):
    if "ha_0" not in state_vars:
        state_vars["ha_0"] = h_a

    rho_ha = rho_h(pmass, h_a, hfact)
    rho_sum, sumdWdh = compute_sums(pmass, id_a, h_a, W, dhW, positions)
    f_iter, df_iter = f_df(rho_ha, rho_sum, sumdWdh, h_a)
    new_h = h_a - f_iter / df_iter

    print(
        f"new_h = {new_h}, h_a = {h_a}, f_iter = {f_iter}, df_iter = {df_iter}, lim m = {h_a * h_max_evol_m}, lim p = {h_a * h_max_evol_p}"
    )
    new_h = max(new_h, h_a * h_max_evol_m)

    new_h = min(new_h, h_a * h_max_evol_p)

    if f_iter > 0 and f_iter / df_iter < 0:
        new_h = h_a * h_max_evol_m

    eps = abs(new_h - h_a) / state_vars["ha_0"]
    is_done = eps < 1e-6
    return new_h, is_done


def bisect_iterate_new_h(h_a, positions, state_vars: dict, h_max_evol_m=0.5, h_max_evol_p=1.1):
    if "ha_0" not in state_vars:
        state_vars["ha_0"] = h_a
        state_vars["lo"] = 0
        state_vars["hi"] = np.inf

    rho_ha = rho_h(pmass, h_a, hfact)
    rho_sum, sumdWdh = compute_sums(pmass, id_a, h_a, W, dhW, positions)
    f_iter, df_iter = f_df(rho_ha, rho_sum, sumdWdh, h_a)

    lo = state_vars["lo"]
    hi = state_vars["hi"]

    if f_iter < 0:
        lo = h_a
    else:
        hi = h_a

    new_h = (lo + hi) / 2
    new_h = max(new_h, h_a * h_max_evol_m)
    new_h = min(new_h, h_a * h_max_evol_p)

    state_vars["lo"] = lo
    state_vars["hi"] = hi

    eps = abs(new_h - h_a) / state_vars["ha_0"]
    is_done = eps < 1e-6
    return new_h, is_done


def bisect_NR_iterate_new_h(h_a, positions, state_vars: dict, h_max_evol_m=0.5, h_max_evol_p=1.1):
    if "ha_0" not in state_vars:
        state_vars["ha_0"] = h_a
        state_vars["lo"] = 0
        state_vars["hi"] = np.inf

    rho_ha = rho_h(pmass, h_a, hfact)
    rho_sum, sumdWdh = compute_sums(pmass, id_a, h_a, W, dhW, positions)
    f_iter, df_iter = f_df(rho_ha, rho_sum, sumdWdh, h_a)

    lo = state_vars["lo"]
    hi = state_vars["hi"]

    if f_iter < 0:
        lo = h_a
    else:
        hi = h_a

    new_h_nr = h_a - f_iter / df_iter
    if lo < new_h_nr < hi:
        new_h = new_h_nr
    else:
        new_h = (lo + hi) / 2

    new_h = max(new_h, h_a * h_max_evol_m)
    new_h = min(new_h, h_a * h_max_evol_p)

    state_vars["lo"] = lo
    state_vars["hi"] = hi

    eps = abs(new_h - h_a) / state_vars["ha_0"]
    is_done = eps < 1e-6
    return new_h, is_done


def analyse_h_convergence(positions: np.ndarray, id_a: int, pmass: float, iterate_new_h):

    h_a_test = np.logspace(-2, 2, 1000)

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

    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    axs[0].plot(
        h_a_test, f_values, label=r"$f(h_a) = \sum_b m_b W(r_{ab}, h_a) - \rho_h(m_a, h_a)$"
    )
    axs[0].plot(
        h_a_test,
        df_values,
        label=r"$f'(h_a) = \sum_b m_b \frac{\partial W}{\partial h}(r_{ab}, h_a) + 3 \rho_h(m_a, h_a) / h_a$",
    )
    axs[0].plot(h_a_test, rho_h_values, label=r"$\rho_h(m_a, h_a)$")
    axs[0].plot(h_a_test, rho_sum_values, label=r"$\rho_sum(m_a, h_a)$")

    # plt.ylim(-10, 10)
    axs[0].set_yscale("symlog", linthresh=1e-4)
    axs[0].set_xscale("log")
    axs[0].set_xlabel("h_a")
    axs[0].legend()

    # sample 10 equally spaced values in h_a_test indexes
    test_h_values = np.append(
        h_a_test[np.linspace(0, h_a_test.shape[0] - 1, 100).astype(int)], 1.7039887744498599
    )

    found_h_a = None
    histories = []
    for init_h_a in test_h_values:
        h_a = init_h_a
        history_h_a = [h_a]
        state_vars = {}
        converged = False
        for i in range(100):
            h_a_prev = h_a
            h_a, is_done = iterate_new_h(h_a, positions, state_vars)
            assert h_a <= H_EVOL_MPI_MAX * h_a_prev, (
                f"h_a = {h_a} is larger than H_EVOL_MPI_MAX * h_a_prev = {H_EVOL_MPI_MAX * h_a_prev}"
            )
            history_h_a.append(h_a)
            if is_done:
                found_h_a = h_a
                converged = True
                break
        histories.append((init_h_a, history_h_a, converged))

    for init_h_a, history_h_a, converged in histories:
        axs[1].plot(np.array(history_h_a) - found_h_a, label=f"init_h_a = {init_h_a}")

    axs[1].set_yscale("symlog", linthresh=1e-3)
    axs[1].set_xlabel("iteration count")
    axs[1].set_ylabel(r"$\delta h_a$")
    axs[1].legend()

    # plt.show()

    iteration_counts = [
        (len(history_h_a) - 1 if converged else np.nan) for _, history_h_a, converged in histories
    ]

    final_f_values = []
    for _, history_h_a, converged in histories:
        final_h_a = history_h_a[-1]
        rho_ha = rho_h(pmass, final_h_a, hfact)
        rho_sum, sumdWdh = compute_sums(pmass, id_a, final_h_a, W, dhW, positions)
        final_f, _ = f_df(rho_ha, rho_sum, sumdWdh, final_h_a)
        final_f_values.append(final_f)

    return test_h_values, iteration_counts, final_f_values


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

algs = {
    "Newton": newton_iterate_new_h,
    "Newton (lim)": newton_iterate_new_h_lim,
    "Bisection": bisect_iterate_new_h,
    "Bisection + NR": bisect_NR_iterate_new_h,
}

scores = {}
f_scores = {}
for name, alg in algs.items():
    test_h_values, iteration_counts, final_f_values = analyse_h_convergence(
        positions, id_a, pmass, alg
    )
    scores[name] = iteration_counts
    f_scores[name] = final_f_values

fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
n_algs = len(scores)
x = np.arange(len(test_h_values))
bar_width = 0.8 / n_algs

for i, (name, iteration_counts) in enumerate(scores.items()):
    axs[0].bar(x + i * bar_width, iteration_counts, width=bar_width, label=name)

axs[0].set_yscale("log")
axs[0].set_ylabel("iteration count")
axs[0].set_title("Convergence speed per strategy")
axs[0].legend()

for i, (name, final_f_values) in enumerate(f_scores.items()):
    print(final_f_values)
    axs[1].bar(x + i * bar_width, final_f_values, width=bar_width, label=name)

axs[1].set_yscale("symlog", linthresh=1e-14)
axs[1].set_xticks(x + bar_width * (n_algs - 1) / 2)
axs[1].set_xticklabels([f"{v:.3g}" for v in test_h_values])
axs[1].set_xlabel("init_h_a")
axs[1].set_ylabel(r"$f(h_a)$")
axs[1].set_title("Residual at convergence per strategy")
axs[1].legend()

plt.show()
