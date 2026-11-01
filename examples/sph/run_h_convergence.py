"""
Showcase smoothing length iteration algorithlm
==============================================
"""

# sphinx_gallery_thumbnail_number = 2
# sphinx_gallery_multi_image = "single"

import sys

import matplotlib.pyplot as plt
import numpy as np
from shamrock.utils.numba_helper import maybe_njit

import shamrock


def W(r, h):
    return shamrock.math.sphkernel.M4_W3d(r, h)


def dhW(r, h):
    return shamrock.math.sphkernel.M4_dhW3d(r, h)


W = maybe_njit(W)
dhW = maybe_njit(dhW)

# W and dhW are bound to scalar C++ kernels (shamrock.math.sphkernel.*), so they
# can't be called directly on arrays: np.vectorize applies them elementwise.
W_vec = np.vectorize(W)
dhW_vec = np.vectorize(dhW)


def compute_sums(pmass, id_a, h_a, positions: np.ndarray):
    dr = positions[id_a, :] - positions
    rab = np.sqrt(np.sum(dr * dr, axis=1))

    rho_sum = pmass * np.sum(W_vec(rab, h_a))
    sumdWdh = pmass * np.sum(dhW_vec(rab, h_a))

    return rho_sum, sumdWdh


compute_sums = maybe_njit(compute_sums)


def count_neighbors(id_a, h_a, W, positions: np.ndarray):
    dr = positions[id_a, :] - positions
    rab = np.sqrt(np.sum(dr * dr, axis=1))

    return int(np.count_nonzero(np.vectorize(W)(rab, h_a) > 0))


count_neighbors = maybe_njit(count_neighbors)

Rkern = shamrock.math.sphkernel.M4_Rkern()


def next_neighbor_distance(id_a, h_a, positions: np.ndarray):
    """Distance to the closest particle outside of the kernel support (Rkern * h_a), i.e.
    the next one to become a neighbor when h_a grows. In a real kernel it is a min
    reduction in the same neighbor loop as the neighbor count."""
    dr = positions[id_a, :] - positions
    rab = np.sqrt(np.sum(dr * dr, axis=1))

    outside = rab[rab >= Rkern * h_a]
    return outside.min() if outside.size > 0 else np.inf


def get_nth_neigh_dist(id_a, n, positions: np.ndarray):
    """Distance of the n-th closest particle to id_a (1-indexed, id_a itself included at
    distance 0, as in count_neighbors), np.inf if there are less than n particles."""
    dr = positions[id_a, :] - positions
    rab = np.sort(np.sqrt(np.sum(dr * dr, axis=1)))

    return rab[n - 1] if n <= len(rab) else np.inf


def rho_h(m, h, hfact):
    return m * (hfact / h) * (hfact / h) * (hfact / h)


rho_h = maybe_njit(rho_h)

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


def newton_iterate_new_h(h_a, positions, cycle_state_vars: dict, outer_state_vars: dict):
    """One Newton-Raphson sweep, reproducing the per-particle branch of
    IterateSmoothingLengthDensity.cpp (src/shammodels/sph/src/modules/
    IterateSmoothingLengthDensity.cpp:52-119).

    cycle_state_vars["ha_0"] is h_old: the h value at the start of the current
    subcycle (reset by the caller each time sph_prestep would rebuild the
    ghost zone), NOT the previous Newton iterate.

    Returns (new_h, eps). eps == -1 is the sentinel the real kernel uses to
    mean "new_h would exceed ha_0 * h_evol_max (htol_up_coarse_cycle)": the
    caller must treat this as sph_prestep does and start a fresh subcycle.
    """
    ha_0 = cycle_state_vars["ha_0"]

    rho_ha = rho_h(pmass, h_a, hfact)
    rho_sum, sumdWdh = compute_sums(pmass, id_a, h_a, positions)
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


def newton_iterate_new_h_fixedp_fallback(
    h_a, positions, cycle_state_vars: dict, outer_state_vars: dict
):
    """newton_iterate_new_h with the fixed point fallback of
    newton_iterate_new_h_fallback (src/shammodels/sph/include/shammodels/sph/math/density.hpp).

    With df <= 0 the Newton step goes away from the root and h diverges. Fall
    back to the fixed point update h = hfact (m / rho_sum)^(1/3), i.e.
    h_a * (rho_ha / rho_sum)^(1/3), which moves towards it.
    """
    ha_0 = cycle_state_vars["ha_0"]

    rho_ha = rho_h(pmass, h_a, hfact)
    rho_sum, sumdWdh = compute_sums(pmass, id_a, h_a, positions)
    f_iter, df_iter = f_df(rho_ha, rho_sum, sumdWdh, h_a)

    if not (df_iter > 0):
        new_h = h_a * np.cbrt(rho_ha / rho_sum)
    else:
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


def newton_iterate_new_h_neigh_lim(
    h_a, positions, cycle_state_vars: dict, outer_state_vars: dict, trigger_threshold=500
):
    """One Newton-Raphson sweep with a neighbor-count safety limiter,
    reproducing the per-particle branch of IterateSmoothingLengthDensityNeighLim.cpp
    (src/shammodels/sph/src/modules/IterateSmoothingLengthDensityNeighLim.cpp:59-148).

    On top of newton_iterate_new_h's clamps, this adds two neighbor-count
    guards evaluated before any clamp is applied:
      - if h_a already has more than trigger_threshold neighbors, shrink h_a
        by h_evol_iter_max and report eps=0 (the caller treats eps < epsilon_h
        as converged, so this freezes the particle at the shrunk h_a for the
        rest of the subcycle).
      - if growing h_a up to h_evol_iter_max * h_a would push the neighbor
        count over trigger_threshold and the raw Newton step wants to grow
        h_a, leave h_a unchanged and also report eps=0.
    """
    ha_0 = cycle_state_vars["ha_0"]

    h_max_evol_m = 1.0 / h_evol_iter_max
    h_max_evol_p = h_evol_iter_max

    count_within = count_neighbors(id_a, h_a, W, positions)
    count_within_next = count_neighbors(id_a, h_a * h_max_evol_p, W, positions)

    rho_ha = rho_h(pmass, h_a, hfact)
    rho_sum, sumdWdh = compute_sums(pmass, id_a, h_a, positions)
    f_iter, df_iter = f_df(rho_ha, rho_sum, sumdWdh, h_a)
    new_h = h_a - f_iter / df_iter

    if count_within > trigger_threshold:
        return h_max_evol_m * h_a, 0.0

    if count_within_next > trigger_threshold and new_h > h_a:
        return h_a, 0.0

    # per-iteration clamp (htol_up_fine_cycle), relative to the previous iterate h_a
    new_h = max(new_h, h_a * h_max_evol_m)
    new_h = min(new_h, h_a * h_max_evol_p)

    # per-subcycle clamp (htol_up_coarse_cycle), relative to ha_0 (h at subcycle start)
    if new_h < ha_0 * h_evol_max:
        eps = abs(new_h - h_a) / ha_0
    else:
        new_h = ha_0 * h_evol_max
        eps = -1.0

    return new_h, eps


def newton_iterate_new_h_poslim(h_a, positions, cycle_state_vars: dict, outer_state_vars: dict):
    """
    Derivative based limiter on top of normal NR
    """
    ha_0 = cycle_state_vars["ha_0"]

    rho_ha = rho_h(pmass, h_a, hfact)
    rho_sum, sumdWdh = compute_sums(pmass, id_a, h_a, positions)
    f_iter, df_iter = f_df(rho_ha, rho_sum, sumdWdh, h_a)

    h_max_evol_m = 1.0 / h_evol_iter_max
    h_max_evol_p = h_evol_iter_max

    # Check if the state is potentially diverging, if yes go down until it converges again
    # (f > 0 and df <= 0, i.e. f / df < 0 without dividing by df)
    if f_iter > 0 and df_iter <= 0:
        new_h = h_a * h_max_evol_m
    else:
        new_h = h_a - f_iter / df_iter

    # per-iteration clamp (htol_up_fine_cycle), relative to the previous iterate h_a
    new_h = max(new_h, h_a * h_max_evol_m)
    new_h = min(new_h, h_a * h_max_evol_p)

    # per-subcycle clamp (htol_up_coarse_cycle), relative to ha_0 (h at subcycle start)
    if new_h < ha_0 * h_evol_max:
        eps = abs(new_h - h_a) / ha_0
    else:
        new_h = ha_0 * h_evol_max
        eps = -1.0

    return new_h, eps


def newton_iterate_new_h_neigh_lim_poslim(
    h_a, positions, cycle_state_vars: dict, outer_state_vars: dict, trigger_threshold=500
):
    """
    Hybrid of derivative limiter + neigh lim
    """
    ha_0 = cycle_state_vars["ha_0"]

    h_max_evol_m = 1.0 / h_evol_iter_max
    h_max_evol_p = h_evol_iter_max

    count_within = count_neighbors(id_a, h_a, W, positions)
    count_within_next = count_neighbors(id_a, h_a * h_max_evol_p, W, positions)

    rho_ha = rho_h(pmass, h_a, hfact)
    rho_sum, sumdWdh = compute_sums(pmass, id_a, h_a, positions)
    f_iter, df_iter = f_df(rho_ha, rho_sum, sumdWdh, h_a)

    # Check if the state is potentially diverging, if yes go down until it converges again
    # (f > 0 and df <= 0, i.e. f / df < 0 without dividing by df)
    if f_iter > 0 and df_iter <= 0:
        new_h = h_a * h_max_evol_m
    else:
        new_h = h_a - f_iter / df_iter

    # per-iteration clamp (htol_up_fine_cycle), relative to the previous iterate h_a
    new_h = max(new_h, h_a * h_max_evol_m)
    new_h = min(new_h, h_a * h_max_evol_p)

    if count_within > trigger_threshold:
        new_h = h_max_evol_m * h_a

    if count_within_next > trigger_threshold and new_h > h_a:
        return h_a, 0.0

    # per-subcycle clamp (htol_up_coarse_cycle), relative to ha_0 (h at subcycle start)
    if new_h < ha_0 * h_evol_max:
        eps = abs(new_h - h_a) / ha_0
    else:
        new_h = ha_0 * h_evol_max
        eps = -1.0

    return new_h, eps


def bisect_iterate_new_h(h_a, positions, cycle_state_vars: dict, outer_state_vars: dict):

    # Normally the allocation are done before but this simlulator only prepares ha_0
    if "lo" not in cycle_state_vars:
        cycle_state_vars["lo"] = 0
        cycle_state_vars["hi"] = np.inf

    h_max_evol_m = 1.0 / h_evol_iter_max
    h_max_evol_p = h_evol_iter_max
    ha_0 = cycle_state_vars["ha_0"]
    lo = cycle_state_vars["lo"]
    hi = cycle_state_vars["hi"]

    rho_ha = rho_h(pmass, h_a, hfact)
    rho_sum, sumdWdh = compute_sums(pmass, id_a, h_a, positions)
    f_iter, df_iter = f_df(rho_ha, rho_sum, sumdWdh, h_a)

    if f_iter < 0:
        lo = h_a
    else:
        hi = h_a

    new_h = (lo + hi) / 2

    # per-iteration clamp (htol_up_fine_cycle), relative to the previous iterate h_a
    new_h = max(new_h, h_a * h_max_evol_m)
    new_h = min(new_h, h_a * h_max_evol_p)

    cycle_state_vars["lo"] = lo
    cycle_state_vars["hi"] = hi

    # per-subcycle clamp (htol_up_coarse_cycle), relative to ha_0 (h at subcycle start)
    if new_h < ha_0 * h_evol_max:
        eps = abs(new_h - h_a) / ha_0
    else:
        new_h = ha_0 * h_evol_max
        eps = -1.0

    return new_h, eps


def bisect_iterate_new_h_neigh_lim(h_a, positions, cycle_state_vars: dict, outer_state_vars: dict):

    # Normally the allocation are done before but this simlulator only prepares ha_0
    if "lo" not in cycle_state_vars:
        cycle_state_vars["lo"] = 0
        cycle_state_vars["hi"] = np.inf

    h_max_evol_m = 0.5
    h_max_evol_p = h_evol_iter_max
    ha_0 = cycle_state_vars["ha_0"]
    lo = cycle_state_vars["lo"]
    hi = cycle_state_vars["hi"]

    rho_ha = rho_h(pmass, h_a, hfact)
    rho_sum, sumdWdh = compute_sums(pmass, id_a, h_a, positions)
    count_within = count_neighbors(id_a, h_a, W, positions)
    f_iter, df_iter = f_df(rho_ha, rho_sum, sumdWdh, h_a)

    is_upper_bound = f_iter > 0 or count_within > 500

    if is_upper_bound:
        hi = h_a
    else:
        lo = h_a

    new_h = (lo + hi) / 2

    # per-iteration clamp (htol_up_fine_cycle), relative to the previous iterate h_a
    new_h = max(new_h, h_a * h_max_evol_m)
    new_h = min(new_h, h_a * h_max_evol_p)

    cycle_state_vars["lo"] = lo
    cycle_state_vars["hi"] = hi

    # per-subcycle clamp (htol_up_coarse_cycle), relative to ha_0 (h at subcycle start)
    if new_h < ha_0 * h_evol_max:
        eps = abs(new_h - h_a) / ha_0
    else:
        new_h = ha_0 * h_evol_max
        eps = -1.0

    return new_h, eps


def bisect_NR_iterate_new_h(h_a, positions, cycle_state_vars: dict, outer_state_vars: dict):

    # Normally the allocation are done before but this simlulator only prepares ha_0
    if "lo" not in outer_state_vars:
        outer_state_vars["lo"] = 0
        outer_state_vars["hi"] = np.inf

    h_max_evol_m = 0.5
    h_max_evol_p = h_evol_iter_max
    ha_0 = cycle_state_vars["ha_0"]
    lo = outer_state_vars["lo"]
    hi = outer_state_vars["hi"]

    h_a_next_min = h_a * h_max_evol_m
    h_a_next_max = h_a * h_max_evol_p

    rho_ha = rho_h(pmass, h_a, hfact)
    rho_sum, sumdWdh = compute_sums(pmass, id_a, h_a, positions)
    f_iter, df_iter = f_df(rho_ha, rho_sum, sumdWdh, h_a)
    new_h_nr = h_a - f_iter / df_iter

    accept_nr = max(lo, h_a_next_min) < new_h_nr < min(hi, h_a_next_max)

    if f_iter < 0:
        lo = max(h_a, lo)
    else:
        hi = min(h_a, hi)

    if accept_nr:
        new_h = new_h_nr
    else:
        new_h = (lo + hi) / 2

    # per-iteration clamp (htol_up_fine_cycle), relative to the previous iterate h_a
    new_h = max(new_h, h_a_next_min)
    new_h = min(new_h, h_a_next_max)

    outer_state_vars["lo"] = lo
    outer_state_vars["hi"] = hi

    # per-subcycle clamp (htol_up_coarse_cycle), relative to ha_0 (h at subcycle start)
    if new_h < ha_0 * h_evol_max:
        eps = abs(new_h - h_a) / ha_0
    else:
        new_h = ha_0 * h_evol_max
        eps = -1.0

    return new_h, eps


def bisect_NR_iterate_new_h_neigh_lim(
    h_a, positions, cycle_state_vars: dict, outer_state_vars: dict
):

    # Normally the allocation are done before but this simlulator only prepares ha_0
    if "lo" not in outer_state_vars:
        outer_state_vars["lo"] = 0
        outer_state_vars["hi"] = np.inf

    h_max_evol_m = 1.0 / h_evol_iter_max
    h_max_evol_p = h_evol_iter_max
    ha_0 = cycle_state_vars["ha_0"]
    lo = outer_state_vars["lo"]
    hi = outer_state_vars["hi"]

    h_a_next_min = h_a * h_max_evol_m
    h_a_next_max = h_a * h_max_evol_p

    rho_ha = rho_h(pmass, h_a, hfact)
    rho_sum, sumdWdh = compute_sums(pmass, id_a, h_a, positions)
    count_within = count_neighbors(id_a, h_a, W, positions)
    count_within_tol = count_neighbors(id_a, min(h_a * (1 + 10 * epsilon_h), hi), W, positions)
    f_iter, df_iter = f_df(rho_ha, rho_sum, sumdWdh, h_a)

    print(h_a, count_within, count_within_tol)

    # f wants a larger h, but h_a already sits at the neighbor limit up to the convergence
    # tolerance (within the limit at h_a, above it at h_a * (1 + epsilon_h)): h_a is the
    # neighbor limited solution, stop there
    if f_iter < 0 and count_within <= 500 < count_within_tol:
        return h_a, 0.0

    new_h_nr = h_a - f_iter / df_iter

    accept_nr = max(lo, h_a_next_min) < new_h_nr < min(hi, h_a_next_max) and count_within < 500

    is_upper_bound = f_iter > 0 or count_within > 500

    if is_upper_bound:
        hi = h_a
    else:
        lo = h_a

    if accept_nr:
        new_h = new_h_nr
    else:
        new_h = (lo + hi) / 2

    print(f"h_a = {h_a} lo = {lo} hi = {hi} new_h_nr = {new_h_nr} count_within = {count_within}")

    # per-iteration clamp (htol_up_fine_cycle), relative to the previous iterate h_a
    new_h = max(new_h, h_a_next_min)
    new_h = min(new_h, h_a_next_max)

    outer_state_vars["lo"] = lo
    outer_state_vars["hi"] = hi

    # per-subcycle clamp (htol_up_coarse_cycle), relative to ha_0 (h at subcycle start)
    if new_h < ha_0 * h_evol_max:
        eps = abs(new_h - h_a) / ha_0
    else:
        new_h = ha_0 * h_evol_max
        eps = -1.0

    return new_h, eps


def bisect_NR_iterate_new_h_neigh_lim_direct(
    h_a, positions, cycle_state_vars: dict, outer_state_vars: dict
):
    """Bisection + NR with the neighbor limit applied as a ceiling on h instead of counting
    neighbors: the largest h with <= 500 neighbors is h_lim = r_501 / Rkern (r_501 the
    distance of the 501st closest particle, see get_nth_neigh_dist), and no new h may exceed
    it. If the limit is active (f < 0 at h_lim), the next iteration caps at h_lim again and
    stops there; otherwise f > 0 at h_lim and the root below it is searched as usual."""

    # Normally the allocation are done before but this simlulator only prepares ha_0
    if "lo" not in outer_state_vars:
        outer_state_vars["lo"] = 0
        outer_state_vars["hi"] = np.inf

    h_max_evol_m = 0.5
    h_max_evol_p = h_evol_iter_max
    ha_0 = cycle_state_vars["ha_0"]
    lo = outer_state_vars["lo"]
    hi = outer_state_vars["hi"]

    h_a_next_min = h_a * h_max_evol_m
    h_a_next_max = h_a * h_max_evol_p

    # ceiling from the neighbor limit
    h_lim = get_nth_neigh_dist(id_a, 501, positions) / Rkern

    rho_ha = rho_h(pmass, h_a, hfact)
    rho_sum, sumdWdh = compute_sums(pmass, id_a, h_a, positions)
    f_iter, df_iter = f_df(rho_ha, rho_sum, sumdWdh, h_a)
    new_h_nr = h_a - f_iter / df_iter

    accept_nr = max(lo, h_a_next_min) < new_h_nr < min(hi, h_a_next_max)

    if f_iter < 0:
        lo = max(h_a, lo)
    else:
        hi = min(h_a, hi)

    if accept_nr:
        new_h = new_h_nr
    else:
        new_h = (lo + hi) / 2

    # never exceed the neighbor limit
    new_h = min(new_h, h_lim)

    # per-iteration clamp (htol_up_fine_cycle), relative to the previous iterate h_a
    new_h = max(new_h, h_a_next_min)
    new_h = min(new_h, h_a_next_max)

    outer_state_vars["lo"] = lo
    outer_state_vars["hi"] = hi

    # per-subcycle clamp (htol_up_coarse_cycle), relative to ha_0 (h at subcycle start)
    if new_h < ha_0 * h_evol_max:
        eps = abs(new_h - h_a) / ha_0
    else:
        new_h = ha_0 * h_evol_max
        eps = -1.0

    return new_h, eps


algs = {
    "Newton": newton_iterate_new_h,
    "Newton (neigh lim)": newton_iterate_new_h_neigh_lim,
    "Newton (fixed point fallback)": newton_iterate_new_h_fixedp_fallback,
    "Newton (ceil)": newton_iterate_new_h_poslim,
    # "Newton (neigh lim + ceil)": newton_iterate_new_h_neigh_lim_poslim,
    "Bisection": bisect_iterate_new_h,
    # "Bisection (neigh_lim)": bisect_iterate_new_h_neigh_lim,
    "Bisection + NR": bisect_NR_iterate_new_h,
    "Bisection + NR (neigh lim)": bisect_NR_iterate_new_h_neigh_lim,
    "Bisection + NR (neigh lim, direct)": bisect_NR_iterate_new_h_neigh_lim_direct,
}


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

    outer_state_vars = {}

    # outer loop: sph_prestep's ghost-zone-rebuild subcycle
    # (src/shammodels/sph/src/Solver.cpp:1235, hstep_cnt < h_max_subcycles_count)
    for hstep_cnt in range(h_max_subcycles_count):
        # each subcycle resets h_old to the current h (Solver.cpp:1245)
        cycle_state_vars = {"ha_0": h_a}

        # inner loop: LoopSmoothingLengthIter's Newton sweep count
        # (LoopSmoothingLengthIter.cpp:31, iter_h < h_iter_per_subcycles)
        for iter_h in range(h_iter_per_subcycles):
            h_a, eps = iterate_new_h(h_a, positions, cycle_state_vars, outer_state_vars)
            history_h_a.append(h_a)

            rho_ha = rho_h(pmass, h_a, hfact)
            rho_sum, sumdWdh = compute_sums(pmass, id_a, h_a, positions)
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
        ha_0 = cycle_state_vars["ha_0"]
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

    candidates = [entry for entry in histories if entry[1][-1] < 10]
    best_entry = min(candidates, key=lambda entry: np.abs(entry[2][-1]))
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

    final_neigh_counts = [
        history_neigh_count[-1] for _, _, _, history_neigh_count, _, _ in histories
    ]

    # number of outer ghost-zone-rebuild subcycles run (each one is an MPI cycle)
    subcycle_counts = [
        len(subcycle_end_indices) for _, _, _, _, _, subcycle_end_indices in histories
    ]

    return (
        histories,
        found_h_a,
        iteration_counts,
        final_f_values,
        final_neigh_counts,
        subcycle_counts,
    )


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
        rho_sum, sumdWdh = compute_sums(pmass, id_a, h_a_test[i], positions)
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


def compare_algs_h_convergence(test_h_values, algs):
    """Plot the convergence of each strategy and their comparison, and return
    the histories of each strategy (see analyse_h_convergence)."""
    results = {}
    histories_by_alg = {}
    for name, alg in algs.items():
        fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        fig.suptitle(name)

        (
            histories,
            found_h_a,
            iteration_counts,
            final_f_values,
            final_neigh_counts,
            subcycle_counts,
        ) = analyse_h_convergence(positions, id_a, pmass, alg, test_h_values)

        plot_h_convergence(histories, found_h_a, axs)
        histories_by_alg[name] = histories

        # histories may hold one more entry than test_h_values (the extra run
        # seeded at found_h_a), so derive the x-axis from histories itself
        init_h_a_values = [entry[0] for entry in histories]
        results[name] = (
            init_h_a_values,
            iteration_counts,
            final_f_values,
            final_neigh_counts,
            subcycle_counts,
        )

        plt.tight_layout()

    fig, axs = plt.subplots(4, 1, figsize=(10, 16))
    fig.suptitle("Algorithm comparison")

    bar_width = 0.8 / len(results)
    for i, (
        name,
        (init_h_a_values, iteration_counts, final_f_values, final_neigh_counts, subcycle_counts),
    ) in enumerate(results.items()):
        x = np.arange(len(init_h_a_values)) + i * bar_width
        axs[0].bar(x, iteration_counts, width=bar_width, label=name)
        axs[1].bar(x, subcycle_counts, width=bar_width, label=name)
        axs[2].bar(x, final_f_values, width=bar_width, label=name)
        axs[3].bar(x, final_neigh_counts, width=bar_width, label=name)

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

    axs[1].set_yscale("log")
    axs[1].set_xticks(xticks)
    axs[1].set_xticklabels(xticklabels)
    axs[1].set_xlabel("init_h_a")
    axs[1].set_ylabel("subcycle count")
    axs[1].set_title("MPI cycles (ghost-zone-rebuild subcycles)")
    axs[1].legend()

    axs[2].set_yscale("symlog", linthresh=1e-14)
    axs[2].set_xticks(xticks)
    axs[2].set_xticklabels(xticklabels)
    axs[2].set_xlabel("init_h_a")
    axs[2].set_ylabel(r"$f(h_a)$")
    axs[2].set_title("Residual at convergence")
    axs[2].legend()

    axs[3].set_yscale("log")
    axs[3].set_xticks(xticks)
    axs[3].set_xticklabels(xticklabels)
    axs[3].set_xlabel("init_h_a")
    axs[3].set_ylabel("neighbor count")
    axs[3].set_title("Neighbor count at convergence")
    axs[3].legend()

    plt.tight_layout()

    return histories_by_alg


def find_root_h(h_a_test):
    """Reference root of f, independent of the strategies: first sign change of f
    from negative to positive on h_a_test, refined by bisection."""

    def f_of(h):
        rho_sum, sumdWdh = compute_sums(pmass, id_a, h, positions)
        return rho_sum - rho_h(pmass, h, hfact)

    f_values = [f_of(h) for h in h_a_test]
    i = next(i for i in range(len(h_a_test) - 1) if f_values[i] < 0 <= f_values[i + 1])

    lo, hi = h_a_test[i], h_a_test[i + 1]
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if f_of(mid) < 0:
            lo = mid
        else:
            hi = mid

    return 0.5 * (lo + hi)


def find_neigh_limit_h(h_a_test, limit):
    """Largest h whose neighbor count stays <= limit: first h of h_a_test exceeding
    the limit, refined by bisection on the neighbor count."""

    counts = [count_neighbors(id_a, h, W, positions) for h in h_a_test]
    i = next((i for i in range(len(h_a_test)) if counts[i] > limit), None)
    if i is None:
        return np.inf

    lo, hi = (h_a_test[i - 1] if i > 0 else 0.0), h_a_test[i]
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if count_neighbors(id_a, mid, W, positions) <= limit:
            lo = mid
        else:
            hi = mid

    return lo


# state and results of each setup, summarised in the last section
setups = {}


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


def generate_random_distrib(Nside, seed=42):
    rng = np.random.default_rng(seed)
    positions = []

    id_a = 0
    for ix in range(Nside):
        for iy in range(Nside):
            for iz in range(Nside):
                positions.append(rng.random(3))

                if ix == 10 and iy == 10 and iz == 10:
                    id_a = len(positions) - 1

    positions = np.array(positions)

    return positions, id_a


def generate_random_distrib_giantpart(Nside, seed=42):
    rng = np.random.default_rng(seed)
    positions = []

    id_a = 0
    for ix in range(Nside):
        for iy in range(Nside):
            for iz in range(Nside):
                positions.append(rng.random(3))

    positions.append((2, 0, 0))
    id_a = len(positions) - 1

    positions = np.array(positions)

    return positions, id_a


def generate_isolated_part_small_cluster(Ncluster, seed=42):
    # particle a at the origin, with only a tight cluster of Ncluster particles at distance 1
    rng = np.random.default_rng(seed)
    positions = [(0, 0, 0)]
    id_a = 0

    for i in range(Ncluster):
        positions.append(0.01 * rng.random(3) + np.array([1, 0, 0]))

    positions = np.array(positions)

    return positions, id_a


# %%

plot_f_df_kernel()

# %%
# Cubic distrib
# -------------

positions, id_a = generate_cubic_distrib(Nside=10)
pmass = 1.0 / 1000.0

# %%

h_a_test = np.logspace(-3, 2, 1000)

plot_rho_f_df(h_a_test)

# %%

# sample 10 equally spaced values in h_a_test indexes
test_h_values = np.logspace(-3, 2, 10)

histories_by_alg = compare_algs_h_convergence(test_h_values, algs)
setups["Cubic distrib"] = (positions, id_a, pmass, test_h_values, histories_by_alg)

plt.show()

# %%
# Random distrib
# --------------

positions, id_a = generate_random_distrib(Nside=10)
pmass = 1.0 / 1000.0

# %%

h_a_test = np.logspace(-3, 2, 1000)

plot_rho_f_df(h_a_test)

# %%

# sample 10 equally spaced values in h_a_test indexes
test_h_values = np.logspace(-3, 2, 10)

histories_by_alg = compare_algs_h_convergence(test_h_values, algs)
setups["Random distrib"] = (positions, id_a, pmass, test_h_values, histories_by_alg)

plt.show()


# %%
# Random distrib (giant particle)
# --------------------------------

positions, id_a = generate_random_distrib_giantpart(Nside=20)
pmass = 1.0 / 1000.0

# %%

h_a_test = np.logspace(-3, 2, 1000)

plot_rho_f_df(h_a_test)

# %%

# sample 10 equally spaced values in h_a_test indexes
test_h_values = np.logspace(-3, 2, 10)

histories_by_alg = compare_algs_h_convergence(test_h_values, algs)
setups["Random distrib (giant particle)"] = (
    positions,
    id_a,
    pmass,
    test_h_values,
    histories_by_alg,
)

plt.show()


# %%
# Isolated particle with a small cluster
# --------------------------------------
# With h well above the cluster distance, all neighbours sit near q = 0, so
# f = rho_sum - rho_ha > 0 and df = -3 f / h < 0: the Newton step goes the wrong way.
# Both "Newton (ceil)" and "Newton (fixed point fallback)" then shrink h, but with
# rho_sum / rho_ha ~ (N + 1) / (pi hfact^3) the fixed point step cbrt(rho_ha / rho_sum)
# is only smaller than 1 / h_evol_iter_max when there are very few neighbours
# (N + 1 < ~7). This is the regime where the two strategies differ.

positions, id_a = generate_isolated_part_small_cluster(Ncluster=5)
pmass = 1.0 / 1000.0

# %%

h_a_test = np.logspace(-3, 2, 1000)

plot_rho_f_df(h_a_test)

# %%

# sample 10 equally spaced values in h_a_test indexes
test_h_values = np.logspace(-3, 2, 10)

histories_by_alg = compare_algs_h_convergence(test_h_values, algs)
setups["Isolated particle with a small cluster"] = (
    positions,
    id_a,
    pmass,
    test_h_values,
    histories_by_alg,
)

plt.show()


# %%
# Summary
# -------
# For each strategy, number of (setup, initial h) runs satisfying each property:
#
# - converges: the run reports convergence before reaching the subcycle limit;
# - correct h: the run converges and ends at the expected h (relative error < 1e-4):
#   the root of f (found by bisection) or, for the neighbor limiting strategies, the
#   largest h whose neighbor count stays <= 500 if it is smaller than the root;
# - single step if converged: started exactly at the expected h, the run converges after
#   a single iteration and stays there (one run per setup);
# - neigh <= 500 (+5%): the final neighbor count stays below the 500 neighbors limit
#   (count > 500 is the trigger of the neighbor limiters), with a 5% tolerance since the
#   returned h is not checked against the limit and can land just above it.
#
# Two more tables give the average cost (iterations and MPI cycles, i.e. ghost-zone
# rebuild subcycles) of the runs that found the correct h (as in the "correct h" column),
# split by initial h: above the root, close to it (h_ref * close_factors) and below it.
# The first one covers the setups where the root has at most 500 neighbors (the neighbor
# limit is not active), the second one the setups where it has more (giant particle).
# "(k/n)" means that only k of the n runs of that group found the correct h.

neigh_limit = 500
neigh_tol = 0.05
close_factors = (0.95, 1.0, 1.05)
groups = ["h0 > h*", "h0 ~ h*", "h0 < h*"]

# strategies whose expected h is capped by the neighbor limit
neigh_limited_algs = {
    newton_iterate_new_h_neigh_lim,
    newton_iterate_new_h_neigh_lim_poslim,
    bisect_iterate_new_h_neigh_lim,
    bisect_NR_iterate_new_h_neigh_lim,
    bisect_NR_iterate_new_h_neigh_lim_direct,
}

criteria = [
    "converges",
    "correct h",
    "single step if converged",
    f"neigh <= {neigh_limit} (+{neigh_tol:.0%})",
]
summary = {name: [0, 0, 0, 0] for name in algs}
totals = [0, 0, 0, 0]

# (iterations, MPI cycles) of each run, None if it did not find the correct h
# per setup category: is the neighbor limit active at the root of f
categories = {
    False: f"setups without neighbor limiting (<= {neigh_limit} neighbors at the root)",
    True: f"setups with neighbor limiting (> {neigh_limit} neighbors at the root)",
}
costs = {
    limited: {name: {group: [] for group in groups} for name in algs} for limited in categories
}

for setup_name, (positions, id_a, pmass, test_h_values, histories_by_alg) in setups.items():
    h_ref = find_root_h(np.logspace(-3, 2, 200))
    h_neigh = find_neigh_limit_h(np.logspace(-3, 2, 200), neigh_limit)
    print(
        f"{setup_name}: reference root h = {h_ref:.6g}, "
        f"largest h with <= {neigh_limit} neighbors = {h_neigh:.6g}"
    )
    setup_costs = costs[h_neigh < h_ref]

    # drop the extra run that analyse_h_convergence seeds at the found root
    is_test_run = lambda entry: np.any(entry[0] == test_h_values)
    n_runs = sum(is_test_run(entry) for entry in next(iter(histories_by_alg.values())))
    totals[0] += n_runs
    totals[1] += n_runs
    totals[2] += 1
    totals[3] += n_runs

    for name, alg in algs.items():
        h_target = min(h_ref, h_neigh) if alg in neigh_limited_algs else h_ref

        for entry in filter(is_test_run, histories_by_alg[name]):
            (
                init_h_a,
                history_h_a,
                history_f,
                history_neigh_count,
                converged,
                subcycle_end_indices,
            ) = entry
            group = "h0 > h*" if init_h_a > h_ref else "h0 < h*"
            setup_costs[name][group].append(
                (len(history_h_a) - 1, len(subcycle_end_indices))
                if converged and abs(history_h_a[-1] / h_target - 1) < 1e-4
                else None
            )
            if converged:
                summary[name][0] += 1
            if converged and abs(history_h_a[-1] / h_target - 1) < 1e-4:
                summary[name][1] += 1
            if history_neigh_count[-1] <= neigh_limit * (1 + neigh_tol):
                summary[name][3] += 1

        for factor in close_factors:
            history_h_a, _, _, _, converged, subcycle_end_indices = simulate_h_iter(
                h_ref * factor, positions, id_a, pmass, alg
            )
            setup_costs[name]["h0 ~ h*"].append(
                (len(history_h_a) - 1, len(subcycle_end_indices))
                if converged and abs(history_h_a[-1] / h_target - 1) < 1e-4
                else None
            )

        history_h_a, _, _, _, converged, _ = simulate_h_iter(h_target, positions, id_a, pmass, alg)
        if converged and len(history_h_a) - 1 == 1 and abs(history_h_a[-1] / h_target - 1) < 1e-4:
            summary[name][2] += 1


# ANSI colors only in a terminal (sphinx-gallery strips them from the captured output anyway)
use_colors = sys.stdout.isatty()


ansi_green_bold = "\033[1;32m"
ansi_bold = "\033[1m"
ansi_orange = "\033[38;5;208m"
ansi_red = "\033[31m"
ansi_reset = "\033[0m"


def rank_styles(scores, higher_is_better):
    """ANSI style of each cell (row, col), ranked per column against the best value
    with r = value / best (best / value if lower is better):
    best (ties included) = green bold, r >= 0.9 = bold, worst (if r < 0.9) = red,
    r < 0.5 = orange. scores[row][col] is None when the cell is not ranked."""
    styles = {}
    for col in range(len(scores[0])):
        ranked = {i: row[col] for i, row in enumerate(scores) if row[col] is not None}
        if not ranked:
            continue
        best = max(ranked.values()) if higher_is_better else min(ranked.values())
        worst = min(ranked.values()) if higher_is_better else max(ranked.values())

        for i, value in ranked.items():
            if value == best:
                styles[(i, col)] = ansi_green_bold
                continue

            if higher_is_better:
                r = value / best
            else:
                r = best / value

            if r >= 0.9:
                styles[(i, col)] = ansi_bold
            elif value == worst:
                styles[(i, col)] = ansi_red
            elif r < 0.5:
                styles[(i, col)] = ansi_orange
    return styles


def print_table(header, rows, styles):
    """Print a table with box-drawing borders, each column sized to its content.
    The first column is left aligned, the others are centered. styles maps a cell
    (row, col), col counted without the first column, to its ANSI style, applied when
    colors are enabled."""
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]

    def line(left, mid, right):
        return left + mid.join("\u2500" * (w + 2) for w in widths) + right

    def row(cells, i_row=None):
        first = f" {cells[0]:<{widths[0]}} "
        others = []
        for col, (c, w) in enumerate(zip(cells[1:], widths[1:])):
            cell = f" {c!s:^{w}} "
            if use_colors and (i_row, col) in styles:
                cell = f"{styles[(i_row, col)]}{cell}{ansi_reset}"
            others.append(cell)
        return "\u2502" + "\u2502".join([first] + others) + "\u2502"

    print(line("\u250c", "\u252c", "\u2510"))
    print(row(header))
    print(line("\u251c", "\u253c", "\u2524"))
    for i_row, r in enumerate(rows):
        print(row(r, i_row))
    print(line("\u2514", "\u2534", "\u2518"))


# %%

# higher count is better
print()
print_table(
    ["strategy"] + criteria,
    [[name] + [f"{n}/{t}" for n, t in zip(counts, totals)] for name, counts in summary.items()],
    rank_styles([counts for counts in summary.values()], higher_is_better=True),
)


def format_mean(runs, i):
    """Mean of the i-th cost over the runs that found the correct h, with (k/n) if some
    did not"""
    values = [run[i] for run in runs if run is not None]
    cell = f"{np.mean(values):.1f}" if values else "-"
    if len(values) < len(runs):
        cell += f" ({len(values)}/{len(runs)})"
    return cell


def mean_score(runs, i):
    """Score of the i-th cost for rank_styles: the mean as displayed, only if every run
    found the correct h (a mean over the few successful runs would be misleading)"""
    if any(run is None for run in runs):
        return None
    return round(np.mean([run[i] for run in runs]), 1)


# lower mean is better, among the strategies where all the runs found the correct h
cost_columns = [(g, 0) for g in groups] + [(g, 1) for g in groups]
for limited, title in categories.items():
    category_costs = costs[limited]
    print()
    print(f"Average cost, {title}:")
    print_table(
        ["strategy"] + [f"steps {g}" for g in groups] + [f"MPI cycles {g}" for g in groups],
        [
            [name] + [format_mean(category_costs[name][g], i) for g, i in cost_columns]
            for name in algs
        ],
        rank_styles(
            [[mean_score(category_costs[name][g], i) for g, i in cost_columns] for name in algs],
            higher_is_better=False,
        ),
    )
