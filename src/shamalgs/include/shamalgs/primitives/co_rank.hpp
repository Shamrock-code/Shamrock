// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file co_rank.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief GPU compatible Merge Path co-rank (partition search) for parallel merges
 */

#include "shambase/aliases_int.hpp"

namespace shamalgs::primitives {

    /**
     * @brief Merge Path co-rank of two sorted ranges
     *
     * Given two sorted ranges `A[0, m)` and `B[0, n)`, and an output index @p k in
     * `[0, m + n]`, returns the number of elements `i` taken from `A` when producing the
     * first @p k elements of `merge(A, B)`. The matching count in `B` is `k - i`.
     *
     * In other words this binary searches the anti-diagonal `i + j == k` of the merge matrix
     * for the split point such that `A[i-1] <= B[j]` and `B[j-1] < A[i]`, which lets each
     * thread of a parallel merge find the sub-ranges of `A` and `B` feeding its own chunk of
     * the output.
     *
     * Ties are resolved in favour of `A`, so a merge driven by this split is stable when the
     * inputs are.
     *
     * @tparam Tkey Key type - must be comparable (supports < and > operators)
     * @param k Output index to find the split for, in `[0, m + n]`
     * @param a Pointer to the first sorted range
     * @param m Length of the first sorted range
     * @param b Pointer to the second sorted range
     * @param n Length of the second sorted range
     * @return The rank `i` in `a`, with `i <= m` and `k - i <= n`
     *
     * @code
     * u32 i = shamalgs::primitives::co_rank(k, a, m, b, n);
     * u32 j = k - i; // a[0, i) and b[0, j) are the first k elements of the merge
     * @endcode
     */
    template<class Tkey>
    constexpr u32 co_rank(
        u32 k, const Tkey *__restrict__ a, u32 m, const Tkey *__restrict__ b, u32 n) {

        u32 i = (k < m) ? k : m;
        u32 j = k - i;

        u32 i_low = (k > n) ? (k - n) : 0;
        u32 j_low = (k > m) ? (k - m) : 0;

        while (true) {
            if (i > 0 && j < n && a[i - 1] > b[j]) {
                u32 delta = (i - i_low + 1) >> 1;
                j_low     = j;
                i -= delta;
                j += delta;
            } else if (j > 0 && i < m && b[j - 1] >= a[i]) {
                u32 delta = (j - j_low + 1) >> 1;
                i_low     = i;
                i += delta;
                j -= delta;
            } else {
                return i;
            }
        }
    }

} // namespace shamalgs::primitives
