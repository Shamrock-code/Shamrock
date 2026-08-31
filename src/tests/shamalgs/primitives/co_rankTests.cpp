// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamalgs/primitives/co_rank.hpp"
#include "shamcomm/logs.hpp"
#include "shamtest/shamtest.hpp"
#include <algorithm>
#include <string>
#include <vector>

namespace {

    /// Enumerate the sorted vectors of a given length over the alphabet [0, alphabet), by
    /// advancing `v` to the next one. Returns false once the last one has been reached.
    bool next_sorted(std::vector<i32> &v, i32 alphabet) {
        i32 len = static_cast<i32>(v.size());
        for (i32 p = len - 1; p >= 0; --p) {
            if (v[p] < alphabet - 1) {
                v[p]++;
                for (i32 q = p + 1; q < len; ++q) {
                    v[q] = v[p];
                }
                return true;
            }
        }
        return false;
    }

} // namespace

NEW_TEST(Unittest, "shamalgs/primitives/co_rank", 1) {

    // Exhaustive check over every pair of sorted ranges of length up to 6 drawn from a three
    // symbol alphabet, and every output index k. For each of them we check the Merge Path
    // invariants, and that splitting the merge at the returned co-rank reproduces the prefix
    // std::merge would have produced.

    constexpr i32 alphabet = 3;
    constexpr i32 max_len  = 6;

    u64 case_count      = 0;
    u64 bound_failures  = 0;
    u64 invariant_fails = 0;
    u64 prefix_failures = 0;
    std::string first_failure;

    for (i32 m = 0; m <= max_len; ++m) {
        for (i32 n = 0; n <= max_len; ++n) {

            std::vector<i32> a(m, 0);
            do {
                std::vector<i32> b(n, 0);
                do {
                    std::vector<i32> merged;
                    std::merge(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(merged));

                    for (i32 k = 0; k <= m + n; ++k) {
                        u32 i = shamalgs::primitives::co_rank<i32>(
                            static_cast<u32>(k),
                            a.data(),
                            static_cast<u32>(m),
                            b.data(),
                            static_cast<u32>(n));
                        case_count++;

                        bool bounds_ok = (i <= static_cast<u32>(m)) && (static_cast<u32>(k) >= i)
                                         && (static_cast<u32>(k) - i <= static_cast<u32>(n));
                        if (!bounds_ok) {
                            bound_failures++;
                            continue;
                        }

                        i32 j = k - static_cast<i32>(i);

                        // A[i-1] <= B[j] and B[j-1] < A[i], on the sides that exist
                        bool inv_ok
                            = (i == 0 || j == n || !(a[i - 1] > b[j]))
                              && (j == 0 || static_cast<i32>(i) == m || !(b[j - 1] >= a[i]));
                        if (!inv_ok) {
                            invariant_fails++;
                            if (first_failure.empty()) {
                                first_failure = "invariant broken at m=" + std::to_string(m) + " n="
                                                + std::to_string(n) + " k=" + std::to_string(k)
                                                + " i=" + std::to_string(i);
                            }
                        }

                        std::vector<i32> prefix;
                        std::merge(
                            a.begin(),
                            a.begin() + i,
                            b.begin(),
                            b.begin() + j,
                            std::back_inserter(prefix));

                        bool prefix_ok = (prefix.size() == static_cast<size_t>(k));
                        for (i32 t = 0; t < k && prefix_ok; ++t) {
                            prefix_ok = (prefix[t] == merged[t]);
                        }
                        if (!prefix_ok) {
                            prefix_failures++;
                            if (first_failure.empty()) {
                                first_failure = "prefix mismatch at m=" + std::to_string(m) + " n="
                                                + std::to_string(n) + " k=" + std::to_string(k)
                                                + " i=" + std::to_string(i);
                            }
                        }
                    }
                } while (n > 0 && next_sorted(b, alphabet));
            } while (m > 0 && next_sorted(a, alphabet));
        }
    }

    REQUIRE(case_count > 0);
    REQUIRE_EQUAL_NAMED("co_rank stays within both ranges", bound_failures, u64(0));
    REQUIRE_EQUAL_NAMED("co_rank satisfies the merge path invariants", invariant_fails, u64(0));
    REQUIRE_EQUAL_NAMED("co_rank split reproduces std::merge", prefix_failures, u64(0));

    if (!first_failure.empty()) {
        shamlog_info_ln("tests", "first co_rank failure :", first_failure);
    }
}
