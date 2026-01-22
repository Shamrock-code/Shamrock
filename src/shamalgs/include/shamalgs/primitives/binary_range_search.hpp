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
 * @file binary_range_search.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Binary range search algorithm
 */

#include "shambase/aliases_int.hpp"
#include "shambase/assert.hpp"
#include "shamalgs/primitives/lower_bound.hpp"
#include "shamalgs/primitives/upper_bound.hpp"

namespace shamalgs::primitives {

    /**
     * @brief Find the range of indices for which key[inf] <= value_min <= value_max <= key[sup]
     *
     * @pre key is sorted in ascending order
     * @pre value_min <= value_max
     * @pre first < last
     * @pre key[first] <= value_min
     * @pre key[last - 1] >= value_max

     * @tparam Tkey Type of the key
     * @param key Array of keys
     * @param first First index of the range
     * @param last Last index of the range
     * @param value_min Lower bound of the range
     * @param value_max Upper bound of the range
     * @param inf Index of the first element of the range
     * @param sup Index of the last element of the range
     *
     * @code{.cpp}
     * // Example usage with a sorted array
     * std::vector<int> keys = {1, 3, 5, 7, 9, 11, 13, 15};
     * u32 inf, sup;
     *
     * // Find range of elements between 5 and 11 (inclusive)
     * binary_range_search(keys.data(), 0, keys.size(), 5, 11, inf, sup);
     * // inf = 2, sup = 5 (elements 5, 7, 9, 11 at indices 2, 3, 4, 5)
     *
     * // Find range with duplicates
     * std::vector<int> keys2 = {1, 2, 2, 2, 3, 4, 4, 5};
     * binary_range_search(keys2.data(), 0, keys2.size(), 2, 4, inf, sup);
     * // inf = 1, sup = 6 (elements 2, 2, 2, 3, 4, 4 at indices 1-6)
     * @endcode
     */
    template<class Tkey>
    constexpr void binary_range_search(
        const Tkey *__restrict__ key,
        u32 first,
        u32 last,
        const Tkey &value_min,
        const Tkey &value_max,
        u32 &inf,
        u32 &sup) {

        SHAM_ASSERT(value_min <= value_max);
        SHAM_ASSERT((first < last) ? key[first] <= value_min : true);
        SHAM_ASSERT((first < last) ? key[last - 1] >= value_max : true);

        inf = binary_search_lower_bound(key, first, last, value_min);
        sup = binary_search_upper_bound(key, first, last, value_max);

        // std::lower_bound and std::upper_bound are quirky:
        //  - std::lower_bound returns the first such that >= value_min (searching 8 in [7,9] return
        //      1 where we want 0)
        //  - std::upper_bound returns the first such that > value_max (searching 8 in
        //      [7,8,9] return 2 where we want 1)
        // The following ajust that to the range with expect in this function
        if (inf > first && inf < last) {
            inf -= (key[inf] > value_min);
        }
        if (sup > first && sup <= last) {
            sup -= (key[sup - 1] == value_max);
        }
    }

} // namespace shamalgs::primitives
