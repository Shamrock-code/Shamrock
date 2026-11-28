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
 * @file tie_order_utils.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Helpers making shamtree tests independent of the order of equal morton codes.
 *
 * The order of objects sharing the same morton code depends on the sort implementation (the
 * bitonic sort is unstable, the LSD radix sort is stable), so tests compare index maps and
 * neighbour lists after putting them in a canonical order.
 */

#include "shambase/aliases_int.hpp"
#include <algorithm>
#include <vector>

namespace shamtree::test_utils {

    /**
     * @brief Sort the object ids of `index_map` within each run of equal keys in `sorted_keys`
     *
     * @param index_map map from sorted morton id to object id
     * @param sorted_keys the sorted morton codes (same length as `index_map`)
     * @return `index_map` with ties in increasing object id order
     */
    template<class Tkey>
    inline std::vector<u32> sort_ties(
        std::vector<u32> index_map, const std::vector<Tkey> &sorted_keys) {
        size_t len = std::min(index_map.size(), sorted_keys.size());
        size_t i   = 0;
        while (i < len) {
            size_t j = i + 1;
            while (j < len && sorted_keys[j] == sorted_keys[i]) {
                j++;
            }
            std::sort(index_map.begin() + i, index_map.begin() + j);
            i = j;
        }
        return index_map;
    }

    /**
     * @brief Sort each segment of a list made of consecutive per-object segments
     *
     * @param list concatenation of the segments (e.g. per-object neighbour lists)
     * @param counts length of each segment
     * @return `list` with every segment sorted
     */
    inline std::vector<u32> sort_segments(std::vector<u32> list, const std::vector<u32> &counts) {
        size_t offset = 0;
        for (u32 cnt : counts) {
            size_t end = std::min(offset + cnt, list.size());
            std::sort(list.begin() + offset, list.begin() + end);
            offset = end;
        }
        return list;
    }

} // namespace shamtree::test_utils
