// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file alg_primitives.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/aliases_int.hpp"
#include <utility>

namespace shambase {

    /**
     * @brief Simple insertion sort on pointer range
     *
     * @tparam T Element type
     * @tparam Comp Comparator type
     * @param data Pointer to data array
     * @param start Starting index (inclusive)
     * @param end Ending index (exclusive)
     * @param comp Comparison function
     */
    template<class T, class Comp>
    inline void ptr_insert_sort(T *data, u32 start, u32 end, Comp &&comp) {
        for (u32 i = start + 1; i < end; ++i) {
            auto key = data[i];
            u32 j    = i;
            while (j > start && comp(key, data[j - 1])) {
                data[j] = data[j - 1];
                --j;
            }
            data[j] = key;
        }
    };

    template<int I, int ArrSize>
    struct OddEvenTransposeSortT {
        template<typename K, typename Comp>
        inline static void Sort(K *keys, const u8 *segment_boundary, Comp comp) {
#pragma unroll
            for (int i = 1 & I; i < ArrSize - 1; i += 2)
                if (!segment_boundary[i] && comp(keys[i + 1], keys[i])) {
                    std::swap(keys[i], keys[i + 1]);
                }
            OddEvenTransposeSortT<I + 1, ArrSize>::Sort(keys, segment_boundary, comp);
        }
    };

    template<int I>
    struct OddEvenTransposeSortT<I, I> {
        template<typename K, typename Comp>
        inline static void Sort(K *keys, const u8 *segment_boundary, Comp comp) {}
    };

    /**
     * @brief Odd-even transpose sort with segment boundaries
     *
     * Sorts array while respecting segment boundaries where comparisons are disabled.
     *
     * @tparam T Element type
     * @tparam ArrSize Compile-time array size
     * @tparam Comp Comparator type
     * @param data Pointer to data array
     * @param segment_boundary Flags indicating segment boundaries (1 = boundary, 0 = no boundary)
     * @param comp Comparison function
     */
    template<class T, int ArrSize, class Comp>
    inline void odd_even_transpose_sort_segment_flags(
        T *data, const u8 *segment_boundary, Comp comp) {
        OddEvenTransposeSortT<0, ArrSize>::Sort(data, segment_boundary, comp);
    }

} // namespace shambase
