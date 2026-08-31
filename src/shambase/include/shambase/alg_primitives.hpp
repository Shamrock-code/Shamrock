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

    template<int I, int ArrSize>
    struct OddEvenTransposeSortByKeyT {
        template<typename Tkey, typename Tval, typename Comp>
        inline static void Sort(Tkey *keys, Tval *vals, Comp comp) {
#pragma unroll
            for (int i = 1 & I; i < ArrSize - 1; i += 2)
                if (comp(keys[i + 1], keys[i])) {
                    std::swap(keys[i], keys[i + 1]);
                    std::swap(vals[i], vals[i + 1]);
                }
            OddEvenTransposeSortByKeyT<I + 1, ArrSize>::Sort(keys, vals, comp);
        }
    };

    template<int I>
    struct OddEvenTransposeSortByKeyT<I, I> {
        template<typename Tkey, typename Tval, typename Comp>
        inline static void Sort(Tkey *keys, Tval *vals, Comp comp) {}
    };

    /**
     * @brief Odd-even transpose sort of a key/value pair of arrays
     *
     * Sorts @p keys in place, applying the same permutation to @p vals. Both loops of the
     * network are unrolled at compile time, so the arrays are expected to be thread-private
     * (registers) rather than global memory.
     *
     * The sort is stable as long as @p comp is a strict ordering (it only swaps when comp
     * reports the later element as strictly smaller).
     *
     * @tparam ArrSize Compile-time array size
     * @tparam Tkey Key type
     * @tparam Tval Value type
     * @tparam Comp Comparator type
     * @param keys Pointer to the keys, sorted in place
     * @param vals Pointer to the values, permuted alongside the keys
     * @param comp Comparison function on the keys
     */
    template<int ArrSize, class Tkey, class Tval, class Comp>
    inline void odd_even_transpose_sort_by_key(Tkey *keys, Tval *vals, Comp comp) {
        OddEvenTransposeSortByKeyT<0, ArrSize>::Sort(keys, vals, comp);
    }

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
