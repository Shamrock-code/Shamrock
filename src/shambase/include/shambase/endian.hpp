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
 * @file endian.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/aliases_int.hpp"
#include "shambase/integer.hpp"
#include <bit>

namespace shambase {

    /**
     * @brief Check if the CPU is in little endian
     *
     * @return true if the CPU is in little endian, false otherwise
     */
    inline bool is_little_endian() { return std::endian::native == std::endian::little; }

    /**
     * @brief Swap the endianness of the input value
     *
     * This function will swap the endianness of the input value `a`.
     *
     * @tparam T type of the input value
     * @param a input value
     */
    template<class T>
    inline void endian_swap(T &a) {

        constexpr i32 sz = sizeof(a);

        // Compute the number of byte swaps to perform, which is half of the size
        // of the type if it is even, or (size - 1) / 2 if it is odd
        auto constexpr lambd = []() {
            if constexpr (sz % 2 == 0) {
                return sz / 2;
            } else {
                return (sz - 1) / 2;
            }
        };

        constexpr i32 steps = lambd();

        u8 *bytes = (u8 *) &a;

        // Perform byte swaps
        for (i32 i = 0; i < steps; i++) {
            xor_swap(bytes[i], bytes[sz - 1 - i]);
        }
    }

    /**
     * @brief Return a copy of the input value with the endianness swapped
     *
     * This function returns a copy of its input value, with the endianness of the value
     * swapped. The input value is not modified.
     *
     * @tparam T The type of the input value.
     * @param a The input value whose endianness is to be swapped.
     * @return A copy of `a` with its endianness swapped.
     */
    template<class T>
    inline T get_endian_swap(T a) {
        T ret = a;
        endian_swap(ret);
        return ret;
    }

} // namespace shambase
