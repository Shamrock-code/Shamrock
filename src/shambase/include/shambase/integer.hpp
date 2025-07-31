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
 * @file integer.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/aliases_int.hpp"
#include "shambase/type_traits.hpp"
#include <climits>

namespace shambase {

    /**
     * @brief determine if v is a power of two
     * Warning : this function return true if v == 0
     * Source : https://graphics.stanford.edu/~seander/bithacks.html
     *
     * @tparam T the type
     * @param v
     * @return true
     * @return false
     */
    template<class T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
    inline constexpr bool is_pow_of_two_fast(T v) noexcept {
        return (v & (v - 1)) == 0;
    }

    /**
     * @brief determine if v is a power of two and check if v==0
     * Source : https://graphics.stanford.edu/~seander/bithacks.html
     * @tparam T the type
     * @param v
     * @return true
     * @return false
     */
    template<class T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
    inline constexpr bool is_pow_of_two(T v) noexcept {
        return v && !(v & (v - 1));
    }

    /**
     * @brief check if the sign of the two integers differs
     * Source : https://graphics.stanford.edu/~seander/bithacks.html
     *
     * @tparam T
     * @param a
     * @param b
     * @return true
     * @return false
     */
    template<class T, std::enable_if_t<std::is_integral_v<T> || std::is_signed_v<T>, int> = 0>
    inline constexpr bool sign_differ(T a, T b) noexcept {
        return ((a ^ b) < 0);
    }

    /**
     * @brief swap two values using xor
     * Source : https://graphics.stanford.edu/~seander/bithacks.html
     *
     * @tparam T
     * @param a
     * @param b
     */
    template<class T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
    inline void xor_swap(T &a, T &b) {
        a ^= b;
        b ^= a;
        a ^= b;
    };

    /**
     * @brief round up to the next power of two
     * Source : https://graphics.stanford.edu/~seander/bithacks.html
     *
     * @tparam T
     * @param v
     * @return constexpr T
     */
    template<class T, std::enable_if_t<std::is_integral_v<T> || (!std::is_signed_v<T>), int> = 0>
    inline constexpr T roundup_pow2(T v) noexcept {

        if constexpr (has_bitlen_v<T, 32>) {
            v--;
            v |= v >> 1;
            v |= v >> 2;
            v |= v >> 4;
            v |= v >> 8;
            v |= v >> 16;
            v++;
            return v;
        }

        if constexpr (has_bitlen_v<T, 64>) {
            v--;
            v |= v >> 1;
            v |= v >> 2;
            v |= v >> 4;
            v |= v >> 8;
            v |= v >> 16;
            v |= v >> 32;
            v++;
            return v;
        }
    };

    /**
     * @brief Calculates the number of groups based on the length and group size
     *
     * @param len The total length
     * @param group_size The size of each group
     * @return constexpr u32 The number of groups
     */
    inline constexpr u32 group_count(u32 len, u32 group_size) {
        return (len + group_size - 1) / group_size;
    }

    /**
     * @brief Selects and returns the bit at a specific position in the given value
     *
     * @tparam T The type of the value
     * @param value The value from which to select the bit
     * @param bitnum The position of the bit to select
     * @return T The selected bit
     */
    template<class T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
    inline T select_bit(T value, T bitnum) {
        return (value >> bitnum) & 1;
    }

    /**
     * @brief Generates a mask with only the most significant bit set
     *
     * @tparam T The type of the value
     * @return constexpr T The mask with only the most significant bit set
     */
    template<class T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
    inline constexpr T most_sig_bit_mask() noexcept {
        return T(std::make_unsigned_t<T>(1) << (sizeof(T) * 8 - 1));
    }

    /**
     * @brief Checks if the most significant bit is set in the given value
     *
     * @tparam T The type of the value
     * @param x The value to check
     * @return constexpr bool True if the most significant bit is set, false otherwise
     */
    template<class T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
    inline constexpr bool is_most_sig_bit_set(const T x) noexcept {
        return (x & most_sig_bit_mask<T>());
    }

    /**
     * @brief Calculates the power of a number at compile time
     *
     * @tparam power The power to raise the number to
     * @tparam T The type of the number
     * @param a The number to raise to the power
     * @return constexpr T The result of the power calculation
     */
    template<i32 power, class T>
    inline constexpr T pow_constexpr(T a) noexcept {

        if constexpr (power == 0) {
            return T{1};
        } else if constexpr (power % 2 == 0) {
            T tmp = pow_constexpr<power / 2>(a);
            return tmp * tmp;
        } else if constexpr (power % 2 == 1) {
            T tmp = pow_constexpr<(power - 1) / 2>(a);
            return tmp * tmp * a;
        }
    }

    template<u32 flag>
    inline constexpr bool is_flag_on(u32 val) {
        return (val & (u32) flag) == (u32) flag;
    }

} // namespace shambase
