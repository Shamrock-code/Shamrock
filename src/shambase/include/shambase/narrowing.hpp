// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shambase/type_traits.hpp"
#include <limits>

/**
 * @file narrowing.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Utilities for safe type narrowing conversions
 *
 */

namespace shambase {

    /**
     * @brief Check if an integer value can be safely narrowed to a target type
     *
     * This function checks whether an integer value of type T can be represented
     * in the target integer type U without overflow.
     * Works with both signed and unsigned integer types for now.
     *
     * @tparam U The target integer type to narrow to
     * @tparam T The source integer type of the value
     * @param val The value to check
     * @return true if the value can be safely narrowed to type U
     * @return false if narrowing would cause overflow
     *
     * @code{.cpp}
     * i32 value = 300;
     * bool can_fit = can_narrow<i8>(value);  // false, 300 > 127
     *
     * i32 small = 100;
     * bool ok = can_narrow<i16>(small);  // true, 100 fits in i16
     *
     * i32 negative = -10;
     * bool unsigned_ok = can_narrow<u32>(negative);  // false, negative value
     * @endcode
     */
    template<class U, class T>
    constexpr bool can_narrow(T val) {
        using lim_T = std::numeric_limits<T>;
        using lim_U = std::numeric_limits<U>;

        if constexpr (lim_T::is_integer && lim_U::is_integer) {
            // Check if signs differ and handle appropriately
            if constexpr (lim_T::is_signed && !lim_U::is_signed) {
                // Signed to unsigned: must be non-negative
                if (val < 0) {
                    return false;
                }
            } else if constexpr (!lim_T::is_signed && lim_U::is_signed) {
                // Unsigned to signed: must not exceed signed max
                if (val > static_cast<T>(lim_U::max())) {
                    return false;
                }
            }

            // Cast to U and back to T - if the value is unchanged, narrowing is safe
            return static_cast<T>(static_cast<U>(val)) == val;
        } else {
            static_assert(
                shambase::always_false_v<T>, "can_narrow is not implemented for this type");
        }
    }

} // namespace shambase
