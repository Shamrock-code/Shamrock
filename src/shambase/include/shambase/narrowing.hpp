// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shambase/SourceLocation.hpp"
#include "shambase/exception.hpp"
#include "shambase/stacktrace.hpp"
#include "shambase/string.hpp"
#include "shambase/type_traits.hpp"
#include <limits>
#include <stdexcept>

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
            if constexpr (lim_T::is_signed) {
                if constexpr (lim_U::is_signed) {
                    // signed -> signed
                    return val >= lim_U::min() && val <= lim_U::max();
                } else {
                    // signed -> unsigned (cast to avoid compiler warning -Wsign-compare)
                    return val >= 0 && static_cast<std::make_unsigned_t<T>>(val) <= lim_U::max();
                }
            } else {
                if constexpr (lim_U::is_signed) {
                    // unsigned -> signed (again cast to avoid warning -Wsign-compare)
                    return val <= static_cast<std::make_unsigned_t<U>>(lim_U::max());
                } else {
                    // unsigned -> unsigned
                    return val <= lim_U::max();
                }
            }
        } else {
            static_assert(
                shambase::always_false_v<T>, "can_narrow is not implemented for this type");
        }
    }

    template<class U, class T>
    inline U narrow_or_throw(T val, SourceLocation &&callsite = SourceLocation{}) {

        __shamrock_log_callsite(callsite);

        if (can_narrow<U, T>(val)) {
            return static_cast<U>(val);
        } else {
            using lim_U = std::numeric_limits<U>;
            throw make_except_with_loc<std::runtime_error>(shambase::format(
                "value {} cannot be narrowed to type U (see signature) static_cast<U>({}) = {} "
                "(U::min() = {}, U::max() = {})",
                val,
                val,
                static_cast<U>(val),
                lim_U::min(),
                lim_U::max()));
        }
    }

    // When we will support it narrow_expected or narrow_or_expected would be nice

} // namespace shambase
