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
 * @file numeric_limits.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/aliases_float.hpp"
#include "shambase/primitive_type_info.hpp"
#include <limits>

namespace shambase {

    template<class T>
    constexpr T get_max() {
        static_assert(
            primitive_type_info<T>::is_specialized,
            "this type was not implemented in shambase::details::numeric_limits");
        return primitive_type_info<T>::max;
    }

    template<class T>
    constexpr T get_min() {
        static_assert(
            primitive_type_info<T>::is_specialized,
            "this type was not implemented in shambase::details::numeric_limits");
        return primitive_type_info<T>::min;
    }

    template<class T>
    constexpr T get_epsilon() {
        static_assert(
            primitive_type_info<T>::is_specialized,
            "this type was not implemented in shambase::details::numeric_limits");
        static_assert(
            primitive_type_info<T>::is_float, "this function can only be called on floats");
        return primitive_type_info<T>::epsilon;
    }

    template<class T>
    constexpr T get_infty() {
        static_assert(
            primitive_type_info<T>::is_specialized,
            "this type was not implemented in shambase::details::numeric_limits");
        static_assert(
            primitive_type_info<T>::is_float, "this function can only be called on floats");
        return primitive_type_info<T>::infty;
    }

} // namespace shambase

#ifndef INT_ALIAS_LIM_DEFINED

constexpr i64 i64_max = shambase::get_max<i64>(); ///< i64 max value
constexpr i32 i32_max = shambase::get_max<i32>(); ///< i32 max value
constexpr i16 i16_max = shambase::get_max<i16>(); ///< i16 max value
constexpr i8 i8_max   = shambase::get_max<i8>();  ///< i8 max value

constexpr i64 i64_min = shambase::get_min<i64>(); ///< i64 min value
constexpr i32 i32_min = shambase::get_min<i32>(); ///< i32 min value
constexpr i16 i16_min = shambase::get_min<i16>(); ///< i16 min value
constexpr i8 i8_min   = shambase::get_min<i8>();  ///< i8 min value

constexpr u64 u64_max = shambase::get_max<u64>(); ///< u64 max value
constexpr u32 u32_max = shambase::get_max<u32>(); ///< u32 max value
constexpr u16 u16_max = shambase::get_max<u16>(); ///< u16 max value
constexpr u8 u8_max   = shambase::get_max<u8>();  ///< u8 max value

constexpr u64 u64_min = shambase::get_min<u64>(); ///< u64 min value
constexpr u32 u32_min = shambase::get_min<u32>(); ///< u32 min value
constexpr u16 u16_min = shambase::get_min<u16>(); ///< u16 min value
constexpr u8 u8_min   = shambase::get_min<u8>();  ///< u8 min value

#endif
