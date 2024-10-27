// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
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
