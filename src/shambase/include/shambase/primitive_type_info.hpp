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
 * @file primitive_type_info.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/aliases_float.hpp"
#include "shambase/aliases_int.hpp"
#include <limits>

namespace shambase {

#if defined(DOXYGEN)
    /**
     * @brief Struct containing information about primitive types
     *
     * @tparam T The type of the primitive type
     */
    template<class _T>
    struct primitive_type_info {
        /// @brief The type of the primitive type
        using T = _T;

        /// @brief Whether the type is a specialized primitive type (i.e. not a float)
        static constexpr bool is_specialized = false;

        /// @brief Whether the type is a float type
        static constexpr bool is_float = false;

        /// @brief Whether the type is an integer type
        static constexpr bool is_int = false;

        /// @brief Whether the type is an unsigned integer type
        static constexpr bool is_unsigned = false;

        /// @brief The maximum value of the type
        static constexpr T max = std::numeric_limits<T>::max();

        /// @brief The minimum value of the type (negative if float)
        static constexpr T min = std::numeric_limits<T>::lowest();

        /// @brief The smallest value of the type that can be represented exactly
        static constexpr T epsilon = std::numeric_limits<T>::epsilon();

        /// @brief Infinity if the type can represent it
        static constexpr T infty = std::numeric_limits<T>::infinity();
    };
#else

    template<class T>
    struct primitive_type_info {
        static constexpr bool is_specialized = false;
        static constexpr bool is_float       = false;
        static constexpr bool is_int         = false;
        static constexpr bool is_unsigned    = false;
    };

    template<typename _T>
    struct primitive_type_info<const _T> : public primitive_type_info<_T> {};

    template<>
    struct primitive_type_info<f64> {
        using T = f64;

        static constexpr bool is_specialized = true;
        static constexpr bool is_float       = true;
        static constexpr bool is_int         = false;
        static constexpr bool is_unsigned    = false;

        static constexpr T max     = std::numeric_limits<T>::max();
        static constexpr T min     = std::numeric_limits<T>::lowest();
        static constexpr T epsilon = std::numeric_limits<T>::epsilon();
        static constexpr T infty   = std::numeric_limits<T>::infinity();
    };

    template<>
    struct primitive_type_info<f32> {
        using T = f32;

        static constexpr bool is_specialized = true;
        static constexpr bool is_float       = true;
        static constexpr bool is_int         = false;
        static constexpr bool is_unsigned    = false;

        static constexpr T max     = std::numeric_limits<T>::max();
        static constexpr T min     = std::numeric_limits<T>::lowest();
        static constexpr T epsilon = std::numeric_limits<T>::epsilon();
        static constexpr T infty   = std::numeric_limits<T>::infinity();
    };

    template<>
    struct primitive_type_info<i8> {
        using T = i8;

        static constexpr bool is_specialized = true;
        static constexpr bool is_float       = false;
        static constexpr bool is_int         = true;
        static constexpr bool is_unsigned    = false;

        static constexpr T max = std::numeric_limits<T>::max();
        static constexpr T min = std::numeric_limits<T>::lowest();
    };

    template<>
    struct primitive_type_info<i16> {
        using T = i16;

        static constexpr bool is_specialized = true;
        static constexpr bool is_float       = false;
        static constexpr bool is_int         = true;
        static constexpr bool is_unsigned    = false;

        static constexpr T max = std::numeric_limits<T>::max();
        static constexpr T min = std::numeric_limits<T>::lowest();
    };

    template<>
    struct primitive_type_info<i32> {
        using T = i32;

        static constexpr bool is_specialized = true;
        static constexpr bool is_float       = false;
        static constexpr bool is_int         = true;
        static constexpr bool is_unsigned    = false;

        static constexpr T max = std::numeric_limits<T>::max();
        static constexpr T min = std::numeric_limits<T>::lowest();
    };

    template<>
    struct primitive_type_info<i64> {
        using T = i64;

        static constexpr bool is_specialized = true;
        static constexpr bool is_float       = false;
        static constexpr bool is_int         = true;
        static constexpr bool is_unsigned    = false;

        static constexpr T max = std::numeric_limits<T>::max();
        static constexpr T min = std::numeric_limits<T>::lowest();
    };
    template<>
    struct primitive_type_info<u8> {
        using T = u8;

        static constexpr bool is_specialized = true;
        static constexpr bool is_float       = false;
        static constexpr bool is_int         = true;
        static constexpr bool is_unsigned    = true;

        static constexpr T max = std::numeric_limits<T>::max();
        static constexpr T min = std::numeric_limits<T>::lowest();
    };

    template<>
    struct primitive_type_info<u16> {
        using T = u16;

        static constexpr bool is_specialized = true;
        static constexpr bool is_float       = false;
        static constexpr bool is_int         = true;
        static constexpr bool is_unsigned    = true;

        static constexpr T max = std::numeric_limits<T>::max();
        static constexpr T min = std::numeric_limits<T>::lowest();
    };

    template<>
    struct primitive_type_info<u32> {
        using T = u32;

        static constexpr bool is_specialized = true;
        static constexpr bool is_float       = false;
        static constexpr bool is_int         = true;
        static constexpr bool is_unsigned    = true;

        static constexpr T max = std::numeric_limits<T>::max();
        static constexpr T min = std::numeric_limits<T>::lowest();
    };

    template<>
    struct primitive_type_info<u64> {
        using T = u64;

        static constexpr bool is_specialized = true;
        static constexpr bool is_float       = false;
        static constexpr bool is_int         = true;
        static constexpr bool is_unsigned    = true;

        static constexpr T max = std::numeric_limits<T>::max();
        static constexpr T min = std::numeric_limits<T>::lowest();
    };
#endif

} // namespace shambase
