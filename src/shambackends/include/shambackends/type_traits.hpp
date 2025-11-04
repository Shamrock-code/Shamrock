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
 * @file type_traits.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Contains traits and utilities for backend related types
 *
 * This file contains traits and utilities for SYCL types.
 * It provides a way to check if a type is a SYCL vector and if it is a valid
 * SYCL vector size.
 */

#include "shambase/aliases_float.hpp"
#include "shambase/type_name_info.hpp"
#include "shambase/type_traits.hpp"
#include "shambackends/typeAliasFp16.hpp"

namespace sham {

    /**
     * @brief Check if the given integer is a valid size for a SYCL vector
     *
     * A valid size for a SYCL vector is 2, 3, 4, 8 or 16.
     *
     * @param N The integer to check
     * @return true If N is a valid SYCL vector size
     * @return false If N is not a valid SYCL vector size
     */
    inline constexpr bool is_valid_sycl_vec_size(int N) {
        return N == 2 || N == 3 || N == 4 || N == 8 || N == 16;
    }

    /**
     * @brief Check if a type is a valid SYCL base type in Shamrock
     *
     * A valid SYCL base type in shamrock is one of: `int64_t`, `int32_t`, `int16_t`, `int8_t`,
     * `uint64_t`, `uint32_t`, `uint16_t`, `uint8_t`, `half`, `float`, `double`.
     *
     * @tparam T Type to check
     * @return true If T is a valid SYCL base type
     * @return false If T is not a valid SYCL base type
     */
    template<class T>
    inline constexpr bool is_valid_sycl_base_type
        = std::is_same_v<T, i64> || std::is_same_v<T, i32> || std::is_same_v<T, i16>
          || std::is_same_v<T, i8> || std::is_same_v<T, u64> || std::is_same_v<T, u32>
          || std::is_same_v<T, u16> || std::is_same_v<T, u8> || std::is_same_v<T, f16>
          || std::is_same_v<T, f32> || std::is_same_v<T, f64>;

} // namespace sham

namespace shambase {

    template<class T, int n>
    struct TypeNameInfo<sycl::vec<T, n>> {
        inline static const std::string name
            = "sycl::vec<" + get_type_name<T>() + "," + std::to_string(n) + ">";
    };

} // namespace shambase
