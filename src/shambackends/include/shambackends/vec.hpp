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
 * @file vec.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/numeric_limits.hpp"
#include "shambackends/typeAliasVec.hpp"

namespace sham {

    template<class T>
    struct VectorProperties {
        using component_type           = T;
        static constexpr u32 dimension = 1;

        static constexpr bool is_float_based = std::is_same<T, f16>::value
                                               || std::is_same<T, f32>::value
                                               || std::is_same<T, f64>::value;
        static constexpr bool is_uint_based
            = std::is_same<T, u8>::value || std::is_same<T, u16>::value
              || std::is_same<T, u32>::value || std::is_same<T, u64>::value;
        static constexpr bool is_int_based
            = std::is_same<T, i8>::value || std::is_same<T, i16>::value
              || std::is_same<T, i32>::value || std::is_same<T, i64>::value;

        static constexpr bool has_info = is_float_based || is_int_based || is_uint_based;

        static constexpr T get_min() {
            return shambase::get_min<T>();
        } // why this f***ing thing gives epsilon on float when you except -max
          // aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaah !!!!
        static constexpr T get_max() { return shambase::get_max<T>(); }
        static constexpr T get_inf() { return shambase::get_infty<T>(); }
        static constexpr T get_zero() { return 0; }
    };

    template<class T, u32 dim>
    struct VectorProperties<sycl::vec<T, dim>> {
        using component_type           = T;
        static constexpr u32 dimension = dim;

        static constexpr bool is_float_based = std::is_same<T, f16>::value
                                               || std::is_same<T, f32>::value
                                               || std::is_same<T, f64>::value;
        static constexpr bool is_uint_based
            = std::is_same<T, u8>::value || std::is_same<T, u16>::value
              || std::is_same<T, u32>::value || std::is_same<T, u64>::value;
        static constexpr bool is_int_based
            = std::is_same<T, i8>::value || std::is_same<T, i16>::value
              || std::is_same<T, i32>::value || std::is_same<T, i64>::value;
        static constexpr bool has_info = is_float_based || is_int_based || is_uint_based;

        static constexpr sycl::vec<T, dim> get_min() {
            constexpr T min = shambase::get_min<T>();
            if constexpr (dim == 2) {
                return {min, min};
            }
            if constexpr (dim == 3) {
                return {min, min, min};
            }
            if constexpr (dim == 4) {
                return {min, min, min, min};
            }
            if constexpr (dim == 8) {
                return {min, min, min, min, min, min, min, min};
            }
            if constexpr (dim == 16) {
                return {
                    min, min, min, min, min, min, min, min, min, min, min, min, min, min, min, min};
            }
        }
        static constexpr sycl::vec<T, dim> get_max() {
            constexpr T max = shambase::get_max<T>();
            if constexpr (dim == 2) {
                return {max, max};
            }
            if constexpr (dim == 3) {
                return {max, max, max};
            }
            if constexpr (dim == 4) {
                return {max, max, max, max};
            }
            if constexpr (dim == 8) {
                return {max, max, max, max, max, max, max, max};
            }
            if constexpr (dim == 16) {
                return {
                    max, max, max, max, max, max, max, max, max, max, max, max, max, max, max, max};
            }
        }
        static constexpr sycl::vec<T, dim> get_zero() {
            constexpr T zero = 0;
            if constexpr (dim == 2) {
                return {zero, zero};
            }
            if constexpr (dim == 3) {
                return {zero, zero, zero};
            }
            if constexpr (dim == 4) {
                return {zero, zero, zero, zero};
            }
            if constexpr (dim == 8) {
                return {zero, zero, zero, zero, zero, zero, zero, zero};
            }
            if constexpr (dim == 16) {
                return {
                    zero,
                    zero,
                    zero,
                    zero,
                    zero,
                    zero,
                    zero,
                    zero,
                    zero,
                    zero,
                    zero,
                    zero,
                    zero,
                    zero,
                    zero,
                    zero};
            }
        }
    };

    template<class T>
    using VecComponent = typename VectorProperties<T>::component_type;

} // namespace sham

namespace shambase {

    template<class T>
    using VectorProperties = sham::VectorProperties<T>;

    template<class T>
    using VecComponent = sham::VecComponent<T>;

} // namespace shambase
