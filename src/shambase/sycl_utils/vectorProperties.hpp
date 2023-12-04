// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file vectorProperties.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */
 
#include "shambackends/typeAliasVec.hpp"
#include <limits>

namespace shambase {

    template<class T>
    struct VectorProperties {
        using component_type           = T;
        static constexpr u32 dimension = 1;

        static constexpr bool is_float_based = std::is_same<T, f16>::value ||
                                               std::is_same<T, f32>::value ||
                                               std::is_same<T, f64>::value;
        static constexpr bool is_uint_based =
            std::is_same<T, u8>::value || std::is_same<T, u16>::value ||
            std::is_same<T, u32>::value || std::is_same<T, u64>::value;
        static constexpr bool is_int_based =
            std::is_same<T, i8>::value || std::is_same<T, i16>::value ||
            std::is_same<T, i32>::value || std::is_same<T, i64>::value;

        static constexpr bool has_info = is_float_based || is_int_based || is_uint_based;

        static constexpr T get_min() { return std::numeric_limits<T>::min(); } //why this f***ing thing gives epsilon on float when you except -max aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaah !!!!
        static constexpr T get_max() { return std::numeric_limits<T>::max(); }
        static constexpr T get_inf() { return std::numeric_limits<T>::infinity(); }
        static constexpr T get_zero(){return 0;}
    };

    template<class T, u32 dim>
    struct VectorProperties<sycl::vec<T, dim>> {
        using component_type           = T;
        static constexpr u32 dimension = dim;

        static constexpr bool is_float_based = std::is_same<T, f16>::value ||
                                               std::is_same<T, f32>::value ||
                                               std::is_same<T, f64>::value;
        static constexpr bool is_uint_based =
            std::is_same<T, u8>::value || std::is_same<T, u16>::value ||
            std::is_same<T, u32>::value || std::is_same<T, u64>::value;
        static constexpr bool is_int_based =
            std::is_same<T, i8>::value || std::is_same<T, i16>::value ||
            std::is_same<T, i32>::value || std::is_same<T, i64>::value;
        static constexpr bool has_info = is_float_based || is_int_based || is_uint_based;

        static constexpr sycl::vec<T, dim> get_min() {
            constexpr T min = std::numeric_limits<T>::min();
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
            constexpr T max = std::numeric_limits<T>::max();
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
                    zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero};
            }
        }
    };

    template<class T>
    using VecComponent = typename VectorProperties<T>::component_type;

} // namespace shambase