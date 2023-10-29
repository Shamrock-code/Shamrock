// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file sycl_utilities.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */
 
#include "shambackends/sycl.hpp"
#include "shambase/type_traits.hpp"
#include "shambase/vectors.hpp"
#include "vectorProperties.hpp"

namespace shambase::sycl_utils {

    template<class T>
    inline T g_sycl_min(T a, T b) {

        static_assert(VectorProperties<T>::has_info, "no info about this type");

        if constexpr (VectorProperties<T>::is_float_based) {
            return sycl::fmin(a, b);
        } else if constexpr (VectorProperties<T>::is_int_based) {
            return sycl::min(a, b);
        } else if constexpr (VectorProperties<T>::is_uint_based) {
            return sycl::min(a, b);
        }
    }

    template<class T>
    inline T g_sycl_max(T a, T b) {

        static_assert(VectorProperties<T>::has_info, "no info about this type");

        if constexpr (VectorProperties<T>::is_float_based) {
            return sycl::fmax(a, b);
        } else if constexpr (VectorProperties<T>::is_int_based) {
            return sycl::max(a, b);
        } else if constexpr (VectorProperties<T>::is_uint_based) {
            return sycl::max(a, b);
        }
    }

    template<class T>
    inline T g_sycl_abs(T a) {

        static_assert(VectorProperties<T>::has_info, "no info about this type");

        if constexpr (VectorProperties<T>::is_float_based) {
            return sycl::fabs(a);
        } else if constexpr (VectorProperties<T>::is_int_based) {
            return sycl::abs(a);
        } else if constexpr (VectorProperties<T>::is_uint_based) {
            return sycl::abs(a);
        }

    }

    template<class T>
    inline T positive_part(T a){
        return (g_sycl_abs(a) + a)/2;
    }

    template<class T>
    inline T negative_part(T a){
        return (g_sycl_abs(a) - a)/2;
    }

    template<class T>
    inline VecComponent<T> g_sycl_dot(T a, T b) {

        static_assert(VectorProperties<T>::has_info, "no info about this type");

        if constexpr (VectorProperties<T>::is_float_based && VectorProperties<T>::dimension <=4) {

            return sycl::dot(a, b);

        } else {

            return sum_accumulate(a * b);
        }
    }

    template<class T>
    inline T max_8points(T v0,T v1,T v2,T v3,T v4,T v5,T v6,T v7){
        return g_sycl_max(
                g_sycl_max( g_sycl_max(v0, v1), g_sycl_max(v2, v3))
            , 
                g_sycl_max( g_sycl_max(v4, v5), g_sycl_max(v6, v7))
            );
    }


    template<class T>
    inline T min_8points(T v0,T v1,T v2,T v3,T v4,T v5,T v6,T v7){
        return g_sycl_min(
                g_sycl_min( g_sycl_min(v0, v1), g_sycl_min(v2, v3))
            , 
                g_sycl_min( g_sycl_min(v4, v5), g_sycl_min(v6, v7))
            );
    }

} // namespace shambase::sycl_utils