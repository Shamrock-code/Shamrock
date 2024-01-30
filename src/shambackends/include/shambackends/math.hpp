// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file math.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/sycl_utils/sycl_utilities.hpp"
#include "shambase/sycl_utils/vec_equals.hpp"


namespace sham::syclbackport {



#ifndef SYCL2020_FEATURE_ISINF
#ifdef SYCL_COMP_ACPP
template<class T> 
bool fallback_is_inf(T value){

    __hipsycl_if_target_host(
        return std::isinf(value);
        )

    __hipsycl_if_target_hiplike(
        return isinf(value);
        ) 

    __hipsycl_if_target_spirv(
        static_assert(false, "this case is not implemented");
        )

}
#endif
#endif

}



namespace sham {

    template<class T>
    inline T min(T a, T b) {
        return shambase::sycl_utils::g_sycl_min(a, b);
    }

    template<class T>
    inline T max(T a, T b) {
        return shambase::sycl_utils::g_sycl_max(a, b);
    }

    template<class T>
    inline T abs(T a) {
        return shambase::sycl_utils::g_sycl_abs(a);
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
    inline bool equals(T a, T b){
        return shambase::vec_equals(a,b);
    }


    inline auto pack32(u32 a, u32 b) -> u64 {
        return (u64(a) << 32U) + b;
    };

    inline auto unpack32 (u64 v) -> sycl::vec<u32, 2> {
        return {u32(v >> 32U), u32(v)};
    };


    
    template<class T>
    inline bool has_nan(T v){
        auto tmp = ! sycl::isnan(v);
        return !(tmp);
    }

    template<class T>
    inline bool has_inf(T v){
        #ifdef SYCL2020_FEATURE_ISINF
            auto tmp = ! sycl::isinf(v);
            return !(tmp);
        #else
            auto tmp = ! syclbackport::fallback_is_inf(v);
            return !(tmp);
        #endif
    }

    template<class T>
    inline bool has_nan_or_inf(T v){
        #ifdef SYCL2020_FEATURE_ISINF
            auto tmp = ! (sycl::isnan(v) || sycl::isinf(v));
            return !(tmp);
        #else
            auto tmp = ! (sycl::isnan(v) || syclbackport::fallback_is_inf(v));
            return !(tmp);
        #endif
    }

    /**
     * @brief return true if vector has a nan
     * 
     * @tparam T 
     * @tparam n 
     * @param v 
     * @return true 
     * @return false 
     */
    template<class T, int n>
    inline bool has_nan(sycl::vec<T,n> v){
        bool has = false;
        #pragma unroll 
        for(i32 i = 0 ; i < n; i ++){
            has = has || (sycl::isnan(v[i]));
        }
        return has;
    }

    /**
     * @brief return true if vector has a inf
     * 
     * @tparam T 
     * @tparam n 
     * @param v 
     * @return true 
     * @return false 
     */
    template<class T, int n>
    inline bool has_inf(sycl::vec<T,n> v){
        bool has = false;
        #pragma unroll 
        for(i32 i = 0 ; i < n; i ++){
            #ifdef SYCL2020_FEATURE_ISINF
                has = has || (sycl::isinf(v[i]));
            #else
                has = has || (syclbackport::fallback_is_inf(v[i]));
            #endif
        }
        return has;
    }

    /**
     * @brief return true if vector has a nan or a inf
     * 
     * @tparam T 
     * @tparam n 
     * @param v 
     * @return true 
     * @return false 
     */
    template<class T, int n>
    inline bool has_nan_or_inf(sycl::vec<T,n> v){
        bool has = false;
        #pragma unroll 
        for(i32 i = 0 ; i < n; i ++){
            #ifdef SYCL2020_FEATURE_ISINF
                has = has || (sycl::isnan(v[i]) || sycl::isinf(v[i]));
            #else
                has = has || (sycl::isnan(v[i]) || syclbackport::fallback_is_inf(v[i]));
            #endif
        }
        return has;
    }

} // namespace sham