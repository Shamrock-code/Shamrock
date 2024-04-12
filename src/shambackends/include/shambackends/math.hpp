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


#include "shambackends/sycl.hpp"
#include "shambackends/vec.hpp"
#include "shambase/vectors.hpp"


namespace sham::syclbackport {



#ifndef SYCL2020_FEATURE_ISINF
#ifdef SYCL_COMP_ACPP
template<class T> 
HIPSYCL_UNIVERSAL_TARGET bool fallback_is_inf(T value){

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

namespace sham::details {
    template<class T>
    inline T g_sycl_min(T a, T b) {

        static_assert(shambase::VectorProperties<T>::has_info, "no info about this type");

        if constexpr (shambase::VectorProperties<T>::is_float_based) {
            return sycl::fmin(a, b);
        } else if constexpr (shambase::VectorProperties<T>::is_int_based) {
            return sycl::min(a, b);
        } else if constexpr (shambase::VectorProperties<T>::is_uint_based) {
            return sycl::min(a, b);
        }
    }

    template<class T>
    inline T g_sycl_max(T a, T b) {

        static_assert(shambase::VectorProperties<T>::has_info, "no info about this type");

        if constexpr (shambase::VectorProperties<T>::is_float_based) {
            return sycl::fmax(a, b);
        } else if constexpr (shambase::VectorProperties<T>::is_int_based) {
            return sycl::max(a, b);
        } else if constexpr (shambase::VectorProperties<T>::is_uint_based) {
            return sycl::max(a, b);
        }
    }

    template<class T>
    inline T g_sycl_abs(T a) {

        static_assert(shambase::VectorProperties<T>::has_info, "no info about this type");

        if constexpr (shambase::VectorProperties<T>::is_float_based) {
            return sycl::fabs(a);
        } else if constexpr (shambase::VectorProperties<T>::is_int_based) {
            return sycl::abs(a);
        } else if constexpr (shambase::VectorProperties<T>::is_uint_based) {
            return sycl::abs(a);
        }

    }

    template<class T>
    inline shambase::VecComponent<T> g_sycl_dot(T a, T b) {

        static_assert(shambase::VectorProperties<T>::has_info, "no info about this type");

        if constexpr (shambase::VectorProperties<T>::is_float_based && shambase::VectorProperties<T>::dimension <=4) {

            return sycl::dot(a, b);

        } else {

            return shambase::sum_accumulate(a * b);
        }
    }


    template<class T>
    inline constexpr bool vec_equals(sycl::vec<T, 2> a, sycl::vec<T, 2> b) noexcept {
        bool eqx = a.x() == b.x();
        bool eqy = a.y() == b.y();
        return eqx && eqy;
    }

    template<class T>
    inline constexpr bool vec_equals(sycl::vec<T, 3> a, sycl::vec<T, 3> b) noexcept {
        bool eqx = a.x() == b.x();
        bool eqy = a.y() == b.y();
        bool eqz = a.z() == b.z();
        return eqx && eqy && eqz;
    }

    template<class T>
    inline constexpr bool vec_equals(sycl::vec<T, 4> a, sycl::vec<T, 4> b) noexcept {
        bool eqx = a.x() == b.x();
        bool eqy = a.y() == b.y();
        bool eqz = a.z() == b.z();
        bool eqw = a.w() == b.w();
        return eqx && eqy && eqz && eqw;
    }

    template<class T>
    inline constexpr bool vec_equals(sycl::vec<T, 8> a, sycl::vec<T, 8> b) noexcept {
        bool eqs0 = a.s0() == b.s0();
        bool eqs1 = a.s1() == b.s1();
        bool eqs2 = a.s2() == b.s2();
        bool eqs3 = a.s3() == b.s3();
        bool eqs4 = a.s4() == b.s4();
        bool eqs5 = a.s5() == b.s5();
        bool eqs6 = a.s6() == b.s6();
        bool eqs7 = a.s7() == b.s7();
        return eqs0 && eqs1 && eqs2 && eqs3 && eqs4 && eqs5 && eqs6 && eqs7;
    }

    template<class T>
    inline constexpr bool vec_equals(sycl::vec<T, 16> a, sycl::vec<T, 16> b) noexcept {
        bool eqs0 = a.s0() == b.s0();
        bool eqs1 = a.s1() == b.s1();
        bool eqs2 = a.s2() == b.s2();
        bool eqs3 = a.s3() == b.s3();
        bool eqs4 = a.s4() == b.s4();
        bool eqs5 = a.s5() == b.s5();
        bool eqs6 = a.s6() == b.s6();
        bool eqs7 = a.s7() == b.s7();

        bool eqs8 = a.s8() == b.s8();
        bool eqs9 = a.s9() == b.s9();
        bool eqsA = a.sA() == b.sA();
        bool eqsB = a.sB() == b.sB();
        bool eqsC = a.sC() == b.sC();
        bool eqsD = a.sD() == b.sD();
        bool eqsE = a.sE() == b.sE();
        bool eqsF = a.sF() == b.sF();

        return eqs0 && eqs1 && eqs2 && eqs3 && eqs4 && eqs5 && eqs6 && eqs7 && eqs8 && eqs9 &&
               eqsA && eqsB && eqsC && eqsD && eqsE && eqsF;
    }

    template<class T>
    inline constexpr bool vec_equals(T a, T b) noexcept {
        bool eqx = a == b;
        return eqx;
    }
}



namespace sham {

    template<class T>
    inline T min(T a, T b) {
        return sham::details::g_sycl_min(a, b);
    }

    template<class T>
    inline T max(T a, T b) {
        return sham::details::g_sycl_max(a, b);
    }

    template<class T>
    inline shambase::VecComponent<T> dot(T a, T b){
        return sham::details::g_sycl_dot(a, b);
    }

    template<class T>
    inline T max_8points(T v0,T v1,T v2,T v3,T v4,T v5,T v6,T v7){
        return max(
                max( max(v0, v1), max(v2, v3))
            , 
                max( max(v4, v5), max(v6, v7))
            );
    }


    template<class T>
    inline T min_8points(T v0,T v1,T v2,T v3,T v4,T v5,T v6,T v7){
        return min(
                min( min(v0, v1), min(v2, v3))
            , 
                min( min(v4, v5), min(v6, v7))
            );
    }

    template<class T>
    inline T abs(T a) {
        return sham::details::g_sycl_abs(a);
    }

    template<class T>
    inline T positive_part(T a){
        return (sham::abs(a) + a)/2;
    }

    template<class T>
    inline T negative_part(T a){
        return (sham::abs(a) - a)/2;
    }

    template<class T>
    inline bool equals(T a, T b){
        return details::vec_equals(a,b);
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

    /**
     * @brief generalized pow constexpr
     * 
     * @tparam power 
     * @tparam T 
     * @param a 
     * @return constexpr T 
     */
    template<i32 power,class T>
    inline constexpr T pow_constexpr(T a) noexcept {

        if constexpr (power < 0) {
            return pow_constexpr<-power>(T{1} / a);
        } else if constexpr (power == 0) {
            return T{1};
        } else if constexpr (power == 1) {
            return a;
        } else if constexpr (power % 2 == 0) {
            T tmp = pow_constexpr<power / 2>(a);
            return tmp * tmp;
        } else if constexpr (power % 2 == 1) {
            T tmp = pow_constexpr<(power - 1) / 2>(a);
            return tmp * tmp * a;
        }

    }

} // namespace sham