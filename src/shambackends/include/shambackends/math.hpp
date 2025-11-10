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
 * @file math.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/integer.hpp"
#include "shambase/type_traits.hpp"
#include "shambackends/sycl.hpp"
#include "shambackends/vec.hpp"
#include <vector>

namespace sham::syclbackport {

#ifndef SYCL2020_FEATURE_ISINF
    #ifdef SYCL_COMP_ACPP
    template<class T>
    HIPSYCL_UNIVERSAL_TARGET bool fallback_is_inf(T value) {

        __hipsycl_if_target_host(return std::isinf(value);)

            __hipsycl_if_target_hiplike(return isinf(value);)

                __hipsycl_if_target_spirv(static_assert(false, "this case is not implemented");)
    }
    #endif
#endif

} // namespace sham::syclbackport

namespace sham {

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // product_accumulate
    ////////////////////////////////////////////////////////////////////////////////////////////////

    template<class T>
    inline constexpr T product_accumulate(T v) noexcept {
        return v;
    }

    template<class T, int n, std::enable_if_t<n == 2, int> = 0>
    inline constexpr T product_accumulate(sycl::vec<T, n> v) noexcept {
        return v.x() * v.y();
    }

    template<class T, int n, std::enable_if_t<n == 3, int> = 0>
    inline constexpr T product_accumulate(sycl::vec<T, n> v) noexcept {
        return v.x() * v.y() * v.z();
    }

    template<class T, int n, std::enable_if_t<n == 4, int> = 0>
    inline constexpr T product_accumulate(sycl::vec<T, n> v) noexcept {
        return v.x() * v.y() * v.z() * v.w();
    }

    template<class T, int n, std::enable_if_t<n == 8, int> = 0>
    inline constexpr T product_accumulate(sycl::vec<T, n> v) noexcept {
        return v.s0() * v.s1() * v.s2() * v.s3() * v.s4() * v.s5() * v.s6() * v.s7();
    }

    template<class T, int n, std::enable_if_t<n == 16, int> = 0>
    inline constexpr T product_accumulate(sycl::vec<T, n> v) noexcept {
        return v.s0() * v.s1() * v.s2() * v.s3() * v.s4() * v.s5() * v.s6() * v.s7() * v.s8()
               * v.s9() * v.sA() * v.sB() * v.sC() * v.sD() * v.sE() * v.sF();
    }

    template<class T>
    inline constexpr T sum_accumulate(T v) noexcept {
        return v;
    }

    template<class T, int n, std::enable_if_t<n == 2, int> = 0>
    inline constexpr T sum_accumulate(sycl::vec<T, n> v) noexcept {
        return v.x() + v.y();
    }

    template<class T, int n, std::enable_if_t<n == 3, int> = 0>
    inline constexpr T sum_accumulate(sycl::vec<T, n> v) noexcept {
        return v.x() + v.y() + v.z();
    }

    template<class T, int n, std::enable_if_t<n == 4, int> = 0>
    inline constexpr T sum_accumulate(sycl::vec<T, n> v) noexcept {
        return v.x() + v.y() + v.z() + v.w();
    }

    template<class T, int n, std::enable_if_t<n == 8, int> = 0>
    inline constexpr T sum_accumulate(sycl::vec<T, n> v) noexcept {
        return v.s0() + v.s1() + v.s2() + v.s3() + v.s4() + v.s5() + v.s6() + v.s7();
    }

    template<class T, int n, std::enable_if_t<n == 16, int> = 0>
    inline constexpr T sum_accumulate(sycl::vec<T, n> v) noexcept {
        return v.s0() + v.s1() + v.s2() + v.s3() + v.s4() + v.s5() + v.s6() + v.s7() + v.s8()
               + v.s9() + v.sA() + v.sB() + v.sC() + v.sD() + v.sE() + v.sF();
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // all_component_are_negative
    ////////////////////////////////////////////////////////////////////////////////////////////////

    template<class T, std::enable_if_t<std::is_signed<T>::value, int> = 0>
    inline constexpr bool all_component_are_negative(T a) {
        return a < 0;
    }

    template<class T, int n, std::enable_if_t<n == 2 && std::is_signed<T>::value, int> = 0>
    inline constexpr bool all_component_are_negative(sycl::vec<T, n> v) noexcept {
        return (v.x() < 0) && (v.y() < 0);
    }

    template<class T, int n, std::enable_if_t<n == 3 && std::is_signed<T>::value, int> = 0>
    inline constexpr bool all_component_are_negative(sycl::vec<T, n> v) noexcept {
        return (v.x() < 0) && (v.y() < 0) && (v.z() < 0);
    }

    template<class T, int n, std::enable_if_t<n == 4 && std::is_signed<T>::value, int> = 0>
    inline constexpr bool all_component_are_negative(sycl::vec<T, n> v) noexcept {
        return (v.x() < 0) && (v.y() < 0) && (v.z() < 0) && (v.w() < 0);
    }

    template<class T, int n, std::enable_if_t<n == 8 && std::is_signed<T>::value, int> = 0>
    inline constexpr bool all_component_are_negative(sycl::vec<T, n> v) noexcept {
        return (v.s0() < 0) && (v.s1() < 0) && (v.s2() < 0) && (v.s3() < 0) && (v.s4() < 0)
               && (v.s5() < 0) && (v.s6() < 0) && (v.s7() < 0);
    }

    template<class T, int n, std::enable_if_t<n == 16 && std::is_signed<T>::value, int> = 0>
    inline constexpr bool all_component_are_negative(sycl::vec<T, n> v) noexcept {
        return (v.s0() < 0) && (v.s1() < 0) && (v.s2() < 0) && (v.s3() < 0) && (v.s4() < 0)
               && (v.s5() < 0) && (v.s6() < 0) && (v.s7() < 0) && (v.s8() < 0) && (v.s9() < 0)
               && (v.sA() < 0) && (v.sB() < 0) && (v.sC() < 0) && (v.sD() < 0) && (v.sE() < 0)
               && (v.sF() < 0);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // vec_compare_geq
    ////////////////////////////////////////////////////////////////////////////////////////////////

    template<class T>
    inline constexpr bool vec_compare_geq(T a, T b) {
        return a >= b;
    }

    template<class T, int n, std::enable_if_t<n == 2, int> = 0>
    inline constexpr bool vec_compare_geq(sycl::vec<T, n> v, sycl::vec<T, n> w) noexcept {
        return (v.x() >= w.x()) && (v.y() >= w.y());
    }

    template<class T, int n, std::enable_if_t<n == 3, int> = 0>
    inline constexpr bool vec_compare_geq(sycl::vec<T, n> v, sycl::vec<T, n> w) noexcept {
        return (v.x() >= w.x()) && (v.y() >= w.y()) && (v.z() >= w.z());
    }

    template<class T, int n, std::enable_if_t<n == 4, int> = 0>
    inline constexpr bool vec_compare_geq(sycl::vec<T, n> v, sycl::vec<T, n> w) noexcept {
        return (v.x() >= w.x()) && (v.y() >= w.y()) && (v.z() >= w.z()) && (v.w() >= w.w());
    }

    template<class T, int n, std::enable_if_t<n == 8, int> = 0>
    inline constexpr bool vec_compare_geq(sycl::vec<T, n> v, sycl::vec<T, n> w) noexcept {
        return (v.s0() >= w.s0()) && (v.s1() >= w.s1()) && (v.s2() >= w.s2()) && (v.s3() >= w.s3())
               && (v.s4() >= w.s4()) && (v.s5() >= w.s5()) && (v.s6() >= w.s6())
               && (v.s7() >= w.s7());
    }

    template<class T, int n, std::enable_if_t<n == 16, int> = 0>
    inline constexpr bool vec_compare_geq(sycl::vec<T, n> v, sycl::vec<T, n> w) noexcept {
        return (v.s0() >= w.s0()) && (v.s1() >= w.s1()) && (v.s2() >= w.s2()) && (v.s3() >= w.s3())
               && (v.s4() >= w.s4()) && (v.s5() >= w.s5()) && (v.s6() >= w.s6())
               && (v.s7() >= w.s7()) && (v.s8() >= w.s8()) && (v.s9() >= w.s9())
               && (v.sA() >= w.sA()) && (v.sB() >= w.sB()) && (v.sC() >= w.sC())
               && (v.sD() >= w.sD()) && (v.sE() >= w.sE()) && (v.sF() >= w.sF());
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // vec_compare_leq
    ////////////////////////////////////////////////////////////////////////////////////////////////

    template<class T>
    inline constexpr bool vec_compare_leq(T a, T b) {
        return a <= b;
    }

    template<class T, int n, std::enable_if_t<n == 2, int> = 0>
    inline constexpr bool vec_compare_leq(sycl::vec<T, n> v, sycl::vec<T, n> w) noexcept {
        return (v.x() <= w.x()) && (v.y() <= w.y());
    }

    template<class T, int n, std::enable_if_t<n == 3, int> = 0>
    inline constexpr bool vec_compare_leq(sycl::vec<T, n> v, sycl::vec<T, n> w) noexcept {
        return (v.x() <= w.x()) && (v.y() <= w.y()) && (v.z() <= w.z());
    }

    template<class T, int n, std::enable_if_t<n == 4, int> = 0>
    inline constexpr bool vec_compare_leq(sycl::vec<T, n> v, sycl::vec<T, n> w) noexcept {
        return (v.x() <= w.x()) && (v.y() <= w.y()) && (v.z() <= w.z()) && (v.w() <= w.w());
    }

    template<class T, int n, std::enable_if_t<n == 8, int> = 0>
    inline constexpr bool vec_compare_leq(sycl::vec<T, n> v, sycl::vec<T, n> w) noexcept {
        return (v.s0() <= w.s0()) && (v.s1() <= w.s1()) && (v.s2() <= w.s2()) && (v.s3() <= w.s3())
               && (v.s4() <= w.s4()) && (v.s5() <= w.s5()) && (v.s6() <= w.s6())
               && (v.s7() <= w.s7());
    }

    template<class T, int n, std::enable_if_t<n == 16, int> = 0>
    inline constexpr bool vec_compare_leq(sycl::vec<T, n> v, sycl::vec<T, n> w) noexcept {
        return (v.s0() <= w.s0()) && (v.s1() <= w.s1()) && (v.s2() <= w.s2()) && (v.s3() <= w.s3())
               && (v.s4() <= w.s4()) && (v.s5() <= w.s5()) && (v.s6() <= w.s6())
               && (v.s7() <= w.s7()) && (v.s8() <= w.s8()) && (v.s9() <= w.s9())
               && (v.sA() <= w.sA()) && (v.sB() <= w.sB()) && (v.sC() <= w.sC())
               && (v.sD() <= w.sD()) && (v.sE() <= w.sE()) && (v.sF() <= w.sF());
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // vec_compare_g
    ////////////////////////////////////////////////////////////////////////////////////////////////

    template<class T>
    inline constexpr bool vec_compare_g(T a, T b) {
        return a > b;
    }

    template<class T, int n, std::enable_if_t<n == 2, int> = 0>
    inline constexpr bool vec_compare_g(sycl::vec<T, n> v, sycl::vec<T, n> w) noexcept {
        return (v.x() > w.x()) && (v.y() > w.y());
    }

    template<class T, int n, std::enable_if_t<n == 3, int> = 0>
    inline constexpr bool vec_compare_g(sycl::vec<T, n> v, sycl::vec<T, n> w) noexcept {
        return (v.x() > w.x()) && (v.y() > w.y()) && (v.z() > w.z());
    }

    template<class T, int n, std::enable_if_t<n == 4, int> = 0>
    inline constexpr bool vec_compare_g(sycl::vec<T, n> v, sycl::vec<T, n> w) noexcept {
        return (v.x() > w.x()) && (v.y() > w.y()) && (v.z() > w.z()) && (v.w() > w.w());
    }

    template<class T, int n, std::enable_if_t<n == 8, int> = 0>
    inline constexpr bool vec_compare_g(sycl::vec<T, n> v, sycl::vec<T, n> w) noexcept {
        return (v.s0() > w.s0()) && (v.s1() > w.s1()) && (v.s2() > w.s2()) && (v.s3() > w.s3())
               && (v.s4() > w.s4()) && (v.s5() > w.s5()) && (v.s6() > w.s6()) && (v.s7() > w.s7());
    }

    template<class T, int n, std::enable_if_t<n == 16, int> = 0>
    inline constexpr bool vec_compare_g(sycl::vec<T, n> v, sycl::vec<T, n> w) noexcept {
        return (v.s0() > w.s0()) && (v.s1() > w.s1()) && (v.s2() > w.s2()) && (v.s3() > w.s3())
               && (v.s4() > w.s4()) && (v.s5() > w.s5()) && (v.s6() > w.s6()) && (v.s7() > w.s7())
               && (v.s8() > w.s8()) && (v.s9() > w.s9()) && (v.sA() > w.sA()) && (v.sB() > w.sB())
               && (v.sC() > w.sC()) && (v.sD() > w.sD()) && (v.sE() > w.sE()) && (v.sF() > w.sF());
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // component_have_a_zero
    ////////////////////////////////////////////////////////////////////////////////////////////////

    template<class T>
    inline constexpr bool component_have_a_zero(T a) {
        return a == 0;
    }

    template<class T, int n, std::enable_if_t<n == 2, int> = 0>
    inline constexpr bool component_have_a_zero(sycl::vec<T, n> v) noexcept {
        return (v.x() == 0) || (v.y() == 0);
    }

    template<class T, int n, std::enable_if_t<n == 3, int> = 0>
    inline constexpr bool component_have_a_zero(sycl::vec<T, n> v) noexcept {
        return (v.x() == 0) || (v.y() == 0) || (v.z() == 0);
    }

    template<class T, int n, std::enable_if_t<n == 4, int> = 0>
    inline constexpr bool component_have_a_zero(sycl::vec<T, n> v) noexcept {
        return (v.x() == 0) || (v.y() == 0) || (v.z() == 0) || (v.w() == 0);
    }

    template<class T, int n, std::enable_if_t<n == 8, int> = 0>
    inline constexpr bool component_have_a_zero(sycl::vec<T, n> v) noexcept {
        return (v.s0() == 0) || (v.s1() == 0) || (v.s2() == 0) || (v.s3() == 0) || (v.s4() == 0)
               || (v.s5() == 0) || (v.s6() == 0) || (v.s7() == 0);
    }

    template<class T, int n, std::enable_if_t<n == 16, int> = 0>
    inline constexpr bool component_have_a_zero(sycl::vec<T, n> v) noexcept {
        return (v.s0() == 0) || (v.s1() == 0) || (v.s2() == 0) || (v.s3() == 0) || (v.s4() == 0)
               || (v.s5() == 0) || (v.s6() == 0) || (v.s7() == 0) || (v.s8() == 0) || (v.s9() == 0)
               || (v.sA() == 0) || (v.sB() == 0) || (v.sC() == 0) || (v.sD() == 0) || (v.sE() == 0)
               || (v.sF() == 0);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // component_have_only_one_zero
    ////////////////////////////////////////////////////////////////////////////////////////////////

    template<class T>
    inline constexpr bool component_have_only_one_zero(T a) {
        return a == 0;
    }

    template<class T, int n, std::enable_if_t<n == 2, int> = 0>
    inline constexpr bool component_have_only_one_zero(sycl::vec<T, n> v) noexcept {
        return (v.x() == 0) != (v.y() == 0);
    }

    template<class T, int n, std::enable_if_t<n == 3, int> = 0>
    inline constexpr bool component_have_only_one_zero(sycl::vec<T, n> v) noexcept {
        return 1 == int{(v.x() == 0)} + int{(v.y() == 0)} + int{(v.z() == 0)};
    }

    template<class T, int n, std::enable_if_t<n == 4, int> = 0>
    inline constexpr bool component_have_only_one_zero(sycl::vec<T, n> v) noexcept {
        return 1 == int{(v.x() == 0)} + int{(v.y() == 0)} + int{(v.z() == 0)} + int{(v.w() == 0)};
    }

    template<class T, int n, std::enable_if_t<n == 8, int> = 0>
    inline constexpr bool component_have_only_one_zero(sycl::vec<T, n> v) noexcept {
        return 1
               == int{(v.s0() == 0)} + int{(v.s1() == 0)} + int{(v.s2() == 0)} + int{(v.s3() == 0)}
                      + int{(v.s4() == 0)} + int{(v.s5() == 0)} + int{(v.s6() == 0)}
                      + int{(v.s7() == 0)};
    }

    template<class T, int n, std::enable_if_t<n == 16, int> = 0>
    inline constexpr bool component_have_only_one_zero(sycl::vec<T, n> v) noexcept {
        return 1
               == int{(v.s0() == 0)} + int{(v.s1() == 0)} + int{(v.s2() == 0)} + int{(v.s3() == 0)}
                      + int{(v.s4() == 0)} + int{(v.s5() == 0)} + int{(v.s6() == 0)}
                      + int{(v.s7() == 0)} + int{(v.s8() == 0)} + int{(v.s9() == 0)}
                      + int{(v.sA() == 0)} + int{(v.sB() == 0)} + int{(v.sC() == 0)}
                      + int{(v.sD() == 0)} + int{(v.sE() == 0)} + int{(v.sF() == 0)};
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // component_have_at_most_one_zero
    ////////////////////////////////////////////////////////////////////////////////////////////////

    template<class T>
    inline constexpr bool component_have_at_most_one_zero(T a) {
        return true;
    }

    template<class T, int n, std::enable_if_t<n == 2, int> = 0>
    inline constexpr bool component_have_at_most_one_zero(sycl::vec<T, n> v) noexcept {
        return 2 > int{(v.x() == 0)} + int{(v.y() == 0)};
    }

    template<class T, int n, std::enable_if_t<n == 3, int> = 0>
    inline constexpr bool component_have_at_most_one_zero(sycl::vec<T, n> v) noexcept {
        return 2 > int{(v.x() == 0)} + int{(v.y() == 0)} + int{(v.z() == 0)};
    }

    template<class T, int n, std::enable_if_t<n == 4, int> = 0>
    inline constexpr bool component_have_at_most_one_zero(sycl::vec<T, n> v) noexcept {
        return 2 > int{(v.x() == 0)} + int{(v.y() == 0)} + int{(v.z() == 0)} + int{(v.w() == 0)};
    }

    template<class T, int n, std::enable_if_t<n == 8, int> = 0>
    inline constexpr bool component_have_at_most_one_zero(sycl::vec<T, n> v) noexcept {
        return 2 > int{(v.s0() == 0)} + int{(v.s1() == 0)} + int{(v.s2() == 0)} + int{(v.s3() == 0)}
                       + int{(v.s4() == 0)} + int{(v.s5() == 0)} + int{(v.s6() == 0)}
                       + int{(v.s7() == 0)};
    }

    template<class T, int n, std::enable_if_t<n == 16, int> = 0>
    inline constexpr bool component_have_at_most_one_zero(sycl::vec<T, n> v) noexcept {
        return 2 > int{(v.s0() == 0)} + int{(v.s1() == 0)} + int{(v.s2() == 0)} + int{(v.s3() == 0)}
                       + int{(v.s4() == 0)} + int{(v.s5() == 0)} + int{(v.s6() == 0)}
                       + int{(v.s7() == 0)} + int{(v.s8() == 0)} + int{(v.s9() == 0)}
                       + int{(v.sA() == 0)} + int{(v.sB() == 0)} + int{(v.sC() == 0)}
                       + int{(v.sD() == 0)} + int{(v.sE() == 0)} + int{(v.sF() == 0)};
    }

} // namespace sham

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

        if constexpr (
            shambase::VectorProperties<T>::is_float_based
            && shambase::VectorProperties<T>::dimension <= 4) {

            return sycl::dot(a, b);

        } else {

            return sum_accumulate(a * b);
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

        return eqs0 && eqs1 && eqs2 && eqs3 && eqs4 && eqs5 && eqs6 && eqs7 && eqs8 && eqs9 && eqsA
               && eqsB && eqsC && eqsD && eqsE && eqsF;
    }

    template<class T>
    inline constexpr bool vec_equals(T a, T b) noexcept {
        bool eqx = a == b;
        return eqx;
    }
} // namespace sham::details

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
    inline shambase::VecComponent<T> max_component(T a) {

        using Tscal = shambase::VecComponent<T>;

        if constexpr (std::is_same_v<T, sycl::vec<Tscal, 2>>) {
            return sycl::max(a.x(), a.y());
        } else if constexpr (std::is_same_v<T, sycl::vec<Tscal, 3>>) {
            return sycl::max(a.x(), sycl::max(a.y(), a.z()));
        } else if constexpr (std::is_same_v<T, sycl::vec<Tscal, 4>>) {
            return sycl::max(sycl::max(a.x(), a.y()), sycl::max(a.z(), a.w()));
        } else if constexpr (std::is_same_v<T, sycl::vec<Tscal, 8>>) {
            return sycl::max(
                sycl::max(sycl::max(a.s0(), a.s1()), sycl::max(a.s2(), a.s3())),
                sycl::max(sycl::max(a.s4(), a.s5()), sycl::max(a.s6(), a.s7())));
        } else if constexpr (std::is_same_v<T, sycl::vec<Tscal, 16>>) {
            return sycl::max(
                sycl::max(
                    sycl::max(sycl::max(a.s0(), a.s1()), sycl::max(a.s2(), a.s3())),
                    sycl::max(sycl::max(a.s4(), a.s5()), sycl::max(a.s6(), a.s7()))),
                sycl::max(
                    sycl::max(sycl::max(a.s8(), a.s9()), sycl::max(a.sA(), a.sB())),
                    sycl::max(sycl::max(a.sC(), a.sD()), sycl::max(a.sE(), a.sF()))));
        } else {
            static_assert(
                shambase::always_false_v<T>, "max_component is not implemented for this type");
        }
    }

    template<class T>
    inline shambase::VecComponent<T> dot(T a, T b) {
        return sham::details::g_sycl_dot(a, b);
    }

    template<class T>
    inline shambase::VecComponent<T> length2(T a) {
        return dot(a, a);
    }

    template<class T>
    inline T max_8points(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7) {
        return max(max(max(v0, v1), max(v2, v3)), max(max(v4, v5), max(v6, v7)));
    }

    template<class T>
    inline T min_8points(T v0, T v1, T v2, T v3, T v4, T v5, T v6, T v7) {
        return min(min(min(v0, v1), min(v2, v3)), min(min(v4, v5), min(v6, v7)));
    }

    template<class T>
    inline T abs(T a) {
        return sham::details::g_sycl_abs(a);
    }

    template<class T>
    inline T positive_part(T a) {
        return (sham::abs(a) + a) / 2;
    }

    template<class T>
    inline T negative_part(T a) {
        return (sham::abs(a) - a) / 2;
    }

    template<class T>
    inline bool equals(T a, T b) {
        return details::vec_equals(a, b);
    }

    /// overload of equals for std::vector
    template<class T>
    inline bool equals(const std::vector<T> &a, const std::vector<T> &b) {
        if (a.size() != b.size()) {
            return false;
        }
        for (u32 i = 0; i < a.size(); i++) {
            if (!sham::equals(a[i], b[i])) {
                return false;
            }
        }
        return true;
    }

    inline auto pack32(u32 a, u32 b) -> u64 { return (u64(a) << 32U) + b; };

    inline auto unpack32(u64 v) -> sycl::vec<u32, 2> { return {u32(v >> 32U), u32(v)}; };

    template<class T>
    inline T m1pown(u32 n) {
        return (n % 2 == 0) ? T(1) : -T(1);
    }

    template<class T>
    inline bool has_nan(T v) {
        auto tmp = !sycl::isnan(v);
        return !(tmp);
    }

    template<class T>
    inline bool has_inf(T v) {
#ifdef SYCL2020_FEATURE_ISINF
        auto tmp = !sycl::isinf(v);
        return !(tmp);
#else
        auto tmp = !syclbackport::fallback_is_inf(v);
        return !(tmp);
#endif
    }

    template<class T>
    inline bool has_nan_or_inf(T v) {
#ifdef SYCL2020_FEATURE_ISINF
        auto tmp = !(sycl::isnan(v) || sycl::isinf(v));
        return !(tmp);
#else
        auto tmp = !(sycl::isnan(v) || syclbackport::fallback_is_inf(v));
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
    inline bool has_nan(sycl::vec<T, n> v) {
        bool has = false;
#pragma unroll
        for (i32 i = 0; i < n; i++) {
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
    inline bool has_inf(sycl::vec<T, n> v) {
        bool has = false;
#pragma unroll
        for (i32 i = 0; i < n; i++) {
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
    inline bool has_nan_or_inf(sycl::vec<T, n> v) {
        bool has = false;
#pragma unroll
        for (i32 i = 0; i < n; i++) {
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
    template<i32 power, class T>
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

    template<class T>
    inline constexpr T clz(T a) noexcept {
#ifdef SYCL2020_FEATURE_CLZ
        return sycl::clz(a);
#else
    #ifdef SYCL_COMP_ACPP

        if constexpr (std::is_same_v<T, u32>) {

            __hipsycl_if_target_host(return __builtin_clz(a);)

                __hipsycl_if_target_hiplike(return __clz(a);)

                    __hipsycl_if_target_spirv(return __spirv_ocl_clz(a);)

                        __hipsycl_if_target_sscp(return sycl::clz(a);)
        }

        if constexpr (std::is_same_v<T, u64>) {

            __hipsycl_if_target_host(return __builtin_clzll(a);)

                __hipsycl_if_target_hiplike(return __clzll(a);)

                    __hipsycl_if_target_spirv(return __spirv_ocl_clz(a);)

                        __hipsycl_if_target_sscp(return sycl::clz(a);)
        }

    #endif
#endif
    }

    /**
     * @brief give the length of the common prefix
     *
     * @tparam T the type
     * @param v
     * @return true
     * @return false
     */
    template<class T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
    inline constexpr T clz_xor(T a, T b) noexcept {
        return sham::clz(a ^ b);
    }

    /**
     * @brief compute the log2 of the number v being a power of 2
     */
    template<class T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
    inline constexpr T log2_pow2_num(T v) noexcept {
        return shambase::bitsizeof<T> - sham::clz(v) - 1;
    }

    /**
     * @brief round up to the next power of two
     *  0 is rounded up to 1 as it is not a pow of 2
     *  every input above the maximum power of 2 returns 0
     *
     * @tparam T
     * @param v
     * @return constexpr T
     */
    template<class T, std::enable_if_t<std::is_integral_v<T> || (!std::is_signed_v<T>), int> = 0>
    inline constexpr T roundup_pow2_clz(T v) noexcept {

        constexpr T max_signed_p1 = (shambase::get_max<T>() >> 1) + 1;

        bool is_pow2      = shambase::is_pow_of_two(v);
        bool is_above_max = v > max_signed_p1;

        return (is_above_max) ? 0 : ((is_pow2) ? v : 1U << (shambase::bitsizeof<T> - sham::clz(v)));
    };

    /**
     * @brief delta operator defined in Karras 2012
     *
     * @tparam Acc
     * @param x
     * @param y
     * @param morton_length
     * @param m
     * @return i32
     */
    template<class Acc>
    inline i32 karras_delta(i32 x, i32 y, u32 morton_length, Acc m) noexcept {
        return ((y > morton_length - 1 || y < 0) ? -1 : int(clz_xor(m[x], m[y])));
    }

    /**
     * @brief inverse saturated (positive numbers only)
     *
     * Computes the inverse of v if v < minsat return satval
     *
     * @param v
     * @param minvsat minimum value below which the inverse is not computed (default 1e-9)
     * @param satval saturation value (default 0)
     * @return T
     */
    template<class T>
    inline T inv_sat_positive(T v, T minvsat = T{1e-9}, T satval = T{0.}) noexcept {
        return (v >= minvsat) ? T{1.} / v : satval;
    }

    /**
     * @brief inverse saturated
     *
     * Computes the inverse of v if |v| < minsat return satval
     *
     * @param v
     * @param minvsat minimum value below which the inverse is not computed (default 1e-9)
     * @param satval saturation value (default 0)
     * @return T
     */
    template<class T>
    inline T inv_sat(T v, T minvsat = T{1e-9}, T satval = T{0.}) noexcept {
        return (std::abs(v) >= minvsat) ? T{1.} / v : satval;
    }

    /**
     * @brief inverse saturated (zero version)
     *
     * Computes the inverse of v if v==0 return satval
     *
     * @param v
     * @param satval saturation value (default 0)
     * @return T
     */
    template<class T>
    inline T inv_sat_zero(T v, T satval = T{0.}) noexcept {
        // return div only if v != 0 and is not NaN
        return (v != T{0} && v == v) ? T{1.} / v : satval;
    }

} // namespace sham
