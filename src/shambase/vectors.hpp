// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file vectors.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */
 
#include "shambase/type_aliases.hpp"

namespace shambase {

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
        return v.s0() * v.s1() * v.s2() * v.s3() * v.s4() * v.s5() * v.s6() * v.s7() * v.s8() *
               v.s9() * v.sA() * v.sB() * v.sC() * v.sD() * v.sE() * v.sF();
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
        return v.s0() + v.s1() + v.s2() + v.s3() + v.s4() + v.s5() + v.s6() + v.s7() + v.s8() +
               v.s9() + v.sA() + v.sB() + v.sC() + v.sD() + v.sE() + v.sF();
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // all_component_are_negative
    ////////////////////////////////////////////////////////////////////////////////////////////////

    template<class T, std::enable_if_t<std::is_signed<T>::value, int> = 0>
    inline constexpr bool all_component_are_negative(T a) {
        return a < 0;
    }

    template<class T, int n, std::enable_if_t<n == 2 && std::is_signed<T>::value, int> = 0>
    inline constexpr T all_component_are_negative(sycl::vec<T, n> v) noexcept {
        return (v.x() < 0) && (v.y() < 0);
    }

    template<class T, int n, std::enable_if_t<n == 3 && std::is_signed<T>::value, int> = 0>
    inline constexpr T all_component_are_negative(sycl::vec<T, n> v) noexcept {
        return (v.x() < 0) && (v.y() < 0) && (v.z() < 0);
    }

    template<class T, int n, std::enable_if_t<n == 4 && std::is_signed<T>::value, int> = 0>
    inline constexpr T all_component_are_negative(sycl::vec<T, n> v) noexcept {
        return (v.x() < 0) && (v.y() < 0) && (v.z() < 0) && (v.w() < 0);
    }

    template<class T, int n, std::enable_if_t<n == 8 && std::is_signed<T>::value, int> = 0>
    inline constexpr T all_component_are_negative(sycl::vec<T, n> v) noexcept {
        return (v.s0() < 0) && (v.s1() < 0) && (v.s2() < 0) && (v.s3() < 0) && (v.s4() < 0) &&
               (v.s5() < 0) && (v.s6() < 0) && (v.s7() < 0);
    }

    template<class T, int n, std::enable_if_t<n == 16 && std::is_signed<T>::value, int> = 0>
    inline constexpr T all_component_are_negative(sycl::vec<T, n> v) noexcept {
        return (v.s0() < 0) && (v.s1() < 0) && (v.s2() < 0) && (v.s3() < 0) && (v.s4() < 0) &&
               (v.s5() < 0) && (v.s6() < 0) && (v.s7() < 0) && (v.s8() < 0) && (v.s9() < 0) &&
               (v.sA() < 0) && (v.sB() < 0) && (v.sC() < 0) && (v.sD() < 0) && (v.sE() < 0) &&
               (v.sF() < 0);
    }


    ////////////////////////////////////////////////////////////////////////////////////////////////
    // vec_compare_geq
    ////////////////////////////////////////////////////////////////////////////////////////////////

    template<class T>
    inline constexpr bool vec_compare_geq(T a, T b) {
        return a >= b;
    }

    template<class T, int n, std::enable_if_t<n == 2, int> = 0>
    inline constexpr T vec_compare_geq(sycl::vec<T, n> v, sycl::vec<T, n> w) noexcept {
        return (v.x() >= w.x()) && (v.y() >= w.y());
    }

    template<class T, int n, std::enable_if_t<n == 3, int> = 0>
    inline constexpr T vec_compare_geq(sycl::vec<T, n> v, sycl::vec<T, n> w) noexcept {
        return (v.x() >= w.x()) && (v.y() >= w.y()) && (v.z() >= w.z());
    }

    template<class T, int n, std::enable_if_t<n == 4, int> = 0>
    inline constexpr T vec_compare_geq(sycl::vec<T, n> v, sycl::vec<T, n> w) noexcept {
        return (v.x() >= w.x()) && (v.y() >= w.y()) && (v.z() >= w.z()) && (v.w() >= w.w());
    }

    template<class T, int n, std::enable_if_t<n == 8, int> = 0>
    inline constexpr T vec_compare_geq(sycl::vec<T, n> v, sycl::vec<T, n> w) noexcept {
        return (v.s0() >= w.s0()) && (v.s1() >= w.s1()) && (v.s2() >= w.s2()) && (v.s3() >= w.s3()) && (v.s4() >= w.s4()) &&
               (v.s5() >= w.s5()) && (v.s6() >= w.s6()) && (v.s7() >= w.s7());
    }

    template<class T, int n, std::enable_if_t<n == 16, int> = 0>
    inline constexpr T vec_compare_geq(sycl::vec<T, n> v, sycl::vec<T, n> w) noexcept {
        return (v.s0() >= w.s0()) && (v.s1() >= w.s1()) && (v.s2() >= w.s2()) && (v.s3() >= w.s3()) && (v.s4() >= w.s4()) &&
               (v.s5() >= w.s5()) && (v.s6() >= w.s6()) && (v.s7() >= w.s7()) && (v.s8() >= w.s8()) && (v.s9() >= w.s9()) &&
               (v.sA() >= w.sA()) && (v.sB() >= w.sB()) && (v.sC() >= w.sC()) && (v.sD() >= w.sD()) && (v.sE() >= w.sE()) &&
               (v.sF() >= w.sF());
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // component_have_a_zero
    ////////////////////////////////////////////////////////////////////////////////////////////////

    template<class T>
    inline constexpr bool component_have_a_zero(T a) {
        return a == 0;
    }

    template<class T, int n, std::enable_if_t<n == 2, int> = 0>
    inline constexpr T component_have_a_zero(sycl::vec<T, n> v) noexcept {
        return (v.x() == 0) || (v.y() == 0);
    }

    template<class T, int n, std::enable_if_t<n == 3, int> = 0>
    inline constexpr T component_have_a_zero(sycl::vec<T, n> v) noexcept {
        return (v.x() == 0) || (v.y() == 0) || (v.z() == 0);
    }

    template<class T, int n, std::enable_if_t<n == 4, int> = 0>
    inline constexpr T component_have_a_zero(sycl::vec<T, n> v) noexcept {
        return (v.x() == 0) || (v.y() == 0) || (v.z() == 0) || (v.w() == 0);
    }

    template<class T, int n, std::enable_if_t<n == 8, int> = 0>
    inline constexpr T component_have_a_zero(sycl::vec<T, n> v) noexcept {
        return (v.s0() == 0) || (v.s1() == 0) || (v.s2() == 0) || (v.s3() == 0) || (v.s4() == 0) ||
               (v.s5() == 0) || (v.s6() == 0) || (v.s7() == 0);
    }

    template<class T, int n, std::enable_if_t<n == 16, int> = 0>
    inline constexpr T component_have_a_zero(sycl::vec<T, n> v) noexcept {
        return (v.s0() == 0) || (v.s1() == 0) || (v.s2() == 0) || (v.s3() == 0) || (v.s4() == 0) ||
               (v.s5() == 0) || (v.s6() == 0) || (v.s7() == 0) || (v.s8() == 0) || (v.s9() == 0) ||
               (v.sA() == 0) || (v.sB() == 0) || (v.sC() == 0) || (v.sD() == 0) || (v.sE() == 0) ||
               (v.sF() == 0);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // component_have_only_one_zero
    ////////////////////////////////////////////////////////////////////////////////////////////////

    template<class T>
    inline constexpr bool component_have_only_one_zero(T a) {
        return a == 0;
    }

    template<class T, int n, std::enable_if_t<n == 2, int> = 0>
    inline constexpr T component_have_only_one_zero(sycl::vec<T, n> v) noexcept {
        return (v.x() == 0) != (v.y() == 0);
    }

    template<class T, int n, std::enable_if_t<n == 3, int> = 0>
    inline constexpr T component_have_only_one_zero(sycl::vec<T, n> v) noexcept {
        return 1 == int{(v.x() == 0)} + int{(v.y() == 0)} + int{(v.z() == 0)};
    }

    template<class T, int n, std::enable_if_t<n == 4, int> = 0>
    inline constexpr T component_have_only_one_zero(sycl::vec<T, n> v) noexcept {
        return 1 == int{(v.x() == 0)} + int{(v.y() == 0)} + int{(v.z() == 0)} + int{(v.w() == 0)};
    }

    template<class T, int n, std::enable_if_t<n == 8, int> = 0>
    inline constexpr T component_have_only_one_zero(sycl::vec<T, n> v) noexcept {
        return 1 == int{(v.s0() == 0)} + int{(v.s1() == 0)} + int{(v.s2() == 0)} +
                        int{(v.s3() == 0)} + int{(v.s4() == 0)} + int{(v.s5() == 0)} +
                        int{(v.s6() == 0)} + int{(v.s7() == 0)};
    }

    template<class T, int n, std::enable_if_t<n == 16, int> = 0>
    inline constexpr T component_have_only_one_zero(sycl::vec<T, n> v) noexcept {
        return 1 == int{(v.s0() == 0)} + int{(v.s1() == 0)} + int{(v.s2() == 0)} +
                        int{(v.s3() == 0)} + int{(v.s4() == 0)} + int{(v.s5() == 0)} +
                        int{(v.s6() == 0)} + int{(v.s7() == 0)} + int{(v.s8() == 0)} +
                        int{(v.s9() == 0)} + int{(v.sA() == 0)} + int{(v.sB() == 0)} +
                        int{(v.sC() == 0)} + int{(v.sD() == 0)} + int{(v.sE() == 0)} +
                        int{(v.sF() == 0)};
    }


    ////////////////////////////////////////////////////////////////////////////////////////////////
    // component_have_at_most_one_zero
    ////////////////////////////////////////////////////////////////////////////////////////////////

    template<class T>
    inline constexpr bool component_have_at_most_one_zero(T a) {
        return true;
    }

    template<class T, int n, std::enable_if_t<n == 2, int> = 0>
    inline constexpr T component_have_at_most_one_zero(sycl::vec<T, n> v) noexcept {
        return 2 > int{(v.x() == 0)} + int{(v.y() == 0)};
    }

    template<class T, int n, std::enable_if_t<n == 3, int> = 0>
    inline constexpr T component_have_at_most_one_zero(sycl::vec<T, n> v) noexcept {
        return 2 > int{(v.x() == 0)} + int{(v.y() == 0)} + int{(v.z() == 0)};
    }

    template<class T, int n, std::enable_if_t<n == 4, int> = 0>
    inline constexpr T component_have_at_most_one_zero(sycl::vec<T, n> v) noexcept {
        return 2 > int{(v.x() == 0)} + int{(v.y() == 0)} + int{(v.z() == 0)} + int{(v.w() == 0)};
    }

    template<class T, int n, std::enable_if_t<n == 8, int> = 0>
    inline constexpr T component_have_at_most_one_zero(sycl::vec<T, n> v) noexcept {
        return 2 > int{(v.s0() == 0)} + int{(v.s1() == 0)} + int{(v.s2() == 0)} +
                        int{(v.s3() == 0)} + int{(v.s4() == 0)} + int{(v.s5() == 0)} +
                        int{(v.s6() == 0)} + int{(v.s7() == 0)};
    }

    template<class T, int n, std::enable_if_t<n == 16, int> = 0>
    inline constexpr T component_have_at_most_one_zero(sycl::vec<T, n> v) noexcept {
        return 2 > int{(v.s0() == 0)} + int{(v.s1() == 0)} + int{(v.s2() == 0)} +
                        int{(v.s3() == 0)} + int{(v.s4() == 0)} + int{(v.s5() == 0)} +
                        int{(v.s6() == 0)} + int{(v.s7() == 0)} + int{(v.s8() == 0)} +
                        int{(v.s9() == 0)} + int{(v.sA() == 0)} + int{(v.sB() == 0)} +
                        int{(v.sC() == 0)} + int{(v.sD() == 0)} + int{(v.sE() == 0)} +
                        int{(v.sF() == 0)};
    }

} // namespace shambase