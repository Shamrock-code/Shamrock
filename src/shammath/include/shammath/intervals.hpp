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
 * @file intervals.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/type_traits.hpp"
#include "shambackends/math.hpp"
#include "shambackends/type_traits.hpp"

namespace shammath {

    /**
     * @brief return true if val is in [min,max[
     *
     * @tparam T
     * @param val
     * @param min
     * @param max
     * @return true
     * @return false
     */
    template<class T, std::enable_if_t<sham::is_valid_sycl_base_type<T>, int> = 0>
    inline bool is_in_half_open(T val, T min, T max) {
        return (val >= min) && (val < max);
    }

    /**
     * @brief Check if value is in [min,max) for 2D vectors
     *
     * @tparam T Component type
     * @param val Value to test
     * @param min Lower bounds
     * @param max Upper bounds
     * @return true if val is in [min, max) for all components
     */
    template<class T>
    inline bool is_in_half_open(sycl::vec<T, 2> val, sycl::vec<T, 2> min, sycl::vec<T, 2> max) {
        return (
            is_in_half_open(val.x(), min.x(), max.x())
            && is_in_half_open(val.y(), min.y(), max.y()));
    }

    /**
     * @brief Check if value is in [min,max) for 3D vectors
     *
     * @tparam T Component type
     * @param val Value to test
     * @param min Lower bounds
     * @param max Upper bounds
     * @return true if val is in [min, max) for all components
     */
    template<class T>
    inline bool is_in_half_open(sycl::vec<T, 3> val, sycl::vec<T, 3> min, sycl::vec<T, 3> max) {
        return (
            is_in_half_open(val.x(), min.x(), max.x()) && is_in_half_open(val.y(), min.y(), max.y())
            && is_in_half_open(val.z(), min.z(), max.z()));
    }

    /**
     * @brief Check if value is in [min,max) for 4D vectors
     *
     * @tparam T Component type
     * @param val Value to test
     * @param min Lower bounds
     * @param max Upper bounds
     * @return true if val is in [min, max) for all components
     */
    template<class T>
    inline bool is_in_half_open(sycl::vec<T, 4> val, sycl::vec<T, 4> min, sycl::vec<T, 4> max) {
        return (
            is_in_half_open(val.x(), min.x(), max.x()) && is_in_half_open(val.y(), min.y(), max.y())
            && is_in_half_open(val.z(), min.z(), max.z())
            && is_in_half_open(val.w(), min.w(), max.w()));
    }

    /**
     * @brief Check if value is in [min,max) for 8D vectors
     *
     * @tparam T Component type
     * @param val Value to test
     * @param min Lower bounds
     * @param max Upper bounds
     * @return true if val is in [min, max) for all components
     */
    template<class T>
    inline bool is_in_half_open(sycl::vec<T, 8> val, sycl::vec<T, 8> min, sycl::vec<T, 8> max) {
        return (
            is_in_half_open(val.s0(), min.s0(), max.s0())
            && is_in_half_open(val.s1(), min.s1(), max.s1())
            && is_in_half_open(val.s2(), min.s2(), max.s2())
            && is_in_half_open(val.s3(), min.s3(), max.s3())
            && is_in_half_open(val.s4(), min.s4(), max.s4())
            && is_in_half_open(val.s5(), min.s5(), max.s5())
            && is_in_half_open(val.s6(), min.s6(), max.s6())
            && is_in_half_open(val.s7(), min.s7(), max.s7()));
    }

    /**
     * @brief Check if value is in [min,max) for 16D vectors
     *
     * @tparam T Component type
     * @param val Value to test
     * @param min Lower bounds
     * @param max Upper bounds
     * @return true if val is in [min, max) for all components
     */
    template<class T>
    inline bool is_in_half_open(sycl::vec<T, 16> val, sycl::vec<T, 16> min, sycl::vec<T, 16> max) {
        return (
            is_in_half_open(val.s0(), min.s0(), max.s0())
            && is_in_half_open(val.s1(), min.s1(), max.s1())
            && is_in_half_open(val.s2(), min.s2(), max.s2())
            && is_in_half_open(val.s3(), min.s3(), max.s3())
            && is_in_half_open(val.s4(), min.s4(), max.s4())
            && is_in_half_open(val.s5(), min.s5(), max.s5())
            && is_in_half_open(val.s6(), min.s6(), max.s6())
            && is_in_half_open(val.s7(), min.s7(), max.s7())
            && is_in_half_open(val.s8(), min.s8(), max.s8())
            && is_in_half_open(val.s9(), min.s9(), max.s9())
            && is_in_half_open(val.sA(), min.sA(), max.sA())
            && is_in_half_open(val.sB(), min.sB(), max.sB())
            && is_in_half_open(val.sC(), min.sC(), max.sC())
            && is_in_half_open(val.sD(), min.sD(), max.sD())
            && is_in_half_open(val.sE(), min.sE(), max.sE())
            && is_in_half_open(val.sF(), min.sF(), max.sF()));
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // domain_are_connected
    ////////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @brief Check if two 1D intervals share boundary or overlap
     *
     * @tparam T Scalar type
     * @param bmin1 Minimum of first interval
     * @param bmax1 Maximum of first interval
     * @param bmin2 Minimum of second interval
     * @param bmax2 Maximum of second interval
     * @return true if intervals are connected (share boundary or overlap)
     */
    template<class T, std::enable_if_t<sham::is_valid_sycl_base_type<T>, int> = 0>
    inline bool domain_are_connected(T bmin1, T bmax1, T bmin2, T bmax2) {
        return sham::max(bmin1, bmin2) <= sham::min(bmax1, bmax2);
    }

    /**
     * @brief Check if two 2D boxes share boundary or overlap
     *
     * @tparam T Component type
     * @param bmin1 Minimum bounds of first box
     * @param bmax1 Maximum bounds of first box
     * @param bmin2 Minimum bounds of second box
     * @param bmax2 Maximum bounds of second box
     * @return true if boxes are connected (share boundary or overlap)
     */
    template<class T>
    inline bool domain_are_connected(
        sycl::vec<T, 2> bmin1,
        sycl::vec<T, 2> bmax1,
        sycl::vec<T, 2> bmin2,
        sycl::vec<T, 2> bmax2) {

        return (
            domain_are_connected(bmin1.x(), bmax1.x(), bmin2.x(), bmax2.x())
            && domain_are_connected(bmin1.y(), bmax1.y(), bmin2.y(), bmax2.y()));
    }

    /**
     * @brief Check if two 3D boxes share boundary or overlap
     *
     * @tparam T Component type
     * @param bmin1 Minimum bounds of first box
     * @param bmax1 Maximum bounds of first box
     * @param bmin2 Minimum bounds of second box
     * @param bmax2 Maximum bounds of second box
     * @return true if boxes are connected (share boundary or overlap)
     */
    template<class T>
    inline bool domain_are_connected(
        sycl::vec<T, 3> bmin1,
        sycl::vec<T, 3> bmax1,
        sycl::vec<T, 3> bmin2,
        sycl::vec<T, 3> bmax2) {

        return (
            domain_are_connected(bmin1.x(), bmax1.x(), bmin2.x(), bmax2.x())
            && domain_are_connected(bmin1.y(), bmax1.y(), bmin2.y(), bmax2.y())
            && domain_are_connected(bmin1.z(), bmax1.z(), bmin2.z(), bmax2.z()));
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // domain_have_intersect
    ////////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @brief Check if two 1D intervals have non-empty intersection
     *
     * @tparam T Scalar type
     * @param bmin1 Minimum of first interval
     * @param bmax1 Maximum of first interval
     * @param bmin2 Minimum of second interval
     * @param bmax2 Maximum of second interval
     * @return true if intervals have non-empty intersection
     */
    template<class T, std::enable_if_t<sham::is_valid_sycl_base_type<T>, int> = 0>
    inline bool domain_have_intersect(T bmin1, T bmax1, T bmin2, T bmax2) {
        return sham::max(bmin1, bmin2) < sham::min(bmax1, bmax2);
    }

    /**
     * @brief Check if two 2D boxes have non-empty intersection
     *
     * @tparam T Component type
     * @param bmin1 Minimum bounds of first box
     * @param bmax1 Maximum bounds of first box
     * @param bmin2 Minimum bounds of second box
     * @param bmax2 Maximum bounds of second box
     * @return true if boxes have non-empty intersection
     */
    template<class T>
    inline bool domain_have_intersect(
        sycl::vec<T, 2> bmin1,
        sycl::vec<T, 2> bmax1,
        sycl::vec<T, 2> bmin2,
        sycl::vec<T, 2> bmax2) {

        return (
            domain_have_intersect(bmin1.x(), bmax1.x(), bmin2.x(), bmax2.x())
            && domain_have_intersect(bmin1.y(), bmax1.y(), bmin2.y(), bmax2.y()));
    }

    /**
     * @brief Check if two 3D boxes have non-empty intersection
     *
     * @tparam T Component type
     * @param bmin1 Minimum bounds of first box
     * @param bmax1 Maximum bounds of first box
     * @param bmin2 Minimum bounds of second box
     * @param bmax2 Maximum bounds of second box
     * @return true if boxes have non-empty intersection
     */
    template<class T>
    inline bool domain_have_intersect(
        sycl::vec<T, 3> bmin1,
        sycl::vec<T, 3> bmax1,
        sycl::vec<T, 3> bmin2,
        sycl::vec<T, 3> bmax2) {

        return (
            domain_have_intersect(bmin1.x(), bmax1.x(), bmin2.x(), bmax2.x())
            && domain_have_intersect(bmin1.y(), bmax1.y(), bmin2.y(), bmax2.y())
            && domain_have_intersect(bmin1.z(), bmax1.z(), bmin2.z(), bmax2.z()));
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // domain_have_common_face
    ////////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @brief Check if two 3D boxes share a common face
     *
     * @tparam T Component type
     * @param bmin1 Minimum bounds of first box
     * @param bmax1 Maximum bounds of first box
     * @param bmin2 Minimum bounds of second box
     * @param bmax2 Maximum bounds of second box
     * @return true if boxes share a common face
     */
    template<class T>
    inline bool domain_have_common_face(
        sycl::vec<T, 3> bmin1,
        sycl::vec<T, 3> bmax1,
        sycl::vec<T, 3> bmin2,
        sycl::vec<T, 3> bmax2) {
        u32 cnt = ((domain_have_common_face(bmin1.x(), bmax1.x(), bmin2.x(), bmax2.x())) ? 1 : 0)
                  + ((domain_have_common_face(bmin1.y(), bmax1.y(), bmin2.y(), bmax2.y())) ? 1 : 0)
                  + ((domain_have_common_face(bmin1.z(), bmax1.z(), bmin2.z(), bmax2.z())) ? 1 : 0);

        return cnt > 1;
    }

} // namespace shammath
