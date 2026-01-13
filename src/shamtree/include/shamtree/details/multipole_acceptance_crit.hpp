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
 * @file multipole_acceptance_crit.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shammath/AABB.hpp"

namespace shamtree::details {

    template<class Tvec, class Tscal>
    inline static bool mac_std_opti(
        shammath::AABB<Tvec> a, shammath::AABB<Tvec> b, Tscal theta_crit) {
        Tvec s_a      = (a.upper - a.lower) / 2;
        Tvec s_b      = (b.upper - b.lower) / 2;
        Tvec r_a      = (a.upper + a.lower) / 2;
        Tvec r_b      = (b.upper + b.lower) / 2;
        Tvec delta_ab = r_a - r_b;

        Tscal delta_ab_sq = sham::dot(delta_ab, delta_ab);

        if (delta_ab_sq == 0) {
            return false;
        }

        Tscal s_a_max = sham::max_component(s_a);
        Tscal s_b_max = sham::max_component(s_b);

        Tscal sum_s = s_a_max + s_b_max;

        Tscal theta_sq = sum_s * sum_s / delta_ab_sq;

        return theta_sq < theta_crit * theta_crit;
    }

    template<class Tvec, class Tscal>
    inline static bool mac_std(shammath::AABB<Tvec> a, shammath::AABB<Tvec> b, Tscal theta_crit) {
        Tvec s_a      = (a.upper - a.lower) / 2;
        Tvec s_b      = (b.upper - b.lower) / 2;
        Tvec r_a      = (a.upper + a.lower) / 2;
        Tvec r_b      = (b.upper + b.lower) / 2;
        Tvec delta_ab = r_a - r_b;

        Tscal delta_ab_sq = sham::dot(delta_ab, delta_ab);

        if (delta_ab_sq == 0) {
            return false;
        }

        Tscal s_a_sq = sham::dot(s_a, s_a);
        Tscal s_b_sq = sham::dot(s_b, s_b);

        Tscal theta_sq = (sycl::sqrt(s_a_sq) + sycl::sqrt(s_b_sq)) / sycl::sqrt(delta_ab_sq);

        return theta_sq < theta_crit;
    }

    template<class Tvec, class Tscal>
    inline static bool mac_quadratic(
        shammath::AABB<Tvec> a, shammath::AABB<Tvec> b, Tscal theta_crit) {
        Tvec s_a      = (a.upper - a.lower);
        Tvec s_b      = (b.upper - b.lower);
        Tvec r_a      = (a.upper + a.lower) / 2;
        Tvec r_b      = (b.upper + b.lower) / 2;
        Tvec delta_ab = r_a - r_b;

        Tscal delta_ab_sq = sham::dot(delta_ab, delta_ab);

        if (delta_ab_sq == 0) {
            return false;
        }

        Tscal s_a_sq = sham::dot(s_a, s_a);
        Tscal s_b_sq = sham::dot(s_b, s_b);

        Tscal theta_sq = (s_a_sq + s_b_sq) / delta_ab_sq;

        return theta_sq < theta_crit * theta_crit;
    }

    template<class Tvec, class Tscal>
    inline static bool mac(shammath::AABB<Tvec> a, shammath::AABB<Tvec> b, Tscal theta_crit) {
        return mac_std_opti(a, b, theta_crit);
    }

} // namespace shamtree::details
