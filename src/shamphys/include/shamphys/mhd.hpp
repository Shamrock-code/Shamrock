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
 * @file mhd.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 */

#include "shambackends/math.hpp"

namespace shamphys {

    template<class Tvec, class Tscal>
    struct MHD_physics {
        inline static constexpr Tscal v_alfven(Tvec B, Tscal rho, Tscal mu_0) {
            return sycl::sqrt(sycl::dot(B, B) / (mu_0 * rho));
        };

        inline static constexpr Tscal v_shock(Tscal cs, Tvec B, Tscal rho, Tscal mu_0) {
            return sycl::sqrt(cs * cs + v_alfven(B, rho, mu_0) * v_alfven(B, rho, mu_0));
        };

        inline static constexpr Tscal vsigB(Tvec v_ab, Tvec r_ab_unit) {
            Tvec v_cross_r = sycl::cross(v_ab, r_ab_unit);
            Tscal vsig_B_a = sycl::sqrt(
                v_cross_r[0] * v_cross_r[0] + v_cross_r[1] * v_cross_r[1]
                + v_cross_r[2] * v_cross_r[2]);
            return vsig_B_a;
        };

        inline static constexpr Tscal vsig_MHD(
            Tvec v_ab,
            Tvec r_ab_unit,
            Tscal cs_a,
            Tvec B_a,
            Tscal rho_a,
            Tscal mu_0,
            Tscal alpha_av,
            Tscal beta_av) {
            Tscal v_ab_r_ab     = sycl::dot(v_ab, r_ab_unit);
            Tscal abs_v_ab_r_ab = sycl::fabs(v_ab_r_ab);
            Tscal v_a           = v_shock(cs_a, B_a, rho_a, mu_0);
            Tscal vsig          = alpha_av * v_a + beta_av * abs_v_ab_r_ab;

            return vsig;
        }
    };

} // namespace shamphys
