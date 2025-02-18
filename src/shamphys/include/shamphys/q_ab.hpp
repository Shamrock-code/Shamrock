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
 * @file q_ab.hpp
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shambackends/typeAliasVec.hpp"
#include "shambackends/vec.hpp"
#include "shambackends/math.hpp"
#include "shamphys/mhd.hpp"

namespace shamphys {
template<class Tvec, class Tscal>
    class q_ab_lambdas {
        public:
        static constexpr auto lambda_qav
            = [](Tscal rho, Tscal cs, Tscal v_scal_rhat, Tscal alpha_AV, Tscal beta_AV) {
                  Tscal abs_v_ab_r_ab = sycl::fabs(v_scal_rhat);
                  Tscal vsig          = alpha_AV * cs + beta_AV * abs_v_ab_r_ab;
                  return sham::max(-Tscal(0.5) * rho * vsig * v_scal_rhat, Tscal(0));
              };

        static constexpr auto lambda_qav_mhd
            = [](Tscal rho, Tscal cs, Tscal v_scal_rhat, Tscal alpha_AV, 
                 Tscal beta_AV, Tvec r_ab_unit, Tvec v_ab, Tscal cs_a, Tscal B_a, Tscal mu_0) {
                  Tscal abs_v_ab_r_ab = sycl::fabs(v_scal_rhat);
                  Tscal vsig          = MHD_physics<Tvec, Tscal>::vsig_MHD(v_ab, r_ab_unit,cs_a, B_a,rho,mu_0,alpha_AV,beta_AV);
                  return sham::max(-Tscal(0.5) * rho * vsig * v_scal_rhat, Tscal(0));
              };

        static constexpr auto  lambda_qav_disc = [](
            Tscal rho, Tscal cs,  Tscal v_scal_rhat, Tscal alpha_AV, Tscal vsig, Tscal h, Tscal rab) {
            Tscal q_av_d;
            Tscal rho1   = 1. / rho;
            Tscal rabinv = sham::inv_sat_positive(rab);
            Tscal prefact = -Tscal(0.5) * rho * sham::abs(rabinv) * h;
            Tscal vsig_disc = (v_scal_rhat < Tscal(0)) ? vsig : (alpha_AV * cs);
            q_av_d = prefact * vsig_disc * v_scal_rhat;
            return q_av_d;
            };

    };
}