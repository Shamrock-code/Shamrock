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
 * @file q_ab.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shambackends/math.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shambackends/vec.hpp"
#include "shamphys/mhd.hpp"

namespace shamrock::sph {

    /**
     * @brief \cite Phantom_2018 eq.40
     *
     * @tparam Tscal
     * @param rho
     * @param vsig
     * @param v_scal_rhat
     * @return Tscal
     */
    template<class Tscal>
    inline constexpr Tscal q_av(const Tscal &rho, const Tscal &vsig, const Tscal &v_scal_rhat) {
        return sham::max(-Tscal(0.5) * rho * vsig * v_scal_rhat, Tscal(0));
    }

    template<class Tscal>
    inline constexpr Tscal q_av_disc(
        const Tscal &rho,
        const Tscal &h,
        const Tscal &rab,
        const Tscal &alpha_av,
        const Tscal &cs,
        const Tscal &vsig,
        const Tscal &v_scal_rhat) {
        Tscal q_av_d;
        Tscal rho1   = 1. / rho;
        Tscal rabinv = sham::inv_sat_positive(rab);

        Tscal prefact = -Tscal(0.5) * rho * sham::abs(rabinv) * h;

        Tscal vsig_disc = (v_scal_rhat < Tscal(0)) ? vsig : (alpha_av * cs);

        q_av_d = prefact * vsig_disc * v_scal_rhat;

        return q_av_d;
    }

} // namespace shamrock::sph
