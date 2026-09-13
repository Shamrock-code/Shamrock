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
 * @file riemann_hll.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr) --no git blame--
 * @author Thomas Guillet (T.A.Guillet@exeter.ac.uk) --no git blame--
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief HLL Riemann solver for the gas equations
 * From original version by Thomas Guillet (T.A.Guillet@exeter.ac.uk)
 */

#include "shammath/riemann_common.hpp"

namespace shammath {

    /**
     * @brief HLL flux across a face with unit normal n
     */
    template<FluidStateSpec FSpec>
    inline constexpr typename FSpec::Tcons hll_flux(
        const FSpec &fspec,
        const typename FSpec::Tprim &prim_l,
        const typename FSpec::Tprim &prim_r,
        const typename FSpec::Tvec &n) {
        const auto cs_l = fspec.sound_speed(prim_l);
        const auto cs_r = fspec.sound_speed(prim_r);

        const auto vn_l = fspec.vn(prim_l, n);
        const auto vn_r = fspec.vn(prim_r, n);

        // NOLINTBEGIN(readability-identifier-naming)

        // Teyssier form
        // const auto S_l = sham::min(vn_l, vn_r) - sham::max(cs_l, cs_r);
        // const auto S_r = sham::max(vn_l, vn_r) + sham::max(cs_l, cs_r);

        // Toro form Equation (10.48)
        const auto S_l = sham::min(vn_l - cs_l, vn_r - cs_r);
        const auto S_r = sham::max(vn_l + cs_l, vn_r + cs_r);

        // NOLINTEND(readability-identifier-naming)

        const auto flux_l = fspec.flux(prim_l, n, vn_l);
        const auto flux_r = fspec.flux(prim_r, n, vn_r);

        // Equation (10.26) from Toro 3rd Edition , Springer 2009
        // const auto S_l_upwind = sham::min(S_l, 0.0);
        // const auto S_r_upwind = sham::max(S_r, 0.0);
        // const auto s_norm     = 1.0 / (S_r_upwind - S_l_upwind);
        // return (flux_l * S_r_upwind - flux_r * S_l_upwind
        //         + (cons_r - cons_l) * S_r_upwind * S_l_upwind)
        //        * s_norm;

        if (S_l >= 0)
            return flux_l;
        else if (S_r <= 0)
            return flux_r;
        else {
            // Only the intermediate (star) state needs the conservative form, so it is
            // formed here rather than at the call site (which only has primitives).
            const auto cons_l = fspec.prim_to_cons(prim_l);
            const auto cons_r = fspec.prim_to_cons(prim_r);
            const auto s_norm = 1.0 / (S_r - S_l);
            return (flux_l * S_r - flux_r * S_l + (cons_r - cons_l) * S_r * S_l) * s_norm;
        }
    }

} // namespace shammath
