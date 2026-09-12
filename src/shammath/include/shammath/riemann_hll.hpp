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
        const typename FSpec::Tprim &primL,
        const typename FSpec::Tprim &primR,
        const typename FSpec::Tvec &n) {
        const auto csL = fspec.sound_speed(primL);
        const auto csR = fspec.sound_speed(primR);

        const auto vnL = fspec.vn(primL, n);
        const auto vnR = fspec.vn(primR, n);

        // Teyssier form
        // const auto S_L = sham::min(vnL, vnR) - sham::max(csL, csR);
        // const auto S_R = sham::max(vnL, vnR) + sham::max(csL, csR);

        // Toro form Equation (10.48)
        const auto S_L = sham::min(vnL - csL, vnR - csR);
        const auto S_R = sham::max(vnL + csL, vnR + csR);

        const auto fluxL = fspec.flux(primL, n, vnL);
        const auto fluxR = fspec.flux(primR, n, vnR);

        // Equation (10.26) from Toro 3rd Edition , Springer 2009
        // const auto S_L_upwind = sham::min(S_L, 0.0);
        // const auto S_R_upwind = sham::max(S_R, 0.0);
        // const auto S_norm     = 1.0 / (S_R_upwind - S_L_upwind);
        // return (fluxL * S_R_upwind - fluxR * S_L_upwind
        //         + (consR - consL) * S_R_upwind * S_L_upwind)
        //        * S_norm;

        if (S_L >= 0)
            return fluxL;
        else if (S_R <= 0)
            return fluxR;
        else {
            // Only the intermediate (star) state needs the conservative form, so it is
            // formed here rather than at the call site (which only has primitives).
            const auto consL  = fspec.prim_to_cons(primL);
            const auto consR  = fspec.prim_to_cons(primR);
            const auto S_norm = 1.0 / (S_R - S_L);
            return (fluxL * S_R - fluxR * S_L + (consR - consL) * S_R * S_L) * S_norm;
        }
    }

} // namespace shammath
