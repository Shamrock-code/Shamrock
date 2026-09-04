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

    template<class Tprim>
    inline constexpr auto hll_flux_x(
        const Tprim primL, const Tprim primR, const typename Tprim::Tscal gamma) {
        const auto csL = sound_speed(primL, gamma);
        const auto csR = sound_speed(primR, gamma);

        // Teyssier form
        // const auto S_L = sham::min(primL.vel[0], primR.vel[0]) - sham::max(csL, csR);
        // const auto S_R = sham::max(primL.vel[0], primR.vel[0]) + sham::max(csL, csR);

        // Toro form Equation (10.48)
        const auto S_L = sham::min(primL.vel[0] - csL, primR.vel[0] - csR);
        const auto S_R = sham::max(primL.vel[0] + csL, primR.vel[0] + csR);

        const auto fluxL = hydro_flux_x(primL, gamma);
        const auto fluxR = hydro_flux_x(primR, gamma);

        // Equation (10.26) from Toro 3rd Edition , Springer 2009
        auto hll_flux = [=]() {
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
                const auto consL  = prim_to_cons(primL, gamma);
                const auto consR  = prim_to_cons(primR, gamma);
                const auto S_norm = 1.0 / (S_R - S_L);
                return (fluxL * S_R - fluxR * S_L + (consR - consL) * S_R * S_L) * S_norm;
            }
        };

        return hll_flux();
    }

    template<class Tprim>
    inline constexpr auto hll_flux_y(Tprim pL, Tprim pR, typename Tprim::Tscal gamma) {
        return x_to_y(hll_flux_x(prim_y_to_x(pL), prim_y_to_x(pR), gamma));
    }

    template<class Tprim>
    inline constexpr auto hll_flux_z(Tprim pL, Tprim pR, typename Tprim::Tscal gamma) {
        return x_to_z(hll_flux_x(prim_z_to_x(pL), prim_z_to_x(pR), gamma));
    }

    template<class Tprim>
    inline constexpr auto hll_flux_mx(Tprim pL, Tprim pR, typename Tprim::Tscal gamma) {
        return invert_axis(hll_flux_x(prim_invert_axis(pL), prim_invert_axis(pR), gamma));
    }

    template<class Tprim>
    inline constexpr auto hll_flux_my(Tprim pL, Tprim pR, typename Tprim::Tscal gamma) {
        return invert_axis(hll_flux_y(prim_invert_axis(pL), prim_invert_axis(pR), gamma));
    }

    template<class Tprim>
    inline constexpr auto hll_flux_mz(Tprim pL, Tprim pR, typename Tprim::Tscal gamma) {
        return invert_axis(hll_flux_z(prim_invert_axis(pL), prim_invert_axis(pR), gamma));
    }

} // namespace shammath
