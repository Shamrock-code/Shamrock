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
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Thomas Guillet (T.A.Guillet@exeter.ac.uk) --no git blame--
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief HLL Riemann solver for the gas equations
 * From original version by Thomas Guillet (T.A.Guillet@exeter.ac.uk)
 */

#include "shammath/riemann_common.hpp"

namespace shammath {

    template<class Tcons>
    inline constexpr auto hll_flux_x(
        const Tcons consL, const Tcons consR, const typename Tcons::Tscal gamma) {
        const auto primL = cons_to_prim(consL, gamma);
        const auto primR = cons_to_prim(consR, gamma);

        const auto csL = sound_speed(primL, gamma);
        const auto csR = sound_speed(primR, gamma);

        // Teyssier form
        // const auto S_L = sham::min(primL.vel[0], primR.vel[0]) - sham::max(csL, csR);
        // const auto S_R = sham::max(primL.vel[0], primR.vel[0]) + sham::max(csL, csR);

        // Toro form Equation (10.48)
        const auto S_L = sham::min(primL.vel[0] - csL, primR.vel[0] - csR);
        const auto S_R = sham::max(primL.vel[0] + csL, primR.vel[0] + csR);

        const auto fluxL = hydro_flux_x(consL, gamma);
        const auto fluxR = hydro_flux_x(consR, gamma);

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
                const auto S_norm = 1.0 / (S_R - S_L);
                return (fluxL * S_R - fluxR * S_L + (consR - consL) * S_R * S_L) * S_norm;
            }
        };

        return hll_flux();
    }

    template<class Tcons>
    inline constexpr Tcons hll_flux_y(Tcons cL, Tcons cR, typename Tcons::Tscal gamma) {
        return x_to_y(hll_flux_x(y_to_x(cL), y_to_x(cR), gamma));
    }

    template<class Tcons>
    inline constexpr Tcons hll_flux_z(Tcons cL, Tcons cR, typename Tcons::Tscal gamma) {
        return x_to_z(hll_flux_x(z_to_x(cL), z_to_x(cR), gamma));
    }

    template<class Tcons>
    inline constexpr Tcons hll_flux_mx(Tcons cL, Tcons cR, typename Tcons::Tscal gamma) {
        return invert_axis(hll_flux_x(invert_axis(cL), invert_axis(cR), gamma));
    }

    template<class Tcons>
    inline constexpr Tcons hll_flux_my(Tcons cL, Tcons cR, typename Tcons::Tscal gamma) {
        return invert_axis(hll_flux_y(invert_axis(cL), invert_axis(cR), gamma));
    }

    template<class Tcons>
    inline constexpr Tcons hll_flux_mz(Tcons cL, Tcons cR, typename Tcons::Tscal gamma) {
        return invert_axis(hll_flux_z(invert_axis(cL), invert_axis(cR), gamma));
    }

} // namespace shammath
