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
 * @file riemann_rusanov.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Thomas Guillet (T.A.Guillet@exeter.ac.uk) --no git blame--
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Rusanov Riemann solver for the gas equations
 * From original version by Thomas Guillet (T.A.Guillet@exeter.ac.uk)
 */

#include "shammath/riemann_common.hpp"

namespace shammath {

    // template<class Tcons>
    // inline constexpr Tcons rusanov_flux_x(Tcons cL, Tcons cR, typename Tcons::Tscal gamma) {
    //     Tcons flux;

    //     const auto primL = cons_to_prim(cL, gamma);
    //     const auto primR = cons_to_prim(cR, gamma);

    //     const auto csL = sound_speed(primL, gamma);
    //     const auto csR = sound_speed(primR, gamma);

    //     const auto S = sham::max(
    //         sham::max(sham::abs(primL.vel[0] - csL), sham::abs(primR.vel[0] - csR)),
    //         sham::max(sham::abs(primL.vel[0] + csL), sham::abs(primR.vel[0] + csR)));

    //     const auto fL = hydro_flux_x(cL, gamma);
    //     const auto fR = hydro_flux_x(cR, gamma);

    //     return (fL + fR) * 0.5 - (cR - cL) * S;
    // }

    template<class Tcons>
    inline constexpr Tcons rusanov_flux_x(Tcons cL, Tcons cR, typename Tcons::Tscal gamma) {
        Tcons flux;

        const auto primL = cons_to_prim(cL, gamma);
        const auto primR = cons_to_prim(cR, gamma);

        const auto csL = sound_speed(primL, gamma);
        const auto csR = sound_speed(primR, gamma);

        // Equation (10.56) from Toro 3rd Edition , Springer 2009
        const auto S = sham::max((sham::abs(primL.vel[0]) + csL), (sham::abs(primR.vel[0]) + csR));

        const auto fL = hydro_flux_x(cL, gamma);
        const auto fR = hydro_flux_x(cR, gamma);

        // Equation (10.55) from Toro 3rd Edition , Springer 2009
        return 0.5 * ((fL + fR) - (cR - cL) * S);
    }

    template<class Tcons>
    inline constexpr Tcons rusanov_flux_y(Tcons cL, Tcons cR, typename Tcons::Tscal gamma) {
        return x_to_y(rusanov_flux_x(y_to_x(cL), y_to_x(cR), gamma));
    }

    template<class Tcons>
    inline constexpr Tcons rusanov_flux_z(Tcons cL, Tcons cR, typename Tcons::Tscal gamma) {
        return x_to_z(rusanov_flux_x(z_to_x(cL), z_to_x(cR), gamma));
    }

    template<class Tcons>
    inline constexpr Tcons rusanov_flux_mx(Tcons cL, Tcons cR, typename Tcons::Tscal gamma) {
        return invert_axis(rusanov_flux_x(invert_axis(cL), invert_axis(cR), gamma));
    }

    template<class Tcons>
    inline constexpr Tcons rusanov_flux_my(Tcons cL, Tcons cR, typename Tcons::Tscal gamma) {
        return invert_axis(rusanov_flux_y(invert_axis(cL), invert_axis(cR), gamma));
    }

    template<class Tcons>
    inline constexpr Tcons rusanov_flux_mz(Tcons cL, Tcons cR, typename Tcons::Tscal gamma) {
        return invert_axis(rusanov_flux_z(invert_axis(cL), invert_axis(cR), gamma));
    }

} // namespace shammath
