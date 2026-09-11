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
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr) --no git blame--
 * @author Thomas Guillet (T.A.Guillet@exeter.ac.uk) --no git blame--
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Rusanov Riemann solver for the gas equations
 * From original version by Thomas Guillet (T.A.Guillet@exeter.ac.uk)
 */

#include "shammath/riemann_common.hpp"

namespace shammath {

    /**
     * @brief Rusanov flux across a face with unit normal n
     */
    template<class Tprim>
    inline constexpr auto rusanov_flux(
        Tprim primL, Tprim primR, typename Tprim::Tscal gamma, typename Tprim::Tvec n) {
        const auto csL = sound_speed(primL, gamma);
        const auto csR = sound_speed(primR, gamma);

        const auto vnL = n[0] * primL.vel[0] + n[1] * primL.vel[1] + n[2] * primL.vel[2];
        const auto vnR = n[0] * primR.vel[0] + n[1] * primR.vel[1] + n[2] * primR.vel[2];

        // Equation (10.56) from Toro 3rd Edition , Springer 2009
        const auto S = sham::max((sham::abs(vnL) + csL), (sham::abs(vnR) + csR));

        const auto fL = hydro_flux_n(primL, n, vnL, gamma);
        const auto fR = hydro_flux_n(primR, n, vnR, gamma);

        const auto consL = prim_to_cons(primL, gamma);
        const auto consR = prim_to_cons(primR, gamma);

        // Equation (10.55) from Toro 3rd Edition , Springer 2009
        return 0.5 * ((fL + fR) - (consR - consL) * S);
    }

} // namespace shammath
