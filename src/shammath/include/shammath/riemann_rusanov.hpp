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
    template<FluidStateSpec FSpec>
    inline constexpr typename FSpec::Tcons rusanov_flux(
        const FSpec &fspec,
        const typename FSpec::Tprim &primL,
        const typename FSpec::Tprim &primR,
        const typename FSpec::Tvec &n) {
        const auto csL = fspec.sound_speed(primL);
        const auto csR = fspec.sound_speed(primR);

        const auto vnL = fspec.vn(primL, n);
        const auto vnR = fspec.vn(primR, n);

        // Equation (10.56) from Toro 3rd Edition , Springer 2009
        const auto S = sham::max((sham::abs(vnL) + csL), (sham::abs(vnR) + csR));

        const auto fL = fspec.flux(primL, n, vnL);
        const auto fR = fspec.flux(primR, n, vnR);

        const auto consL = fspec.prim_to_cons(primL);
        const auto consR = fspec.prim_to_cons(primR);

        // Equation (10.55) from Toro 3rd Edition , Springer 2009
        return 0.5 * ((fL + fR) - (consR - consL) * S);
    }

} // namespace shammath
