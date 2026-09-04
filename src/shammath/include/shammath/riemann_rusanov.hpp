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

    template<class Tprim>
    inline constexpr auto rusanov_flux_x(Tprim primL, Tprim primR, typename Tprim::Tscal gamma) {
        const auto csL = sound_speed(primL, gamma);
        const auto csR = sound_speed(primR, gamma);

        // Equation (10.56) from Toro 3rd Edition , Springer 2009
        const auto S = sham::max((sham::abs(primL.vel[0]) + csL), (sham::abs(primR.vel[0]) + csR));

        const auto fL = hydro_flux_x(primL, gamma);
        const auto fR = hydro_flux_x(primR, gamma);

        const auto consL = prim_to_cons(primL, gamma);
        const auto consR = prim_to_cons(primR, gamma);

        // Equation (10.55) from Toro 3rd Edition , Springer 2009
        return 0.5 * ((fL + fR) - (consR - consL) * S);
    }

    template<class Tprim>
    inline constexpr auto rusanov_flux_y(Tprim pL, Tprim pR, typename Tprim::Tscal gamma) {
        return x_to_y(rusanov_flux_x(prim_y_to_x(pL), prim_y_to_x(pR), gamma));
    }

    template<class Tprim>
    inline constexpr auto rusanov_flux_z(Tprim pL, Tprim pR, typename Tprim::Tscal gamma) {
        return x_to_z(rusanov_flux_x(prim_z_to_x(pL), prim_z_to_x(pR), gamma));
    }

    template<class Tprim>
    inline constexpr auto rusanov_flux_mx(Tprim pL, Tprim pR, typename Tprim::Tscal gamma) {
        return invert_axis(rusanov_flux_x(prim_invert_axis(pL), prim_invert_axis(pR), gamma));
    }

    template<class Tprim>
    inline constexpr auto rusanov_flux_my(Tprim pL, Tprim pR, typename Tprim::Tscal gamma) {
        return invert_axis(rusanov_flux_y(prim_invert_axis(pL), prim_invert_axis(pR), gamma));
    }

    template<class Tprim>
    inline constexpr auto rusanov_flux_mz(Tprim pL, Tprim pR, typename Tprim::Tscal gamma) {
        return invert_axis(rusanov_flux_z(prim_invert_axis(pL), prim_invert_axis(pR), gamma));
    }

} // namespace shammath
