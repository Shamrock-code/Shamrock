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
        const typename FSpec::Tprim &prim_l,
        const typename FSpec::Tprim &prim_r,
        const typename FSpec::Tvec &n) {
        const auto cs_l = fspec.sound_speed(prim_l);
        const auto cs_r = fspec.sound_speed(prim_r);

        const auto vn_l = fspec.vn(prim_l, n);
        const auto vn_r = fspec.vn(prim_r, n);

        // Equation (10.56) from Toro 3rd Edition , Springer 2009
        const auto s = sham::max((sham::abs(vn_l) + cs_l), (sham::abs(vn_r) + cs_r));

        const auto f_l = fspec.flux(prim_l, n, vn_l);
        const auto f_r = fspec.flux(prim_r, n, vn_r);

        const auto cons_l = fspec.prim_to_cons(prim_l);
        const auto cons_r = fspec.prim_to_cons(prim_r);

        // Equation (10.55) from Toro 3rd Edition , Springer 2009
        return 0.5 * ((f_l + f_r) - (cons_r - cons_l) * s);
    }

} // namespace shammath
