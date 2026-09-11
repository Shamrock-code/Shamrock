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
 * @file riemann_dust_hll.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr) --no git blame--
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Dust HLL Riemann solver
 */

#include "shammath/riemann_common.hpp"

namespace shammath {

    /**
     * @brief Dust HLL flux across a face with unit normal n
     *
     * Krapp et al. 2024, A Fast second-order solver for stiff multifluid dust and gas
     * hydrodynamics, Appendice E
     */
    template<class Tprim>
    inline constexpr auto d_hll_flux(Tprim d_primL, Tprim d_primR, typename Tprim::Tvec n) {
        const auto vnL = n[0] * d_primL.vel[0] + n[1] * d_primL.vel[1] + n[2] * d_primL.vel[2];
        const auto vnR = n[0] * d_primR.vel[0] + n[1] * d_primR.vel[1] + n[2] * d_primR.vel[2];
        const auto S   = sham::max(sham::abs(vnL), sham::abs(vnR));

        const auto fL = d_hydro_flux_n(d_primL, n, vnL);
        const auto fR = d_hydro_flux_n(d_primR, n, vnR);

        const auto cL = d_prim_to_cons(d_primL);
        const auto cR = d_prim_to_cons(d_primR);

        return 0.5 * ((fL + fR) - S * (cR - cL));
    }

} // namespace shammath
