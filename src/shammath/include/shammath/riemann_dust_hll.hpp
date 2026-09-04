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

    // Krapp et al. 2024, A Fast second-order solver for stiff multifluid dust and gas hydrodynamics
    // Appendice E
    template<class Tprim>
    inline constexpr auto d_hll_flux_x(Tprim d_primL, Tprim d_primR) {
        const auto S = sham::max(sham::abs(d_primL.vel[0]), sham::abs(d_primR.vel[0]));

        const auto fL = d_hydro_flux_x(d_primL);
        const auto fR = d_hydro_flux_x(d_primR);

        const auto cL = d_prim_to_cons(d_primL);
        const auto cR = d_prim_to_cons(d_primR);

        return 0.5 * ((fL + fR) - S * (cR - cL));
    }

    template<class Tprim>
    inline constexpr auto d_hll_flux_y(Tprim pL, Tprim pR) {
        return d_x_to_y(d_hll_flux_x(d_prim_y_to_x(pL), d_prim_y_to_x(pR)));
    }

    template<class Tprim>
    inline constexpr auto d_hll_flux_z(Tprim pL, Tprim pR) {
        return d_x_to_z(d_hll_flux_x(d_prim_z_to_x(pL), d_prim_z_to_x(pR)));
    }

    template<class Tprim>
    inline constexpr auto d_hll_flux_mx(Tprim pL, Tprim pR) {
        return d_invert_axis(d_hll_flux_x(d_prim_invert_axis(pL), d_prim_invert_axis(pR)));
    }

    template<class Tprim>
    inline constexpr auto d_hll_flux_my(Tprim pL, Tprim pR) {
        return d_invert_axis(d_hll_flux_y(d_prim_invert_axis(pL), d_prim_invert_axis(pR)));
    }

    template<class Tprim>
    inline constexpr auto d_hll_flux_mz(Tprim pL, Tprim pR) {
        return d_invert_axis(d_hll_flux_z(d_prim_invert_axis(pL), d_prim_invert_axis(pR)));
    }

} // namespace shammath
