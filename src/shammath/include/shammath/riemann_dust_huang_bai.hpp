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
 * @file riemann_dust_huang_bai.hpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr) --no git blame--
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Huang & Bai (2022) pressureless dust Riemann solver
 */

#include "shammath/riemann_common.hpp"

namespace shammath {

    // Huang & Bai, 2022 ,A Multifluid Dust Module in Athena++: Algorithms and Numerical Tests
    // Equation (32)
    template<class Tprim>
    inline constexpr auto huang_bai_flux_x(Tprim d_primL, Tprim d_primR) {
        const auto fL = d_hydro_flux_x(d_primL);
        const auto fR = d_hydro_flux_x(d_primR);

        DustConsState<typename Tprim::Tvec> d_flux{};

        if (d_primL.vel[0] > 0 && d_primR.vel[0] > 0)
            d_flux = fL;
        else if (d_primL.vel[0] < 0 && d_primR.vel[0] < 0)
            d_flux = fR;
        else if (d_primL.vel[0] < 0 && d_primR.vel[0] > 0)
            d_flux *= 0;
        else if (d_primL.vel[0] > 0 && d_primR.vel[0] < 0)
            d_flux = (fL + fR);

        return d_flux;
    }

    template<class Tprim>
    inline constexpr auto huang_bai_flux_y(Tprim pL, Tprim pR) {
        return d_x_to_y(huang_bai_flux_x(d_prim_y_to_x(pL), d_prim_y_to_x(pR)));
    }

    template<class Tprim>
    inline constexpr auto huang_bai_flux_z(Tprim pL, Tprim pR) {
        return d_x_to_z(huang_bai_flux_x(d_prim_z_to_x(pL), d_prim_z_to_x(pR)));
    }

    template<class Tprim>
    inline constexpr auto huang_bai_flux_mx(Tprim pL, Tprim pR) {
        return d_invert_axis(huang_bai_flux_x(d_prim_invert_axis(pL), d_prim_invert_axis(pR)));
    }

    template<class Tprim>
    inline constexpr auto huang_bai_flux_my(Tprim pL, Tprim pR) {
        return d_invert_axis(huang_bai_flux_y(d_prim_invert_axis(pL), d_prim_invert_axis(pR)));
    }

    template<class Tprim>
    inline constexpr auto huang_bai_flux_mz(Tprim pL, Tprim pR) {
        return d_invert_axis(huang_bai_flux_z(d_prim_invert_axis(pL), d_prim_invert_axis(pR)));
    }

} // namespace shammath
