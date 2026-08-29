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
    template<class Tcons>
    inline constexpr auto huang_bai_flux_x(Tcons cL, Tcons cR) {
        Tcons d_flux;
        const auto d_primL = d_cons_to_prim(cL);
        const auto d_primR = d_cons_to_prim(cR);

        const auto fL = d_hydro_flux_x(cL);
        const auto fR = d_hydro_flux_x(cR);

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

    template<class Tcons>
    inline constexpr Tcons huang_bai_flux_y(Tcons cL, Tcons cR) {
        return d_x_to_y(huang_bai_flux_x(d_y_to_x(cL), d_y_to_x(cR)));
    }

    template<class Tcons>
    inline constexpr Tcons huang_bai_flux_z(Tcons cL, Tcons cR) {
        return d_x_to_z(huang_bai_flux_x(d_z_to_x(cL), d_z_to_x(cR)));
    }

    template<class Tcons>
    inline constexpr Tcons huang_bai_flux_mx(Tcons cL, Tcons cR) {
        return d_invert_axis(huang_bai_flux_x(d_invert_axis(cL), d_invert_axis(cR)));
    }

    template<class Tcons>
    inline constexpr Tcons huang_bai_flux_my(Tcons cL, Tcons cR) {
        return d_invert_axis(huang_bai_flux_y(d_invert_axis(cL), d_invert_axis(cR)));
    }

    template<class Tcons>
    inline constexpr Tcons huang_bai_flux_mz(Tcons cL, Tcons cR) {
        return d_invert_axis(huang_bai_flux_z(d_invert_axis(cL), d_invert_axis(cR)));
    }

} // namespace shammath
