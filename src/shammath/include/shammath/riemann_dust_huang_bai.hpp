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

    /**
     * @brief Huang & Bai dust flux across a face with unit normal n
     *
     * Huang & Bai, 2022, A Multifluid Dust Module in Athena++: Algorithms and Numerical
     * Tests, Equation (32)
     * @tparam FSpec
     * @param fspec dust state spec (flux/vn operations, no equation of state)
     * @param prim_l left  primitive state
     * @param prim_r right primitive state
     * @param n face unit normal
     */
    template<DustFluidStateSpec FSpec>
    inline constexpr typename FSpec::Tcons huang_bai_flux(
        const FSpec &fspec,
        const typename FSpec::Tprim &prim_l,
        const typename FSpec::Tprim &prim_r,
        const typename FSpec::Tvec &n) {
        using Tscal = typename FSpec::Tscal;
        using Tcons = typename FSpec::Tcons;

        const Tscal vn_l = fspec.vn(prim_l, n);
        const Tscal vn_r = fspec.vn(prim_r, n);

        const Tcons f_l = fspec.flux(prim_l, n, vn_l);
        const Tcons f_r = fspec.flux(prim_r, n, vn_r);

        Tcons flux{};

        if (vn_l > 0 && vn_r > 0)
            flux = f_l;
        else if (vn_l < 0 && vn_r < 0)
            flux = f_r;
        else if (vn_l < 0 && vn_r > 0)
            flux *= 0;
        else if (vn_l > 0 && vn_r < 0)
            flux = (f_l + f_r);

        return flux;
    }

} // namespace shammath
