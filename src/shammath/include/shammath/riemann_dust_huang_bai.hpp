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
     * @param primL left  primitive state
     * @param primR right primitive state
     * @param n face unit normal
     */
    template<DustFluidStateSpec FSpec>
    inline constexpr typename FSpec::Tcons huang_bai_flux(
        const FSpec &fspec,
        const typename FSpec::Tprim &primL,
        const typename FSpec::Tprim &primR,
        const typename FSpec::Tvec &n) {
        using Tscal = typename FSpec::Tscal;
        using Tcons = typename FSpec::Tcons;

        const Tscal vnL = fspec.vn(primL, n);
        const Tscal vnR = fspec.vn(primR, n);

        const Tcons fL = fspec.flux(primL, n, vnL);
        const Tcons fR = fspec.flux(primR, n, vnR);

        Tcons flux{};

        if (vnL > 0 && vnR > 0)
            flux = fL;
        else if (vnL < 0 && vnR < 0)
            flux = fR;
        else if (vnL < 0 && vnR > 0)
            flux *= 0;
        else if (vnL > 0 && vnR < 0)
            flux = (fL + fR);

        return flux;
    }

} // namespace shammath
