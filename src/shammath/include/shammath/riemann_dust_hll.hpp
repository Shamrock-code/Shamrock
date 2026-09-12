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
     * @tparam FSpec
     * @param fspec dust state spec (flux/vn operations, no equation of state)
     * @param primL left  primitive state
     * @param primR right primitive state
     * @param n face unit normal
     */
    template<DustFluidStateSpec FSpec>
    inline constexpr typename FSpec::Tcons d_hll_flux(
        const FSpec &fspec,
        const typename FSpec::Tprim &primL,
        const typename FSpec::Tprim &primR,
        const typename FSpec::Tvec &n) {
        using Tscal = typename FSpec::Tscal;
        using Tcons = typename FSpec::Tcons;

        const Tscal vnL = fspec.vn(primL, n);
        const Tscal vnR = fspec.vn(primR, n);
        const Tscal S   = sham::max(sham::abs(vnL), sham::abs(vnR));

        const Tcons fL = fspec.flux(primL, n, vnL);
        const Tcons fR = fspec.flux(primR, n, vnR);

        const Tcons cL = fspec.prim_to_cons(primL);
        const Tcons cR = fspec.prim_to_cons(primR);

        return 0.5 * ((fL + fR) - S * (cR - cL));
    }

} // namespace shammath
