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
     * @param prim_l left  primitive state
     * @param prim_r right primitive state
     * @param n face unit normal
     */
    template<DustFluidStateSpec FSpec>
    inline constexpr typename FSpec::Tcons d_hll_flux(
        const FSpec &fspec,
        const typename FSpec::Tprim &prim_l,
        const typename FSpec::Tprim &prim_r,
        const typename FSpec::Tvec &n) {
        using Tscal = typename FSpec::Tscal;
        using Tcons = typename FSpec::Tcons;

        const Tscal vn_l = fspec.vn(prim_l, n);
        const Tscal vn_r = fspec.vn(prim_r, n);
        const Tscal s    = sham::max(sham::abs(vn_l), sham::abs(vn_r));

        const Tcons f_l = fspec.flux(prim_l, n, vn_l);
        const Tcons f_r = fspec.flux(prim_r, n, vn_r);

        const Tcons c_l = fspec.prim_to_cons(prim_l);
        const Tcons c_r = fspec.prim_to_cons(prim_r);

        return 0.5 * ((f_l + f_r) - s * (c_r - c_l));
    }

} // namespace shammath
