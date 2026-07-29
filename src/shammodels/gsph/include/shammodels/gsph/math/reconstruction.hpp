// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shambackends/math.hpp"

/**
 * @file reconstruction.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief Effective face (volume element) interpolation for the Inutsuka (2002) GSPH formulation
 *
 * Inutsuka's GSPH momentum equation replaces the standard SPH kernel-gradient/rho^2
 * weighting by an effective face between each particle pair, described by an
 * effective squared volume element V2_ij and an effective face location s*.
 * These quantities are obtained by interpolating each particle's specific volume
 * (1/rho for equal-mass particles) along the line joining the pair.
 *
 * This file implements the linear (1st order) interpolation, which only needs the
 * volume at each particle (no gradients).
 *
 * Reference:
 * - Inutsuka, S. (2002) "Reformulation of Smoothed Particle Hydrodynamics with
 *   Riemann Solver"
 */

namespace shammodels::gsph {

    /**
     * @brief Result of the effective face interpolation
     *
     * @tparam Tscal Scalar type
     */
    template<class Tscal>
    struct EffectiveFace {
        Tscal V2;    ///< Effective squared volume element for the pair
        Tscal s_ast; ///< Effective face location (signed distance from particle a)
    };

    /**
     * @brief Linear (1st order) interpolation of the effective face between a pair
     *
     * Reconstructs the volume function V(s) linearly between the two particles using
     * only their volume elements (no gradients needed), following Inutsuka (2002).
     *
     * @tparam Tscal Scalar type
     * @param vol_a Specific volume of particle a (1/rho_a for equal-mass particles)
     * @param vol_b Specific volume of particle b (1/rho_b for equal-mass particles)
     * @param h_a Smoothing length of particle a
     * @param h_b Smoothing length of particle b
     * @param rab_inv Inverse of the pair separation (1/|r_a - r_b|)
     * @return EffectiveFace with V2 and s_ast
     */
    template<class Tscal>
    inline EffectiveFace<Tscal> lin_v2_sast_ij(
        Tscal vol_a, Tscal vol_b, Tscal h_a, Tscal h_b, Tscal rab_inv) {

        const Tscal C  = (vol_a - vol_b) * rab_inv;
        const Tscal D  = Tscal{0.5} * (vol_a + vol_b);
        const Tscal h2 = Tscal{0.5} * (h_a * h_a + h_b * h_b);

        const Tscal V2    = Tscal{0.25} * h2 * C * C + D * D;
        const Tscal s_ast = Tscal{0.5} * h2 * C * D * sham::inv_sat_zero(V2);

        return EffectiveFace<Tscal>{V2, s_ast};
    }

} // namespace shammodels::gsph
