// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file forces.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr) --no git blame--
 * @brief GSPH force computation using Riemann solver results
 *
 * Implements the Godunov SPH (GSPH) force formulation following Cha & Whitworth (2003).
 * The key difference from standard SPH is that the interface pressure p* comes from
 * solving the Riemann problem, rather than using artificial viscosity.
 *
 * References:
 * - Cha, S.-H. & Whitworth, A.P. (2003) "Implementations and tests of Godunov-type
 *   particle hydrodynamics"
 * - Inutsuka, S. (2002) "Reformulation of Smoothed Particle Hydrodynamics with
 *   Riemann Solver"
 */

#include "shambackends/math.hpp"
#include "shambackends/sycl.hpp"
#include "shammodels/gsph/math/riemann/iterative.hpp"

namespace shammodels::gsph {

    /**
     * @brief Compute GSPH acceleration from Riemann solver result
     *
     * Following Cha & Whitworth (2003) Eq. 10:
     *   dv_a/dt = -sum_b m_b * p*_ab * (nabla W_ab(h_a) / (rho_a^2 * Omega_a)
     *                                 + nabla W_ab(h_b) / (rho_b^2 * Omega_b))
     *
     * This is the GSPH analog of the SPH pressure gradient, but using the
     * interface pressure p* from the Riemann solver instead of (P_a + P_b)/2
     * or P_a/rho_a^2 + P_b/rho_b^2.
     *
     * @tparam Tvec Vector type
     * @tparam Tscal Scalar type
     * @param m_b Mass of particle b
     * @param p_star Interface pressure from Riemann solver
     * @param rho_a_sq Density squared of particle a
     * @param rho_b_sq Density squared of particle b
     * @param omega_a Grad-h correction factor for particle a
     * @param omega_b Grad-h correction factor for particle b
     * @param nabla_W_a Kernel gradient at r_ab with smoothing length h_a
     * @param nabla_W_b Kernel gradient at r_ab with smoothing length h_b
     * @return Acceleration contribution from this pair
     */
    template<class Tvec, class Tscal>
    inline Tvec gsph_acceleration(
        Tscal m_b,
        Tscal p_star,
        Tscal rho_a_sq,
        Tscal rho_b_sq,
        Tscal omega_a,
        Tscal omega_b,
        Tvec nabla_W_a,
        Tvec nabla_W_b) {

        const Tscal factor_a = p_star / (rho_a_sq * omega_a);
        const Tscal factor_b = p_star / (rho_b_sq * omega_b);

        return -m_b * (factor_a * nabla_W_a + factor_b * nabla_W_b);
    }

    /**
     * @brief Compute GSPH energy equation contribution
     *
     * Following Cha & Whitworth (2003) Eq. 11:
     *   du_a/dt = sum_b m_b * p*_ab * (v*_ab - v_a) dot nabla W_ab(h_a) / (rho_a^2 * Omega_a)
     *
     * where v*_ab is the interface velocity from the Riemann solver.
     * This ensures proper energy conservation in shocks.
     *
     * @tparam Tvec Vector type
     * @tparam Tscal Scalar type
     * @param m_b Mass of particle b
     * @param p_star Interface pressure from Riemann solver
     * @param v_star Interface velocity (scalar, in direction of r_ab)
     * @param rho_a_sq Density squared of particle a
     * @param omega_a Grad-h correction factor for particle a
     * @param v_a Velocity of particle a
     * @param r_ab_unit Unit vector from a to b
     * @param nabla_W_a Kernel gradient at r_ab with smoothing length h_a
     * @return Energy rate contribution from this pair
     */
    template<class Tvec, class Tscal>
    inline Tscal gsph_energy_rate(
        Tscal m_b,
        Tscal p_star,
        Tscal v_star,
        Tscal rho_a_sq,
        Tscal omega_a,
        Tvec v_a,
        Tvec r_ab_unit,
        Tvec nabla_W_a) {

        // Interface velocity vector (in direction of pair axis)
        Tvec v_star_vec = v_star * r_ab_unit;

        // Energy flux: p* * (v* - v_a) dot nabla W
        const Tscal factor = p_star / (rho_a_sq * omega_a);
        return m_b * factor * sycl::dot(v_star_vec - v_a, nabla_W_a);
    }

    /**
     * @brief Add GSPH force contribution from a single neighbor pair
     *
     * Convenience function that computes both acceleration and energy rate
     * contributions from a single particle pair, given the Riemann solver result.
     *
     * @tparam Tvec Vector type
     * @tparam Tscal Scalar type
     * @param m_b Mass of neighbor particle
     * @param p_star Interface pressure from Riemann solver
     * @param v_star Interface velocity from Riemann solver
     * @param rho_a Density of particle a
     * @param rho_b Density of particle b
     * @param omega_a Grad-h correction factor for particle a
     * @param omega_b Grad-h correction factor for particle b
     * @param Fab_a Kernel gradient magnitude |nabla W_ab(h_a)|
     * @param Fab_b Kernel gradient magnitude |nabla W_ab(h_b)|
     * @param r_ab_unit Unit vector from a to b (points toward b)
     * @param v_a Velocity of particle a
     * @param[out] dv_dt Accumulated acceleration
     * @param[out] du_dt Accumulated energy rate
     */
    template<class Tvec, class Tscal>
    inline void add_gsph_force_contribution(
        Tscal m_b,
        Tscal p_star,
        Tscal v_star,
        Tscal rho_a,
        Tscal rho_b,
        Tscal omega_a,
        Tscal omega_b,
        Tscal Fab_a,
        Tscal Fab_b,
        Tvec r_ab_unit,
        Tvec v_a,
        Tvec &dv_dt,
        Tscal &du_dt) {

        const Tscal rho_a_sq = rho_a * rho_a;
        const Tscal rho_b_sq = rho_b * rho_b;

        // Kernel gradient vectors (pointing from a to b)
        Tvec nabla_W_a = Fab_a * r_ab_unit;
        Tvec nabla_W_b = Fab_b * r_ab_unit;

        // Acceleration
        dv_dt += gsph_acceleration<Tvec, Tscal>(
            m_b, p_star, rho_a_sq, rho_b_sq, omega_a, omega_b, nabla_W_a, nabla_W_b);

        // Energy rate
        du_dt += gsph_energy_rate<Tvec, Tscal>(
            m_b, p_star, v_star, rho_a_sq, omega_a, v_a, r_ab_unit, nabla_W_a);
    }

    /**
     * @brief Project velocity onto pair axis for 1D Riemann problem
     *
     * The Riemann problem at each particle interface is solved in 1D along
     * the line connecting the two particles. This function projects the
     * 3D velocity onto this axis.
     *
     * @tparam Tvec Vector type
     * @tparam Tscal Scalar type
     * @param v Velocity vector
     * @param r_ab_unit Unit vector along pair axis (from a to b)
     * @return Scalar velocity component along pair axis
     */
    template<class Tvec, class Tscal>
    inline Tscal project_velocity(Tvec v, Tvec r_ab_unit) {
        return sycl::dot(v, r_ab_unit);
    }

    /**
     * @brief Compute density from smoothing length using mass-density relation
     *
     * For SPH with adaptive smoothing length: rho = m * (h_fact/h)^dim
     *
     * @tparam dim Spatial dimension (1, 2, or 3)
     * @tparam Tscal Scalar type
     * @param m Particle mass
     * @param h Smoothing length
     * @param hfactd h_fact^dim constant from kernel
     * @return Density
     */
    template<u32 dim, class Tscal>
    inline Tscal rho_from_h(Tscal m, Tscal h, Tscal hfactd) {
        Tscal h_inv = Tscal{1} / h;
        return m * hfactd * sycl::pown(h_inv, static_cast<int>(dim));
    }

} // namespace shammodels::gsph
