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
 * @author Timothée David--Cléris (tim.shamrock@proton.me) --no git blame--
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr) --no git blame--
 * @brief Newtonian GSPH force computation using Riemann solver results
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

#include "riemann/Iterative.hpp"
#include "shambackends/math.hpp"
#include "shambackends/sycl.hpp"
#include "shammodels/sph/math/forces.hpp"

namespace shammodels::gsph::physics::newtonian {

    // Note: For GSPH acceleration, use shamrock::sph::sph_pressure_symetric() with p_star
    // as both P_a and P_b. This provides proper handling of zero denominators via
    // sham::inv_sat_zero() and avoids code duplication.

    /**
     * @brief Compute GSPH energy equation contribution
     *
     * Following Cha & Whitworth (2003), the energy equation uses the same
     * symmetric force as the momentum equation:
     *   f_ab = m_b * p* * (nabla_W_a / (rho_a^2 * Omega_a) + nabla_W_b / (rho_b^2 * Omega_b))
     *   du_a/dt = -f_ab dot (v* - v_a)
     *
     * This ensures proper energy conservation in shocks by using the same
     * force that appears in the momentum equation.
     *
     * @tparam Tvec Vector type
     * @tparam Tscal Scalar type
     * @param m_b Mass of particle b
     * @param p_star Interface pressure from Riemann solver
     * @param v_star Interface velocity (scalar, in direction of r_ab)
     * @param rho_a_sq Density squared of particle a
     * @param rho_b_sq Density squared of particle b
     * @param omega_a Grad-h correction factor for particle a
     * @param omega_b Grad-h correction factor for particle b
     * @param v_a Velocity of particle a
     * @param r_ab_unit Unit vector from a to b
     * @param nabla_W_a Kernel gradient at r_ab with smoothing length h_a
     * @param nabla_W_b Kernel gradient at r_ab with smoothing length h_b
     * @return Energy rate contribution from this pair
     */
    template<class Tvec, class Tscal>
    inline Tscal gsph_energy_rate(
        Tscal m_b,
        Tscal p_star,
        Tscal v_star,
        Tscal rho_a_sq,
        Tscal rho_b_sq,
        Tscal omega_a,
        Tscal omega_b,
        Tvec v_a,
        Tvec r_ab_unit,
        Tvec nabla_W_a,
        Tvec nabla_W_b) {

        // Interface velocity vector (in direction of pair axis)
        Tvec v_star_vec = v_star * r_ab_unit;

        // Compute symmetric force (same as momentum equation)
        // f = m_b * p* * (nabla_W_a / (rho_a^2 * Omega_a) + nabla_W_b / (rho_b^2 * Omega_b))
        Tscal sub_fact_a = rho_a_sq * omega_a;
        Tscal sub_fact_b = rho_b_sq * omega_b;
        Tvec f           = m_b * p_star
                 * (nabla_W_a * sham::inv_sat_zero(sub_fact_a)
                    + nabla_W_b * sham::inv_sat_zero(sub_fact_b));

        // Energy rate: -f dot (v* - v_a)
        return -sycl::dot(f, v_star_vec - v_a);
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

        // Acceleration: use sph_pressure_symetric with p_star as both P_a and P_b
        // This provides proper handling of zero denominators via sham::inv_sat_zero()
        dv_dt += shamrock::sph::sph_pressure_symetric<Tvec, Tscal>(
            m_b, rho_a_sq, rho_b_sq, p_star, p_star, omega_a, omega_b, nabla_W_a, nabla_W_b);

        // Energy rate (uses symmetric force, same as momentum equation)
        du_dt += gsph_energy_rate<Tvec, Tscal>(
            m_b,
            p_star,
            v_star,
            rho_a_sq,
            rho_b_sq,
            omega_a,
            omega_b,
            v_a,
            r_ab_unit,
            nabla_W_a,
            nabla_W_b);
    }

    // Note: For velocity projection onto pair axis, use sycl::dot(v, r_ab_unit) directly.
    // For density from smoothing length, use shamrock::sph::rho_h() from density.hpp.

} // namespace shammodels::gsph::physics::newtonian
