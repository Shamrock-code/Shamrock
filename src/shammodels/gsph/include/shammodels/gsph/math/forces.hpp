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
 * @file forces.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
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

#include "shambase/constants.hpp"
#include "shambackends/math.hpp"
#include "shambackends/sycl.hpp"
#include "shammodels/gsph/math/reconstruction.hpp"
#include "shammodels/gsph/math/riemann/iterative.hpp"
#include "shammodels/sph/math/forces.hpp"

namespace shammodels::gsph {

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

    /**
     * @brief Add Inutsuka (2002) GSPH force contribution from a single neighbor pair
     *
     * Uses the effective volume/face formulation instead of the Cha & Whitworth
     * symmetric SPH form: acc -= m_b * p* * V2_ij * grad_W_ij, where V2_ij is the
     * effective squared volume element between the pair (see math/reconstruction.hpp)
     * and grad_W_ij is the pair-symmetrized kernel gradient.
     *
     * @tparam Tvec Vector type
     * @tparam Tscal Scalar type
     * @param m_b Mass of neighbor particle
     * @param p_star Interface pressure from Riemann solver
     * @param v_star Interface velocity from Riemann solver
     * @param V2_ij Effective squared volume element for the pair
     * @param grad_W_ij Pair-symmetrized kernel gradient vector
     * @param r_ab_unit Unit vector from a to b (points toward b)
     * @param v_a Velocity of particle a
     * @param[out] dv_dt Accumulated acceleration
     * @param[out] du_dt Accumulated energy rate
     */
    template<class Tvec, class Tscal>
    inline void add_gsph_force_contribution_inutsuka(
        Tscal m_b,
        Tscal p_star,
        Tscal v_star,
        Tscal V2_ij,
        Tvec grad_W_ij,
        Tvec r_ab_unit,
        Tvec v_a,
        Tvec &dv_dt,
        Tscal &du_dt) {

        dv_dt -= m_b * p_star * V2_ij * grad_W_ij;

        Tvec v_star_vec = v_star * r_ab_unit;
        du_dt -= m_b * p_star * V2_ij * sycl::dot(grad_W_ij, v_star_vec - v_a);
    }

    /**
     * @brief Dispatch a single neighbor pair's force contribution to ChaWhitworth or
     * InutsukaV2, given the pair's Riemann solver result
     *
     * Shared by update_derivs_iterative()/update_derivs_exact() in UpdateDerivs.cpp,
     * which only differ in which Riemann solver produced (p_star, v_star).
     *
     * @tparam Kernel SPH kernel type (e.g. SPHKernel<Tscal>), for Kernel::dW_3d
     * @tparam Tvec Vector type
     * @tparam Tscal Scalar type
     * @param use_inutsuka_v2 Selects InutsukaV2 (true) or ChaWhitworth (false)
     * @param pmass Particle mass (equal-mass GSPH)
     * @param p_star Interface pressure from Riemann solver
     * @param v_star Interface velocity from Riemann solver
     * @param rho_a Density of particle a
     * @param rho_b Density of particle b
     * @param omega_a Grad-h correction factor for particle a
     * @param omega_b Grad-h correction factor for particle b
     * @param rab Pair separation |r_a - r_b|
     * @param rab_inv Inverse of rab
     * @param h_a Smoothing length of particle a
     * @param h_b Smoothing length of particle b
     * @param r_ab_unit Unit vector from a to b
     * @param vxyz_a Velocity of particle a
     * @param[out] sum_axyz Accumulated acceleration
     * @param[out] sum_du_a Accumulated energy rate
     */
    template<class Kernel, class Tvec, class Tscal>
    inline void accumulate_gsph_pair_force(
        bool use_inutsuka_v2,
        Tscal pmass,
        Tscal p_star,
        Tscal v_star,
        Tscal rho_a,
        Tscal rho_b,
        Tscal omega_a,
        Tscal omega_b,
        Tscal rab,
        Tscal rab_inv,
        Tscal h_a,
        Tscal h_b,
        Tvec r_ab_unit,
        Tvec vxyz_a,
        Tvec &sum_axyz,
        Tscal &sum_du_a) {

        if (use_inutsuka_v2) {
            // Effective volume/face interpolation (Inutsuka 2002), linear
            // (1st order): specific volume is 1/rho for equal-mass particles.
            const Tscal vol_a = Tscal{1} / rho_a;
            const Tscal vol_b = Tscal{1} / rho_b;

            auto face = lin_v2_sast_ij<Tscal>(vol_a, vol_b, h_a, h_b, rab_inv);

            // Pair-symmetrized kernel gradient at sqrt(2)*h (Inutsuka 2002)
            constexpr Tscal sqrt2 = shambase::constants::sqrt_2<Tscal>;
            const Tscal Fab2_a    = Kernel::dW_3d(rab, sqrt2 * h_a);
            const Tscal Fab2_b    = Kernel::dW_3d(rab, sqrt2 * h_b);
            const Tvec grad_W_ij  = (Fab2_a + Fab2_b) * r_ab_unit;

            add_gsph_force_contribution_inutsuka<Tvec, Tscal>(
                pmass, p_star, v_star, face.V2, grad_W_ij, r_ab_unit, vxyz_a, sum_axyz, sum_du_a);
        } else {
            const Tscal Fab_a = Kernel::dW_3d(rab, h_a);
            const Tscal Fab_b = Kernel::dW_3d(rab, h_b);

            add_gsph_force_contribution<Tvec, Tscal>(
                pmass,
                p_star,
                v_star,
                rho_a,
                rho_b,
                omega_a,
                omega_b,
                Fab_a,
                Fab_b,
                r_ab_unit,
                vxyz_a,
                sum_axyz,
                sum_du_a);
        }
    }

} // namespace shammodels::gsph
