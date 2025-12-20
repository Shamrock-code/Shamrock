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
 * @file iterative.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr) --no git blame--
 * @brief Iterative Riemann solver for GSPH (van Leer 1997)
 *
 * Implements the van Leer (1997) iterative Riemann solver for ideal gas.
 * Uses Newton-Raphson iteration to find the exact solution (p*, v*) at
 * particle interfaces.
 *
 * References:
 * - van Leer, B. (1997) "Towards the ultimate conservative difference scheme"
 * - Toro, E.F. (2009) "Riemann Solvers and Numerical Methods for Fluid Dynamics"
 */

#include "shambackends/math.hpp"
#include "shambackends/sycl.hpp"

namespace shammodels::gsph::riemann {

    /**
     * @brief Result of Riemann solver
     *
     * Contains the interface pressure (p*) and velocity (v*) computed
     * by solving the Riemann problem between left and right states.
     */
    template<class Tscal>
    struct RiemannResult {
        Tscal p_star; ///< Interface pressure
        Tscal v_star; ///< Interface velocity (normal component)
    };

    /**
     * @brief Iterative Riemann solver (van Leer 1997)
     *
     * Solves the Riemann problem exactly for an ideal gas using Newton-Raphson
     * iteration. Returns the interface pressure and velocity (p*, v*).
     *
     * The left/right convention is:
     * - Left state (L): particle on the "minus" side of the interface
     * - Right state (R): particle on the "plus" side of the interface
     * - Positive velocity points from L to R
     *
     * @tparam Tscal Scalar type (f32 or f64)
     * @param u_L Left state velocity (normal component)
     * @param rho_L Left state density
     * @param p_L Left state pressure
     * @param u_R Right state velocity (normal component)
     * @param rho_R Right state density
     * @param p_R Right state pressure
     * @param gamma Adiabatic index
     * @param tol Convergence tolerance (default: 1e-6)
     * @param max_iter Maximum iterations (default: 20)
     * @return RiemannResult with p_star and v_star
     */
    template<class Tscal>
    inline RiemannResult<Tscal> iterative_solver(
        Tscal u_L,
        Tscal rho_L,
        Tscal p_L,
        Tscal u_R,
        Tscal rho_R,
        Tscal p_R,
        Tscal gamma,
        Tscal tol    = Tscal{1.0e-6},
        u32 max_iter = 20) {

        RiemannResult<Tscal> result;

        // Safety check for non-physical values
        const Tscal smallp   = Tscal{1.0e-25};
        const Tscal smallrho = Tscal{1.0e-25};

        if (rho_L < smallrho || rho_R < smallrho || p_L < smallp || p_R < smallp) {
            // Return acoustic approximation for near-vacuum
            result.p_star = sycl::fmax(smallp, Tscal{0.5} * (p_L + p_R));
            result.v_star = Tscal{0.5} * (u_L + u_R);
            return result;
        }

        // Derived constants
        const Tscal gm1    = gamma - Tscal{1};
        const Tscal gp1    = gamma + Tscal{1};
        const Tscal gamma1 = Tscal{0.5} * gp1 / gamma; // (gamma+1)/(2*gamma)

        // Specific volumes
        const Tscal V_L = Tscal{1} / rho_L;
        const Tscal V_R = Tscal{1} / rho_R;

        // Lagrangian sound speeds: c_lag = sqrt(gamma * p * rho)
        const Tscal c_L = sycl::sqrt(gamma * p_L * rho_L);
        const Tscal c_R = sycl::sqrt(gamma * p_R * rho_R);

        // Initial guess for p_star using PVRS (Primitive Variable Riemann Solver)
        // p_star = p_L + (p_R - p_L - c_R*(u_R - u_L)) * c_L / (c_L + c_R)
        Tscal p_star = p_R - p_L - c_R * (u_R - u_L);
        p_star       = p_L + p_star * c_L / (c_L + c_R);
        p_star       = sycl::fmax(p_star, smallp);

        // Newton-Raphson iteration
        for (u32 iter = 0; iter < max_iter; ++iter) {
            const Tscal p_star_old = p_star;

            // Left wave impedance: W_L = c_L * sqrt(1 + gamma1*(p_star - p_L)/p_L)
            Tscal W_L = Tscal{1} + gamma1 * (p_star - p_L) / p_L;
            W_L       = c_L * sycl::sqrt(sycl::fmax(W_L, smallp));

            // Right wave impedance: W_R = c_R * sqrt(1 + gamma1*(p_star - p_R)/p_R)
            Tscal W_R = Tscal{1} + gamma1 * (p_star - p_R) / p_R;
            W_R       = c_R * sycl::sqrt(sycl::fmax(W_R, smallp));

            // Derivatives dW/dp for Newton-Raphson
            // Z_L = -dW_L/dp * W_L (note the sign convention)
            Tscal Z_L = Tscal{4} * V_L * W_L * W_L;
            Z_L       = -Z_L * W_L / (Z_L - gp1 * (p_star - p_L));

            // Z_R = dW_R/dp * W_R
            Tscal Z_R = Tscal{4} * V_R * W_R * W_R;
            Z_R       = Z_R * W_R / (Z_R - gp1 * (p_star - p_R));

            // Intermediate velocities from each side
            // u*_L = u_L - (p* - p_L) / W_L
            // u*_R = u_R + (p* - p_R) / W_R
            const Tscal ustar_L = u_L - (p_star - p_L) / W_L;
            const Tscal ustar_R = u_R + (p_star - p_R) / W_R;

            // Newton-Raphson update: p_new = p - f(p)/f'(p)
            // where f(p) = u*_R - u*_L (velocity mismatch)
            // and f'(p) = du*_R/dp - du*_L/dp = 1/Z_R - 1/Z_L
            const Tscal denom = Z_R - Z_L;
            if (sycl::fabs(denom) > smallp) {
                p_star = p_star + (ustar_R - ustar_L) * (Z_L * Z_R) / denom;
            }
            p_star = sycl::fmax(smallp, p_star);

            // Check convergence
            if (sycl::fabs(p_star - p_star_old) / p_star < tol) {
                break;
            }
        }

        // Recalculate wave impedances with final p_star
        Tscal W_L = Tscal{1} + gamma1 * (p_star - p_L) / p_L;
        W_L       = c_L * sycl::sqrt(sycl::fmax(W_L, smallp));

        Tscal W_R = Tscal{1} + gamma1 * (p_star - p_R) / p_R;
        W_R       = c_R * sycl::sqrt(sycl::fmax(W_R, smallp));

        // Calculate final u_star (average of left and right estimates)
        const Tscal ustar_L = u_L - (p_star - p_L) / W_L;
        const Tscal ustar_R = u_R + (p_star - p_R) / W_R;
        const Tscal u_star  = Tscal{0.5} * (ustar_L + ustar_R);

        result.p_star = p_star;
        result.v_star = u_star;

        return result;
    }

    /**
     * @brief HLLC approximate Riemann solver
     *
     * Harten-Lax-van Leer-Contact approximate solver. Faster than iterative
     * but still captures contact discontinuities accurately.
     *
     * @tparam Tscal Scalar type (f32 or f64)
     * @param u_L Left state velocity
     * @param rho_L Left state density
     * @param p_L Left state pressure
     * @param u_R Right state velocity
     * @param rho_R Right state density
     * @param p_R Right state pressure
     * @param gamma Adiabatic index
     * @return RiemannResult with p_star and v_star
     */
    template<class Tscal>
    inline RiemannResult<Tscal> hllc_solver(
        Tscal u_L,
        Tscal rho_L,
        Tscal p_L,
        Tscal u_R,
        Tscal rho_R,
        Tscal p_R,
        Tscal gamma) {

        RiemannResult<Tscal> result;

        // Compute Eulerian sound speeds from gamma
        const Tscal c_L = sycl::sqrt(gamma * p_L / rho_L);
        const Tscal c_R = sycl::sqrt(gamma * p_R / rho_R);

        // Roe averages for wave speed estimates
        const Tscal sqrt_rho_L = sycl::sqrt(rho_L);
        const Tscal sqrt_rho_R = sycl::sqrt(rho_R);
        const Tscal roe_inv    = Tscal{1} / (sqrt_rho_L + sqrt_rho_R);

        const Tscal u_roe = (sqrt_rho_L * u_L + sqrt_rho_R * u_R) * roe_inv;
        const Tscal c_roe = (sqrt_rho_L * c_L + sqrt_rho_R * c_R) * roe_inv;

        // Wave speed estimates
        const Tscal S_L = sycl::fmin(u_L - c_L, u_roe - c_roe);
        const Tscal S_R = sycl::fmax(u_R + c_R, u_roe + c_roe);

        // Contact wave speed (star velocity)
        const Tscal denom = rho_L * (S_L - u_L) - rho_R * (S_R - u_R);
        const Tscal numer = p_R - p_L + rho_L * u_L * (S_L - u_L) - rho_R * u_R * (S_R - u_R);

        const Tscal smallval = Tscal{1.0e-25};
        Tscal S_star;
        if (sycl::fabs(denom) > smallval) {
            S_star = numer / denom;
        } else {
            S_star = Tscal{0.5} * (u_L + u_R);
        }

        // Star pressure
        Tscal p_star = p_L + rho_L * (S_L - u_L) * (S_star - u_L);
        p_star       = sycl::fmax(smallval, p_star);

        result.p_star = p_star;
        result.v_star = S_star;

        return result;
    }

} // namespace shammodels::gsph::riemann
