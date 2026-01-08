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
 * @file RiemannBase.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr) --no git blame--
 * @brief Riemann solver interface for Special Relativistic hydrodynamics
 *
 * Defines the interface and result type for SR Riemann solvers.
 * Unlike Newtonian solvers, SR solvers must handle:
 * - Lorentz factor constraints
 * - Tangent velocity continuity at interfaces
 * - Relativistic wave speeds
 *
 * The SR Riemann problem includes BOTH normal and tangent velocity
 * components, as tangent velocity affects the Lorentz factor.
 *
 * Based on:
 * - Kitajima, Inutsuka, Seno (2025) arXiv:2510.18251v1
 * - Pons et al. (2000) for exact relativistic Riemann solver
 */

#include "shambase/aliases_float.hpp"
#include <string_view>

namespace shammodels::gsph::physics::sr::riemann {

    /**
     * @brief Result of an SR Riemann solver
     *
     * Contains interface values including tangent velocity.
     * The tangent velocity v_t is continuous across contact discontinuities
     * in SR (unlike Newtonian case where it's irrelevant).
     */
    template<class Tscal>
    struct Result {
        Tscal P_star;          ///< Interface pressure
        Tscal v_x_star;        ///< Interface normal velocity
        Tscal v_t_star;        ///< Interface tangent velocity (SR-specific!)
        bool converged = true; ///< Solver convergence flag (for iterative solvers)
    };

    /**
     * @brief Configuration for exact SR Riemann solver
     */
    struct ExactConfig {
        f64 tol      = 1e-10; ///< Newton-Raphson convergence tolerance
        u32 max_iter = 100;   ///< Maximum iterations
    };

    /**
     * @brief Configuration for HLLC-SR Riemann solver
     */
    struct HLLC_SRConfig {
        // HLLC-SR has no tunable parameters
    };

    // ════════════════════════════════════════════════════════════════════════════
    // Solver function declarations
    // ════════════════════════════════════════════════════════════════════════════

    /**
     * @brief Exact SR Riemann solver
     *
     * Solves the exact relativistic Riemann problem using Newton-Raphson
     * iteration. Based on Pons et al. (2000).
     *
     * The SR Riemann problem requires:
     * - Normal velocity v_x (projected onto pair axis)
     * - Tangent velocity magnitude |v_t| (perpendicular to pair axis)
     * - Rest-frame density n (NOT lab-frame N)
     * - Pressure P
     *
     * @param v_x_L Left state normal velocity
     * @param v_t_L Left state tangent velocity magnitude
     * @param n_L Left state rest-frame density
     * @param P_L Left state pressure
     * @param v_x_R Right state normal velocity
     * @param v_t_R Right state tangent velocity magnitude
     * @param n_R Right state rest-frame density
     * @param P_R Right state pressure
     * @param gamma_eos Adiabatic index
     * @param c Speed of light
     * @param config Solver configuration
     * @return Interface state (P*, v_x*, v_t*)
     */
    template<class Tscal>
    Result<Tscal> solve_exact(
        Tscal v_x_L,
        Tscal v_t_L,
        Tscal n_L,
        Tscal P_L,
        Tscal v_x_R,
        Tscal v_t_R,
        Tscal n_R,
        Tscal P_R,
        Tscal gamma_eos,
        Tscal c,
        const ExactConfig &config);

    /**
     * @brief HLLC approximate SR Riemann solver
     *
     * Three-wave approximate solver adapted for special relativity.
     * Faster than exact solver but less accurate for strong shocks.
     *
     * @param v_x_L Left state normal velocity
     * @param v_t_L Left state tangent velocity magnitude
     * @param n_L Left state rest-frame density
     * @param P_L Left state pressure
     * @param v_x_R Right state normal velocity
     * @param v_t_R Right state tangent velocity magnitude
     * @param n_R Right state rest-frame density
     * @param P_R Right state pressure
     * @param gamma_eos Adiabatic index
     * @param c Speed of light
     * @param config Solver configuration
     * @return Interface state (P*, v_x*, v_t*)
     */
    template<class Tscal>
    Result<Tscal> solve_hllc_sr(
        Tscal v_x_L,
        Tscal v_t_L,
        Tscal n_L,
        Tscal P_L,
        Tscal v_x_R,
        Tscal v_t_R,
        Tscal n_R,
        Tscal P_R,
        Tscal gamma_eos,
        Tscal c,
        const HLLC_SRConfig &config);

} // namespace shammodels::gsph::physics::sr::riemann
