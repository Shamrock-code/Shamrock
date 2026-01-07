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
 * @brief Riemann solver interface for Newtonian hydrodynamics
 *
 * Defines the interface and result type for Newtonian Riemann solvers.
 * Newtonian solvers work with the simple state (u, ρ, P) and return
 * interface values (p*, v*).
 *
 * Available implementations:
 * - Iterative (van Leer 1997)
 * - HLL (Harten-Lax-van Leer)
 * - HLLC (HLL with Contact)
 * - Roe (linearized)
 */

#include "shambase/aliases_float.hpp"
#include <string_view>

namespace shammodels::gsph::physics::newtonian::riemann {

    /**
     * @brief Result of a Newtonian Riemann solver
     *
     * Contains the interface pressure p* and velocity v* at the
     * contact discontinuity. These are used to compute GSPH forces.
     */
    template<class Tscal>
    struct Result {
        Tscal p_star; ///< Interface pressure
        Tscal v_star; ///< Interface velocity (contact velocity)
    };

    /**
     * @brief Configuration for iterative Riemann solver
     */
    struct IterativeConfig {
        f64 tol      = 1e-8; ///< Convergence tolerance
        u32 max_iter = 100;  ///< Maximum iterations
    };

    /**
     * @brief Configuration for HLL Riemann solver
     */
    struct HLLConfig {
        // HLL has no tunable parameters
    };

    /**
     * @brief Configuration for HLLC Riemann solver
     */
    struct HLLCConfig {
        // HLLC has no tunable parameters
    };

    /**
     * @brief Configuration for Roe Riemann solver
     */
    struct RoeConfig {
        f64 entropy_fix = 0.1; ///< Entropy fix parameter
    };

    // ════════════════════════════════════════════════════════════════════════════
    // Solver function declarations (implementations in separate files)
    // ════════════════════════════════════════════════════════════════════════════

    /**
     * @brief Iterative Riemann solver (van Leer 1997)
     *
     * Uses Newton-Raphson iteration to find the exact interface pressure.
     * Robust and accurate but slower than approximate solvers.
     *
     * @param u_L Left state velocity (projected onto pair axis)
     * @param rho_L Left state density
     * @param P_L Left state pressure
     * @param u_R Right state velocity
     * @param rho_R Right state density
     * @param P_R Right state pressure
     * @param gamma Adiabatic index
     * @param config Solver configuration (tolerance, max iterations)
     * @return Interface state (p*, v*)
     */
    template<class Tscal>
    Result<Tscal> solve_iterative(
        Tscal u_L,
        Tscal rho_L,
        Tscal P_L,
        Tscal u_R,
        Tscal rho_R,
        Tscal P_R,
        Tscal gamma,
        const IterativeConfig &config);

    /**
     * @brief HLL approximate Riemann solver
     *
     * Two-wave approximate solver. Fast but diffusive at contact
     * discontinuities. Good for smooth flows.
     *
     * @param u_L Left state velocity
     * @param rho_L Left state density
     * @param P_L Left state pressure
     * @param u_R Right state velocity
     * @param rho_R Right state density
     * @param P_R Right state pressure
     * @param gamma Adiabatic index
     * @param config Solver configuration
     * @return Interface state (p*, v*)
     */
    template<class Tscal>
    Result<Tscal> solve_hll(
        Tscal u_L,
        Tscal rho_L,
        Tscal P_L,
        Tscal u_R,
        Tscal rho_R,
        Tscal P_R,
        Tscal gamma,
        const HLLConfig &config);

    /**
     * @brief HLLC approximate Riemann solver
     *
     * Three-wave solver that resolves contact discontinuities better
     * than HLL. Good balance of accuracy and speed.
     *
     * @param u_L Left state velocity
     * @param rho_L Left state density
     * @param P_L Left state pressure
     * @param u_R Right state velocity
     * @param rho_R Right state density
     * @param P_R Right state pressure
     * @param gamma Adiabatic index
     * @param config Solver configuration
     * @return Interface state (p*, v*)
     */
    template<class Tscal>
    Result<Tscal> solve_hllc(
        Tscal u_L,
        Tscal rho_L,
        Tscal P_L,
        Tscal u_R,
        Tscal rho_R,
        Tscal P_R,
        Tscal gamma,
        const HLLCConfig &config);

    /**
     * @brief Roe linearized Riemann solver
     *
     * Linearizes the equations about the Roe-averaged state.
     * Very accurate for weak shocks but may fail for strong shocks
     * without entropy fix.
     *
     * @param u_L Left state velocity
     * @param rho_L Left state density
     * @param P_L Left state pressure
     * @param u_R Right state velocity
     * @param rho_R Right state density
     * @param P_R Right state pressure
     * @param gamma Adiabatic index
     * @param config Solver configuration (entropy fix parameter)
     * @return Interface state (p*, v*)
     */
    template<class Tscal>
    Result<Tscal> solve_roe(
        Tscal u_L,
        Tscal rho_L,
        Tscal P_L,
        Tscal u_R,
        Tscal rho_R,
        Tscal P_R,
        Tscal gamma,
        const RoeConfig &config);

} // namespace shammodels::gsph::physics::newtonian::riemann
