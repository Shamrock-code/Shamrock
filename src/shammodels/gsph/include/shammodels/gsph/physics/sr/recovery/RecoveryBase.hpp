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
 * @file RecoveryBase.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr) --no git blame--
 * @brief Primitive recovery interface for Special Relativistic hydrodynamics
 *
 * In conservative formulations of SR hydrodynamics, we evolve conserved
 * variables (S = γHv, e = γH - P/(Nc²)) but need primitive variables
 * (v, P, n) for the Riemann solver and output.
 *
 * The recovery process involves solving a nonlinear equation (typically
 * a quartic in the Lorentz factor γ) to recover primitives from conserved
 * quantities.
 *
 * Available implementations:
 * - NewtonRaphson: Standard quartic solve for γ (current implementation)
 * - Noble2D: 2D scheme from Noble et al. (2006) - more robust for extreme cases
 */

#include "shambase/aliases_float.hpp"
#include <string_view>

namespace shammodels::gsph::physics::sr::recovery {

    /**
     * @brief Result of primitive recovery
     *
     * Contains the recovered primitive variables and convergence information.
     */
    template<class Tscal>
    struct Result {
        Tscal gamma;    ///< Lorentz factor γ = 1/√(1 - v²/c²)
        Tscal n;        ///< Rest-frame number density
        Tscal P;        ///< Pressure
        Tscal v_mag;    ///< Velocity magnitude |v|
        bool converged; ///< True if iteration converged
        u32 iterations; ///< Number of iterations used
    };

    /**
     * @brief Configuration for Newton-Raphson recovery
     */
    struct NewtonRaphsonConfig {
        f64 tol      = 1e-12; ///< Convergence tolerance
        u32 max_iter = 50;    ///< Maximum iterations
    };

    /**
     * @brief Configuration for Noble 2D recovery scheme
     */
    struct Noble2DConfig {
        f64 tol      = 1e-10; ///< Convergence tolerance
        u32 max_iter = 100;   ///< Maximum iterations
    };

    // ════════════════════════════════════════════════════════════════════════════
    // Recovery function declarations
    // ════════════════════════════════════════════════════════════════════════════

    /**
     * @brief Newton-Raphson primitive recovery
     *
     * Solves the quartic equation for the Lorentz factor γ using Newton-Raphson
     * iteration. Standard method for SR hydrodynamics.
     *
     * The conserved variables are:
     * - S = γ H v (momentum per baryon)
     * - e = γ H - P/(N c²) (energy per baryon)
     *
     * where H = 1 + ε + P/(ρc²) is the specific enthalpy.
     *
     * @param S_mag Magnitude of momentum |S|
     * @param e Energy density
     * @param N Lab-frame number density (from SPH summation)
     * @param gamma_eos Adiabatic index (for ideal gas EOS)
     * @param c Speed of light
     * @param config Solver configuration
     * @return Recovered primitive variables and convergence info
     */
    template<class Tscal>
    Result<Tscal> recover_newton_raphson(
        Tscal S_mag, Tscal e, Tscal N, Tscal gamma_eos, Tscal c, const NewtonRaphsonConfig &config);

    /**
     * @brief Noble 2D primitive recovery scheme
     *
     * Two-dimensional scheme from Noble et al. (2006) that solves for
     * (W², v²) where W = γ. More robust than direct quartic solve for
     * extreme cases (very high Lorentz factors, low densities).
     *
     * Reference:
     * Noble, S.C., Gammie, C.F., McKinney, J.C., Del Zanna, L. (2006)
     * "Primitive Variable Solvers for Conservative General Relativistic
     * Magnetohydrodynamics"
     *
     * @param S_mag Magnitude of momentum |S|
     * @param e Energy density
     * @param N Lab-frame number density
     * @param gamma_eos Adiabatic index
     * @param c Speed of light
     * @param config Solver configuration
     * @return Recovered primitive variables and convergence info
     */
    template<class Tscal>
    Result<Tscal> recover_noble_2d(
        Tscal S_mag, Tscal e, Tscal N, Tscal gamma_eos, Tscal c, const Noble2DConfig &config);

} // namespace shammodels::gsph::physics::sr::recovery
