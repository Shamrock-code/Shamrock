// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothee David--Cleris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file SolverModes.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief Solver mode modifiers for GSPH
 *
 * Defines how the solver evolves:
 * - NormalEvolution: Standard time evolution
 * - RelaxationMode: Damped evolution toward equilibrium (Lane-Emden)
 * - DustMode: Pressureless limit
 *
 * Modes can modify the computed derivatives before time integration.
 */

#include "shambackends/sycl.hpp"
#include "shambackends/vec.hpp"

namespace shammodels::gsph::modes {

    // ========================================================================
    // NormalEvolution: Standard time evolution
    // ========================================================================

    /**
     * @brief Normal time evolution mode
     *
     * No modifications to computed derivatives. This is the default mode
     * for production simulations.
     */
    struct NormalEvolution {
        /// No convergence checking
        static constexpr bool has_convergence_check = false;

        /**
         * @brief Modify derivatives (no-op for normal evolution)
         *
         * @tparam Derivs Derivative container type
         * @tparam State State container type
         * @param derivs Derivatives to potentially modify
         * @param state Current state
         */
        template<class Derivs, class State>
        static void modify_derivatives(Derivs & /* derivs */, const State & /* state */) {
            // No-op
        }
    };

    // ========================================================================
    // RelaxationMode: Damped evolution toward equilibrium
    // ========================================================================

    /**
     * @brief Relaxation mode for finding equilibrium configurations
     *
     * Adds velocity damping to accelerate convergence toward equilibrium.
     * Useful for:
     * - Lane-Emden polytropic spheres
     * - Disk equilibria
     * - Stellar structure relaxation
     *
     * @tparam Tvec Vector type
     */
    template<class Tvec>
    struct RelaxationMode {
        using Tscal = shambase::VecComponent<Tvec>;

        /// This mode has convergence checking
        static constexpr bool has_convergence_check = true;

        // =====================================================================
        // Damping parameters
        // =====================================================================

        /// Velocity damping coefficient: dv/dt += -velocity_damping * v
        Tscal velocity_damping = Tscal{0.1};

        /// Optional thermal damping coefficient
        Tscal energy_damping = Tscal{0};

        /// Force isothermal evolution (du/dt = 0)
        bool isothermal = false;

        /// Target temperature for isothermal or thermal damping
        Tscal target_temperature = Tscal{0};

        // =====================================================================
        // Convergence criteria
        // =====================================================================

        /// Maximum velocity threshold for convergence
        Tscal velocity_threshold = Tscal{1e-6};

        /// Virial ratio threshold: |2K/W + 1| < virial_threshold
        Tscal virial_threshold = Tscal{0.01};

        /// Maximum relative density change threshold
        Tscal density_threshold = Tscal{1e-6};

        // =====================================================================
        // Methods
        // =====================================================================

        /**
         * @brief Apply damping to derivatives
         *
         * @tparam Derivs Derivative container type (must have acceleration, energy_rate)
         * @tparam State State container type (must have velocity)
         * @param derivs Derivatives to modify
         * @param state Current state
         */
        template<class Derivs, class State>
        void modify_derivatives(Derivs &derivs, const State &state) const {
            // Velocity damping: dv/dt += -eta * v
            derivs.acceleration -= velocity_damping * state.velocity;

            if (isothermal) {
                // Force constant temperature
                derivs.energy_rate = Tscal{0};
            } else if (energy_damping > Tscal{0}) {
                // Damp toward target temperature
                // Note: This requires temperature access in state
                // For now, just zero energy rate if damping is set
                derivs.energy_rate *= (Tscal{1} - energy_damping);
            }
        }

        /**
         * @brief Check if relaxation has converged
         *
         * @param max_velocity Maximum velocity magnitude across all particles
         * @param virial_ratio Current virial ratio (2K/|W|)
         * @param max_drho_rel Maximum relative density change rate
         * @return true if all criteria are met
         */
        bool is_converged(Tscal max_velocity, Tscal virial_ratio, Tscal max_drho_rel) const {
            return (max_velocity < velocity_threshold)
                   && (sycl::fabs(virial_ratio - Tscal{1}) < virial_threshold)
                   && (max_drho_rel < density_threshold);
        }
    };

    // ========================================================================
    // DustMode: Pressureless limit
    // ========================================================================

    /**
     * @brief Dust (pressureless) mode
     *
     * Forces zero pressure contribution. Useful for:
     * - Cold dark matter simulations
     * - Pressureless collapse problems
     */
    struct DustMode {
        /// No convergence checking
        static constexpr bool has_convergence_check = false;

        /**
         * @brief Modify derivatives for dust limit
         *
         * Sets energy rate to zero (no thermal evolution).
         *
         * @tparam Derivs Derivative container type
         * @tparam State State container type
         * @param derivs Derivatives to modify
         * @param state Current state (unused)
         */
        template<class Derivs, class State>
        static void modify_derivatives(Derivs &derivs, const State & /* state */) {
            // No thermal evolution in dust limit
            derivs.energy_rate = 0;
            // Note: Pressure = 0 is handled in EOS, not here
        }
    };

} // namespace shammodels::gsph::modes
