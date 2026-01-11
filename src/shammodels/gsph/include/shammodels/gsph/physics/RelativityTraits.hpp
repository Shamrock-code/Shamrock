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
 * @file RelativityTraits.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief Relativity level traits for GSPH physics composition
 *
 * Defines the relativity treatment:
 * - NewtonianPhysics: Classical mechanics, trivial primitive recovery
 * - SRPhysics: Special Relativity, 1D/2D root finding for primitives
 * - GRPhysics: General Relativity, metric-dependent recovery
 *
 * Each relativity level provides:
 * - Conserved variable structure
 * - Primitive recovery algorithm
 * - Signal speed calculation
 */

#include "shambackends/sycl.hpp"
#include "shambackends/vec.hpp"
#include "shammodels/gsph/physics/SpacetimeTraits.hpp"

namespace shammodels::gsph::physics {

    // ========================================================================
    // NewtonianPhysics: Classical mechanics
    // ========================================================================

    /**
     * @brief Newtonian (classical) physics
     *
     * In Newtonian physics:
     * - Conserved = Primitive (trivial inversion)
     * - No Lorentz factor
     * - Signal speed = |v| + c_s
     *
     * @tparam Tvec Vector type
     * @tparam MatterT Matter model (HydroMatter, MHDMatter, etc.)
     * @tparam SpacetimeT Spacetime (must be MinkowskiSpacetime for Newtonian)
     */
    template<class Tvec, class MatterT, class SpacetimeT = MinkowskiSpacetime<Tvec>>
    struct NewtonianPhysics {
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        using Matter    = MatterT;
        using Spacetime = SpacetimeT;

        /// Feature flags
        static constexpr bool needs_primitive_recovery = false;
        static constexpr bool needs_lorentz_factor     = false;
        static constexpr bool needs_metric             = false;

        /// Conserved variables (same as primitive for Newtonian hydro)
        struct ConservedVars {
            Tscal rho;     ///< Density
            Tvec momentum; ///< Momentum density rho * v
            Tscal E;       ///< Total energy density (kinetic + internal)
        };

        /**
         * @brief Recover primitive variables from conserved (trivial for Newtonian)
         *
         * @param U Conserved variables
         * @param W_old Previous primitives (unused, for interface compatibility)
         * @param gamma Adiabatic index (for pressure calculation)
         * @return Primitive variables
         */
        template<class EOSType>
        SYCL_EXTERNAL static auto recover_primitives(
            const ConservedVars &U,
            const typename Matter::PrimitiveVars & /* W_old */,
            const EOSType &eos) -> typename Matter::PrimitiveVars {

            typename Matter::PrimitiveVars W;
            W.rho      = U.rho;
            W.velocity = U.momentum / U.rho;

            Tscal v2 = sycl::dot(W.velocity, W.velocity);
            Tscal ke = Tscal{0.5} * v2;
            W.uint   = U.E / U.rho - ke;

            W.pressure = eos.pressure_from_rho_u(W.rho, W.uint);

            return W;
        }

        /**
         * @brief Maximum signal speed for CFL condition
         *
         * @param W Primitive variables
         * @param cs Sound speed
         * @return Maximum wave speed |v| + c_s
         */
        SYCL_EXTERNAL static Tscal max_signal_speed(
            const typename Matter::PrimitiveVars &W, Tscal cs) {
            return sycl::length(W.velocity) + cs;
        }

        /**
         * @brief Convert primitives to conserved
         *
         * @param W Primitive variables
         * @return Conserved variables
         */
        SYCL_EXTERNAL static ConservedVars to_conserved(const typename Matter::PrimitiveVars &W) {
            ConservedVars U;
            U.rho      = W.rho;
            U.momentum = W.rho * W.velocity;

            Tscal v2 = sycl::dot(W.velocity, W.velocity);
            U.E      = W.rho * (W.uint + Tscal{0.5} * v2);

            return U;
        }
    };

    // ========================================================================
    // SRPhysics: Special Relativity (STUB)
    // ========================================================================

    /**
     * @brief Special Relativistic physics (STUB - not yet implemented)
     *
     * In SR:
     * - Conserved: D = rho W, S = rho h W^2 v, tau = rho h W^2 - P - D
     * - Primitive recovery: 1D root find for hydro, 2D for MHD
     * - Signal speed: relativistic addition
     *
     * @tparam Tvec Vector type
     * @tparam MatterT Matter model
     */
    template<class Tvec, class MatterT>
    struct SRPhysics {
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        using Matter    = MatterT;
        using Spacetime = MinkowskiSpacetime<Tvec>; // Always flat for SR

        static constexpr bool needs_primitive_recovery = true;
        static constexpr bool needs_lorentz_factor     = true;
        static constexpr bool needs_metric             = false;

        /// Conserved variables (relativistic)
        struct ConservedVars {
            Tscal D;   ///< D = rho * W (relativistic mass density)
            Tvec S;    ///< S^i = rho * h * W^2 * v^i (momentum density)
            Tscal tau; ///< tau = rho * h * W^2 - P - D (energy - rest mass)
        };

        /// Lorentz factor from velocity
        SYCL_EXTERNAL static Tscal lorentz_factor(Tvec v) {
            Tscal v2 = sycl::dot(v, v);
            return Tscal{1} / sycl::sqrt(Tscal{1} - v2); // c = 1
        }

        /**
         * @brief Recover primitive variables (STUB)
         *
         * Uses Newton-Raphson iteration to solve for pressure.
         * For hydro: 1D solve
         * For MHD: 2D solve for (P, W) or similar
         */
        template<class EOSType>
        SYCL_EXTERNAL static auto recover_primitives(
            const ConservedVars &U,
            const typename Matter::PrimitiveVars &W_old,
            const EOSType &eos,
            u32 max_iter = 50,
            Tscal tol    = Tscal{1e-10}) -> typename Matter::PrimitiveVars {
            // STUB: Not implemented
            (void) U;
            (void) eos;
            (void) max_iter;
            (void) tol;
            return W_old;
        }

        /// Maximum signal speed (relativistic addition)
        SYCL_EXTERNAL static Tscal max_signal_speed(
            const typename Matter::PrimitiveVars &W, Tscal cs) {
            Tscal v = sycl::length(W.velocity);
            // Relativistic velocity addition: (v + cs) / (1 + v*cs)
            return (v + cs) / (Tscal{1} + v * cs);
        }
    };

    // ========================================================================
    // GRPhysics: General Relativity (STUB)
    // ========================================================================

    /**
     * @brief General Relativistic physics (STUB - not yet implemented)
     *
     * In GR with fixed spacetime:
     * - Conserved variables include metric factors (sqrt(gamma))
     * - Primitive recovery requires metric at each point
     * - Geometric source terms from Christoffel symbols
     *
     * @tparam Tvec Vector type
     * @tparam MatterT Matter model
     * @tparam SpacetimeT Spacetime model (must be curved)
     */
    template<class Tvec, class MatterT, class SpacetimeT>
    struct GRPhysics {
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        using Matter    = MatterT;
        using Spacetime = SpacetimeT;

        static constexpr bool needs_primitive_recovery = true;
        static constexpr bool needs_lorentz_factor     = true;
        static constexpr bool needs_metric             = true;

        /// Conserved variables (GR form)
        struct ConservedVars {
            Tscal D;   ///< sqrt(gamma) * rho * W
            Tvec S;    ///< sqrt(gamma) * T^0_i
            Tscal tau; ///< sqrt(gamma) * (T^00 - D)
        };

        /**
         * @brief Recover primitives with metric (STUB)
         *
         * @param U Conserved variables
         * @param W_old Previous primitives (initial guess)
         * @param eos Equation of state
         * @param spacetime Spacetime geometry
         * @param position Particle position (for metric evaluation)
         */
        template<class EOSType>
        SYCL_EXTERNAL static auto recover_primitives(
            const ConservedVars &U,
            const typename Matter::PrimitiveVars &W_old,
            const EOSType &eos,
            const Spacetime &spacetime,
            Tvec position,
            u32 max_iter = 50,
            Tscal tol    = Tscal{1e-10}) -> typename Matter::PrimitiveVars {
            // STUB: Not implemented
            (void) U;
            (void) eos;
            (void) spacetime;
            (void) position;
            (void) max_iter;
            (void) tol;
            return W_old;
        }
    };

} // namespace shammodels::gsph::physics
