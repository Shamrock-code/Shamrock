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
 * @file PhysicsComposition.hpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief Compose physics from orthogonal traits
 *
 * This file provides the main template for combining:
 * - Matter model (HydroMatter, MHDMatter, RadHydroMatter)
 * - Relativity level (Newtonian, SR, GR)
 * - Spacetime (Minkowski, Schwarzschild, Kerr, Numerical)
 * - Solver mode (NormalEvolution, RelaxationMode, DustMode)
 *
 * The composition provides compile-time feature flags that control
 * which solvergraph nodes are instantiated.
 */

#include "shammodels/gsph/physics/MatterTraits.hpp"
#include "shammodels/gsph/physics/RelativityTraits.hpp"
#include "shammodels/gsph/physics/SpacetimeTraits.hpp"
#include <type_traits>

// Forward declaration of solver modes (defined in SolverModes.hpp)
namespace shammodels::gsph::modes {
    struct NormalEvolution;
    template<class Tvec>
    struct RelaxationMode;
    struct DustMode;
} // namespace shammodels::gsph::modes

namespace shammodels::gsph {

    /**
     * @brief Compose physics from orthogonal choices
     *
     * This is the main physics configuration template. Each template parameter
     * represents an independent axis of physics choices:
     *
     * - MatterModel: What fields exist (hydro, MHD, radiation)
     * - RelativityLevel: How equations transform (Newtonian, SR, GR)
     * - SpacetimeModel: Geometry (flat, curved)
     * - SolverMode: Evolution behavior (normal, relaxation, dust)
     *
     * @tparam Tvec Vector type (e.g., f64_3)
     * @tparam MatterModel Matter traits template (e.g., physics::HydroMatter)
     * @tparam RelativityLevel Relativity traits template
     * @tparam SpacetimeModel Spacetime traits (default: MinkowskiSpacetime)
     * @tparam SolverMode Solver mode (default: NormalEvolution)
     */
    template<
        class Tvec,
        template<class> class MatterModel,
        template<class, class, class...> class RelativityLevel,
        class SpacetimeModel = physics::MinkowskiSpacetime<Tvec>,
        class SolverMode     = modes::NormalEvolution>
    struct PhysicsComposition {
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        // =====================================================================
        // Component types
        // =====================================================================

        using Matter     = MatterModel<Tvec>;
        using Spacetime  = SpacetimeModel;
        using Relativity = RelativityLevel<Tvec, Matter, Spacetime>;
        using Mode       = SolverMode;

        // =====================================================================
        // Derived types
        // =====================================================================

        using PrimitiveVars = typename Matter::PrimitiveVars;
        using ConservedVars = typename Relativity::ConservedVars;

        // =====================================================================
        // Compile-time feature flags
        // These control which solvergraph nodes are instantiated
        // =====================================================================

        /// Does this physics require primitive variable recovery?
        /// (True for SR/GR, false for Newtonian)
        static constexpr bool needs_primitive_recovery = Relativity::needs_primitive_recovery;

        /// Does this physics have a Lorentz factor?
        static constexpr bool needs_lorentz_factor = Relativity::needs_lorentz_factor;

        /// Does this physics need metric data at each particle?
        static constexpr bool needs_metric = Spacetime::is_curved;

        /// Does this physics have divergence constraints (div B = 0)?
        static constexpr bool needs_divergence_cleaning = Matter::has_divergence_constraint;

        /// Does this physics have geometric source terms?
        static constexpr bool needs_geometric_source = Spacetime::is_curved;

        /// Does this physics have radiation cooling?
        static constexpr bool needs_radiation_cooling = Matter::has_radiation_cooling;

        /// Does this mode have convergence checking (relaxation)?
        /// Uses SFINAE to check if SolverMode has static constexpr has_convergence_check
        template<class T, class = void>
        struct has_convergence_check_trait : std::false_type {};

        template<class T>
        struct has_convergence_check_trait<T, std::void_t<decltype(T::has_convergence_check)>>
            : std::bool_constant<T::has_convergence_check> {};

        static constexpr bool has_convergence_check
            = has_convergence_check_trait<SolverMode>::value;
    };

    // =========================================================================
    // Convenient type aliases for common configurations
    // =========================================================================

    // -------------------------------------------------------------------------
    // Pure Hydrodynamics
    // -------------------------------------------------------------------------

    /// Newtonian hydrodynamics (default, most common)
    template<class Tvec>
    using NewtonianHydro
        = PhysicsComposition<Tvec, physics::HydroMatter, physics::NewtonianPhysics>;

    /// Special Relativistic hydrodynamics (STUB)
    template<class Tvec>
    using SRHydro = PhysicsComposition<Tvec, physics::HydroMatter, physics::SRPhysics>;

    // GRHydro requires a metric type parameter, so it's not a simple alias
    // Usage: PhysicsComposition<Tvec, HydroMatter, GRPhysics, SchwarzschildSpacetime<Tvec>>

    // -------------------------------------------------------------------------
    // Magnetohydrodynamics (STUBS)
    // -------------------------------------------------------------------------

    /// Newtonian MHD
    template<class Tvec>
    using NewtonianMHD = PhysicsComposition<Tvec, physics::MHDMatter, physics::NewtonianPhysics>;

    /// Special Relativistic MHD
    template<class Tvec>
    using SRMHD = PhysicsComposition<Tvec, physics::MHDMatter, physics::SRPhysics>;

    // -------------------------------------------------------------------------
    // Radiation Hydrodynamics (STUBS)
    // -------------------------------------------------------------------------

    /// Newtonian radiation hydrodynamics
    template<class Tvec>
    using NewtonianRadHydro
        = PhysicsComposition<Tvec, physics::RadHydroMatter, physics::NewtonianPhysics>;

    // -------------------------------------------------------------------------
    // Special Modes
    // -------------------------------------------------------------------------

    /// Newtonian hydrodynamics with relaxation damping (Lane-Emden equilibrium)
    template<class Tvec>
    using NewtonianHydroRelaxation = PhysicsComposition<
        Tvec,
        physics::HydroMatter,
        physics::NewtonianPhysics,
        physics::MinkowskiSpacetime<Tvec>,
        modes::RelaxationMode<Tvec>>;

    /// Newtonian hydrodynamics in dust (pressureless) limit
    template<class Tvec>
    using NewtonianHydroDust = PhysicsComposition<
        Tvec,
        physics::HydroMatter,
        physics::NewtonianPhysics,
        physics::MinkowskiSpacetime<Tvec>,
        modes::DustMode>;

} // namespace shammodels::gsph
