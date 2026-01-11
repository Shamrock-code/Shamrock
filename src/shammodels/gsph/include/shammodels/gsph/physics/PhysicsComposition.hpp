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
 * Combines matter model, relativity level, spacetime, and solver mode
 * into a single physics configuration with compile-time feature flags.
 */

#include "shammodels/gsph/physics/MatterTraits.hpp"
#include "shammodels/gsph/physics/RelativityTraits.hpp"
#include "shammodels/gsph/physics/SpacetimeTraits.hpp"
#include <type_traits>

namespace shammodels::gsph::modes {
    struct NormalEvolution;
} // namespace shammodels::gsph::modes

namespace shammodels::gsph {

    /**
     * @brief Compose physics from orthogonal choices
     *
     * @tparam Tvec Vector type
     * @tparam MatterModel Matter traits template
     * @tparam RelativityLevel Relativity traits template
     * @tparam SpacetimeModel Spacetime traits
     * @tparam SolverMode Solver mode
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

        using Matter     = MatterModel<Tvec>;
        using Spacetime  = SpacetimeModel;
        using Relativity = RelativityLevel<Tvec, Matter, Spacetime>;
        using Mode       = SolverMode;

        using PrimitiveVars = typename Matter::PrimitiveVars;
        using ConservedVars = typename Relativity::ConservedVars;

        static constexpr bool needs_primitive_recovery  = Relativity::needs_primitive_recovery;
        static constexpr bool needs_lorentz_factor      = Relativity::needs_lorentz_factor;
        static constexpr bool needs_metric              = Spacetime::is_curved;
        static constexpr bool needs_divergence_cleaning = Matter::has_divergence_constraint;

        template<class T, class = void>
        struct has_convergence_check_trait : std::false_type {};

        template<class T>
        struct has_convergence_check_trait<T, std::void_t<decltype(T::has_convergence_check)>>
            : std::bool_constant<T::has_convergence_check> {};

        static constexpr bool has_convergence_check
            = has_convergence_check_trait<SolverMode>::value;
    };

    /// Newtonian hydrodynamics
    template<class Tvec>
    using NewtonianHydro
        = PhysicsComposition<Tvec, physics::HydroMatter, physics::NewtonianPhysics>;

    /// Special Relativistic hydrodynamics (STUB)
    template<class Tvec>
    using SRHydro = PhysicsComposition<Tvec, physics::HydroMatter, physics::SRPhysics>;

} // namespace shammodels::gsph
