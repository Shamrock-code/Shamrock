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
 * @file ModifierApplyDiscWarp.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shamalgs/collective/indexing.hpp"
#include "shammodels/sph/SolverConfig.hpp"
#include "shammodels/sph/modules/setup/ISPHSetupNode.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"

namespace shammodels::sph::modules {
    template<class Tvec, template<class> class SPHKernel>
    class ModifierApplyDiscWarp : public ISPHSetupNode {
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        using Config = SolverConfig<Tvec, SPHKernel>;

        ShamrockCtx &context;

        SetupNodePtr parent;

        Tscal Rwarp_;
        Tscal Hwarp_;
        Tscal inclination_;
        Tscal posangle_;

        public:
        ModifierApplyDiscWarp(
            ShamrockCtx &context,
            Config &solver_config,
            SetupNodePtr parent,
            Tscal Rwarp,
            Tscal Hwarp,
            Tscal inclination,
            Tscal posangle)
            : context(context), parent(parent), Rwarp_(Rwarp), Hwarp_(Hwarp),
              inclination_(inclination), posangle_(posangle) {}

        bool is_done() { return parent->is_done(); }

        shamrock::patch::PatchData next_n(u32 nmax);

        std::string get_name() { return "ApplyDiscWarp"; }
        ISPHSetupNode_Dot get_dot_subgraph() {
            return ISPHSetupNode_Dot{get_name(), 2, {parent->get_dot_subgraph()}};
        }
    };
} // namespace shammodels::sph::modules
