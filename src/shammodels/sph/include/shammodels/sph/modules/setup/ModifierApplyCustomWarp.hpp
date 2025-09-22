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
 * @file ModifierApplyCustomWarp.hpp
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
    class ModifierApplyCustomWarp : public ISPHSetupNode {
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        using Config = SolverConfig<Tvec, SPHKernel>;

        ShamrockCtx &context;
        SetupNodePtr parent;

        std::function<Tscal(Tscal)> inc_profile;
        std::function<Tscal(Tscal)> psi_profile;
        std::function<Tvec(Tscal)> k_profile;

        public:
        ModifierApplyCustomWarp(
            ShamrockCtx &context,
            Config &solver_config,
            SetupNodePtr parent,
            std::function<Tscal(Tscal)> inc_profile,
            std::function<Tscal(Tscal)> psi_profile,
            std::function<Tvec(Tscal)> k_profile)
            : context(context), parent(parent), inc_profile(inc_profile), psi_profile(psi_profile),
              k_profile(k_profile) {}

        bool is_done() { return parent->is_done(); }

        shamrock::patch::PatchDataLayer next_n(u32 nmax);

        std::string get_name() { return "ApplyCustomWarp"; }

        ISPHSetupNode_Dot get_dot_subgraph() {
            return ISPHSetupNode_Dot{get_name(), 2, {parent->get_dot_subgraph()}};
        }
    };
} // namespace shammodels::sph::modules
