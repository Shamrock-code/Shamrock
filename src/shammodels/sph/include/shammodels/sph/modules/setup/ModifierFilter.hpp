// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file ModifierFilter.hpp
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shamalgs/collective/indexing.hpp"
#include "shammodels/sph/SolverConfig.hpp"
#include "shammodels/sph/modules/setup/ISPHSetupNode.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include <functional>

namespace shammodels::sph::modules {
    template<class Tvec, template<class> class SPHKernel>
    class ModifierFilter : public ISPHSetupNode {
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        using Config = SolverConfig<Tvec, SPHKernel>;

        ShamrockCtx &context;

        SetupNodePtr parent;

        std::function<bool(Tvec)> filter;

        public:
        ModifierFilter(ShamrockCtx &context, SetupNodePtr parent, std::function<bool(Tvec)> filter)
            : context(context), parent(parent), filter(filter) {}

        bool is_done() { return parent->is_done(); }

        shamrock::patch::PatchData next_n(u32 nmax);

        std::string get_name() { return "ModifierFilter"; }
        ISPHSetupNode_Dot get_dot_subgraph() {
            return ISPHSetupNode_Dot{get_name(), 2, {parent->get_dot_subgraph()}};
        }
    };
} // namespace shammodels::sph::modules
