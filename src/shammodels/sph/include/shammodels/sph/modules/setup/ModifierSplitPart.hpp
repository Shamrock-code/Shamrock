// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file ModifierSplitPart.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamalgs/collective/InvariantParallelGenerator.hpp"
#include "shamalgs/collective/indexing.hpp"
#include "shammodels/sph/SolverConfig.hpp"
#include "shammodels/sph/modules/setup/ISPHSetupNode.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"

namespace shammodels::sph::modules {

    template<class Tvec>
    class ModifierSplitPart : public ISPHSetupNode {
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;

        ShamrockCtx &context;

        SetupNodePtr parent;

        shamalgs::collective::InvariantParallelGenerator<std::mt19937_64> generator;
        u64 n_split;
        Tscal h_scaling;

        public:
        ModifierSplitPart(
            ShamrockCtx &context,
            SetupNodePtr parent,
            u64 n_split,
            u64 seed,
            Tscal h_scaling = 1. / 1.5)
            : context(context), parent(parent), n_split(n_split), generator(seed),
              h_scaling(h_scaling) {
            if (n_split == 0) {
                throw shambase::make_except_with_loc<std::invalid_argument>(
                    "n_split must be greater than 0");
            }
        }

        bool is_done() override { return parent->is_done(); }

        shamrock::patch::PatchDataLayer next_n(u32 nmax) override;

        std::string get_name() override { return "ModifierSplitPart"; }
        ISPHSetupNode_Dot get_dot_subgraph() override {
            return ISPHSetupNode_Dot{get_name(), 2, {parent->get_dot_subgraph()}};
        }
    };
} // namespace shammodels::sph::modules
