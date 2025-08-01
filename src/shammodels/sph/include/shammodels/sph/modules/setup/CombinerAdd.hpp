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
 * @file CombinerAdd.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamalgs/collective/indexing.hpp"
#include "shammath/AABB.hpp"
#include "shammath/crystalLattice.hpp"
#include "shammodels/sph/modules/setup/ISPHSetupNode.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"

namespace shammodels::sph::modules {

    template<class Tvec>
    class CombinerAdd : public ISPHSetupNode {
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;
        using Lattice            = shammath::LatticeHCP<Tvec>;
        using LatticeIter        = typename shammath::LatticeHCP<Tvec>::IteratorDiscontinuous;

        ShamrockCtx &context;

        SetupNodePtr parent1;
        SetupNodePtr parent2;

        public:
        CombinerAdd(ShamrockCtx &context, SetupNodePtr parent1, SetupNodePtr parent2)
            : context(context), parent1(parent1), parent2(parent2) {}

        bool is_done() { return parent1->is_done() && parent2->is_done(); }

        shamrock::patch::PatchDataLayer next_n(u32 nmax) {

            u32 cnt1 = (parent1->is_done()) ? 0 : (nmax / 2);
            u32 cnt2 = nmax - cnt1;

            shamrock::patch::PatchDataLayer tmp1 = parent1->next_n(cnt1);
            shamrock::patch::PatchDataLayer tmp2 = parent2->next_n(cnt2);

            tmp1.insert_elements(tmp2);
            return tmp1;
        }

        std::string get_name() { return "CombinerAdd"; }
        ISPHSetupNode_Dot get_dot_subgraph() {
            return ISPHSetupNode_Dot{
                get_name(), 2, {parent1->get_dot_subgraph(), parent2->get_dot_subgraph()}};
        }
    };

} // namespace shammodels::sph::modules
