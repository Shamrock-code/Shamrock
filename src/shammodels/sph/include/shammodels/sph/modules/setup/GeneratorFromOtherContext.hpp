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
 * @file GeneratorFromOtherContext.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/DistributedData.hpp"
#include "shambase/stacktrace.hpp"
#include "shamalgs/collective/indexing.hpp"
#include "shammath/AABB.hpp"
#include "shammath/crystalLattice.hpp"
#include "shammodels/sph/modules/setup/ISPHSetupNode.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"

namespace shammodels::sph::modules {

    template<class Tvec>
    class GeneratorFromOtherContext : public ISPHSetupNode {
        using Tscal              = shambase::VecComponent<Tvec>;
        static constexpr u32 dim = shambase::VectorProperties<Tvec>::dimension;
        using Lattice            = shammath::LatticeCubic<Tvec>;
        using LatticeIter        = typename shammath::LatticeCubic<Tvec>::IteratorDiscontinuous;

        ShamrockCtx &context;
        shambase::DistributedData<shamrock::patch::PatchDataLayer> data_other;

        public:
        GeneratorFromOtherContext(ShamrockCtx &context, ShamrockCtx &context_other)
            : context(context) {

            context_other.sched->for_each_patchdata_nonempty(
                [&](const shamrock::patch::Patch &patch, shamrock::patch::PatchDataLayer &pdat) {
                    data_other.add_obj(patch.id_patch, pdat.duplicate());
                });
        }

        bool is_done() {
            bool ret = true;
            data_other.for_each([&](u64 id_patch, shamrock::patch::PatchDataLayer &pdat) {
                if (pdat.get_obj_cnt() > 0) {
                    ret = false;
                }
            });
            return ret;
        }

        shamrock::patch::PatchDataLayer next_n(u32 nmax) {
            StackEntry stack_loc{};

            using namespace shamrock::patch;
            PatchScheduler &sched = shambase::get_check_ref(context.sched);
            auto dev_sched        = shamsys::instance::get_compute_scheduler_ptr();

            // Make a patchdata to receive the data from the other context
            PatchDataLayer tmp(sched.get_layout_ptr());

            data_other.for_each([&](u64 id_patch, shamrock::patch::PatchDataLayer &pdat) {
                if (pdat.get_obj_cnt() > 0 && tmp.get_obj_cnt() < nmax) {

                    u32 remain_to_gen = nmax - tmp.get_obj_cnt();
                    remain_to_gen     = std::min(remain_to_gen, pdat.get_obj_cnt());

                    if (remain_to_gen == 0) {
                        return;
                    }

                    sham::DeviceBuffer<u32> indices(remain_to_gen, dev_sched);

                    indices.fill_lambda([&](u32 i) {
                        return i;
                    });

                    pdat.extract_elements(indices, tmp);
                }
            });

            return tmp;
        }

        std::string get_name() { return "GeneratorFromOtherContext"; }
        ISPHSetupNode_Dot get_dot_subgraph() { return ISPHSetupNode_Dot{get_name(), 0, {}}; }
    };

} // namespace shammodels::sph::modules
