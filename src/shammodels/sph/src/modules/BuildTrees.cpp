// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file BuildTrees.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shammodels/sph/modules/BuildTrees.hpp"
#include "shammath/AABB.hpp"
#include "shammodels/sph/SPHSolverImpl.hpp"

namespace shammodels::sph::modules {

    template<class Tvec, template<class> class SPHKernel>
    void BuildTrees<Tvec, SPHKernel>::build_merged_pos_trees() {

        // interface_control
        using GhostHandle        = sph::BasicSPHGhostHandler<Tvec>;
        using GhostHandleCache   = typename GhostHandle::CacheMap;
        using PreStepMergedField = typename GhostHandle::PreStepMergedField;

        StackEntry stack_loc{};

        SPHSolverImpl solver(context);

        auto &merged_xyzh = storage.merged_xyzh.get();
        auto dev_sched    = shamsys::instance::get_compute_scheduler_ptr();

        shambase::DistributedData<RTree> trees
            = merged_xyzh.template map<RTree>([&](u64 id, PreStepMergedField &merged) {
                  Tvec bmin = merged.bounds.lower;
                  Tvec bmax = merged.bounds.upper;

                  auto bvh = RTree::make_empty(dev_sched);
                  bvh.rebuild_from_positions(
                      merged.field_pos.get_buf(),
                      merged.field_pos.get_obj_cnt(),
                      shammath::AABB<Tvec>(bmin, bmax),
                      solver_config.tree_reduction_level);

                  return bvh;
              });

        storage.merged_pos_trees.set(std::move(trees));
    };

} // namespace shammodels::sph::modules

using namespace shammath;

template class shammodels::sph::modules::BuildTrees<f64_3, M4>;
template class shammodels::sph::modules::BuildTrees<f64_3, M6>;
template class shammodels::sph::modules::BuildTrees<f64_3, M8>;

template class shammodels::sph::modules::BuildTrees<f64_3, C2>;
template class shammodels::sph::modules::BuildTrees<f64_3, C4>;
template class shammodels::sph::modules::BuildTrees<f64_3, C6>;
