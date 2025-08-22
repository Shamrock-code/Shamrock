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

#include "shambase/logs/loglevels.hpp"
#include "shambase/numeric_limits.hpp"
#include "shammath/AABB.hpp"
#include "shammodels/sph/modules/BuildTrees.hpp"
#include <cmath>

namespace shammodels::sph::modules {

    template<class Tvec, template<class> class SPHKernel>
    void BuildTrees<Tvec, SPHKernel>::build_merged_pos_trees() {

        // interface_control
        using GhostHandle        = sph::BasicSPHGhostHandler<Tvec>;
        using GhostHandleCache   = typename GhostHandle::CacheMap;
        using PreStepMergedField = typename GhostHandle::PreStepMergedField;

        StackEntry stack_loc{};

        auto &merged_xyzh = storage.merged_xyzh.get();
        auto dev_sched    = shamsys::instance::get_compute_scheduler_ptr();

        shambase::DistributedData<RTree> trees
            = merged_xyzh.template map<RTree>([&](u64 id, PreStepMergedField &merged) {
                  PatchDataField<Tvec> &pos = merged.field_pos;
                  Tvec bmax                 = pos.compute_max();
                  Tvec bmin                 = pos.compute_min();

                  shammath::AABB<Tvec> aabb(bmin, bmax);

                  Tscal infty = std::numeric_limits<Tscal>::infinity();

                  // ensure that no particle is on the boundary of the AABB
                  // TODO: make this a aabb function at some point
                  aabb.lower[0] = std::nextafter(aabb.lower[0], -infty);
                  aabb.lower[1] = std::nextafter(aabb.lower[1], -infty);
                  aabb.lower[2] = std::nextafter(aabb.lower[2], -infty);
                  aabb.upper[0] = std::nextafter(aabb.upper[0], infty);
                  aabb.upper[1] = std::nextafter(aabb.upper[1], infty);
                  aabb.upper[2] = std::nextafter(aabb.upper[2], infty);

                  auto bvh = RTree::make_empty(dev_sched);
                  bvh.rebuild_from_positions(
                      merged.field_pos.get_buf(),
                      merged.field_pos.get_obj_cnt(),
                      aabb,
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
