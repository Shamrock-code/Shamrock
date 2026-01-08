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
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief GSPH spatial tree building implementation
 */

#include "shambase/stacktrace.hpp"
#include "shambackends/kernel_call.hpp"
#include "shammodels/gsph/modules/BuildTrees.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtree/KarrasRadixTreeField.hpp"

namespace shammodels::gsph::modules {

    template<class Tvec, template<class> class SPHKernel>
    void BuildTrees<Tvec, SPHKernel>::build_merged_pos_trees() {
        StackEntry stack_loc{};

        auto &merged_xyzh = storage.merged_xyzh.get();
        auto dev_sched    = shamsys::instance::get_compute_scheduler_ptr();

        shambase::DistributedData<RTree> trees
            = merged_xyzh.template map<RTree>([&](u64 id, shamrock::patch::PatchDataLayer &merged) {
                  PatchDataField<Tvec> &pos = merged.template get_field<Tvec>(0);
                  Tvec bmax                 = pos.compute_max();
                  Tvec bmin                 = pos.compute_min();

                  shammath::AABB<Tvec> aabb(bmin, bmax);

                  Tscal infty = std::numeric_limits<Tscal>::infinity();

                  aabb.lower[0] = std::nextafter(aabb.lower[0], -infty);
                  aabb.lower[1] = std::nextafter(aabb.lower[1], -infty);
                  aabb.lower[2] = std::nextafter(aabb.lower[2], -infty);
                  aabb.upper[0] = std::nextafter(aabb.upper[0], infty);
                  aabb.upper[1] = std::nextafter(aabb.upper[1], infty);
                  aabb.upper[2] = std::nextafter(aabb.upper[2], infty);

                  auto bvh = RTree::make_empty(dev_sched);
                  bvh.rebuild_from_positions(
                      pos.get_buf(), pos.get_obj_cnt(), aabb, solver_config.tree_reduction_level);

                  return bvh;
              });

        storage.merged_pos_trees.set(std::move(trees));
    }

    template<class Tvec, template<class> class SPHKernel>
    void BuildTrees<Tvec, SPHKernel>::compute_presteps_rint() {
        StackEntry stack_loc{};

        auto &xyzh_merged = storage.merged_xyzh.get();
        auto dev_sched    = shamsys::instance::get_compute_scheduler_ptr();

        storage.rtree_rint_field.set(
            storage.merged_pos_trees.get().template map<shamtree::KarrasRadixTreeField<Tscal>>(
                [&](u64 id, RTree &rtree) -> shamtree::KarrasRadixTreeField<Tscal> {
                    shamrock::patch::PatchDataLayer &tmp = xyzh_merged.get(id);
                    auto &buf                            = tmp.get_field_buf_ref<Tscal>(1);
                    auto buf_int = shamtree::new_empty_karras_radix_tree_field<Tscal>();

                    auto ret = shamtree::compute_tree_field_max_field<Tscal>(
                        rtree.structure,
                        rtree.reduced_morton_set.get_leaf_cell_iterator(),
                        std::move(buf_int),
                        buf);

                    // c_smooth defaults to 1.0 for Newtonian, larger for SR
                    Tscal htol = solver_config.htol_up_coarse_cycle * solver_config.c_smooth;
                    sham::kernel_call(
                        dev_sched->get_queue(),
                        sham::MultiRef{},
                        sham::MultiRef{ret.buf_field},
                        ret.buf_field.get_size(),
                        [htol](u32 i, Tscal *h_tree) {
                            h_tree[i] *= htol;
                        });

                    return std::move(ret);
                }));
    }

    // Explicit instantiations
    template class BuildTrees<f64_3, shammath::M4>;
    template class BuildTrees<f64_3, shammath::M6>;
    template class BuildTrees<f64_3, shammath::M8>;
    template class BuildTrees<f64_3, shammath::C2>;
    template class BuildTrees<f64_3, shammath::C4>;
    template class BuildTrees<f64_3, shammath::C6>;
    template class BuildTrees<f64_3, shammath::TGauss3>;

} // namespace shammodels::gsph::modules
