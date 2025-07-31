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
 * @file SPHSolverImpl.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shammath/sphkernels.hpp"
#include "shammodels/sph/BasicSPHGhosts.hpp"
#include "shamrock/scheduler/InterfacesUtility.hpp"
#include "shamrock/scheduler/PatchScheduler.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include "shamtree/RadixTree.hpp"
#include "shamtree/TreeTraversal.hpp"

namespace shammodels {

    class SPHSolverImpl {
        public:
        using flt                = f64;
        using vec                = f64_3;
        static constexpr u32 dim = 3;
        using u_morton           = u32;
        using Kernel             = shammath::M4<flt>;

        static constexpr flt Rkern = Kernel::Rkern;

        ShamrockCtx &context;
        inline PatchScheduler &scheduler() { return shambase::get_check_ref(context.sched); }
        SPHSolverImpl(ShamrockCtx &ctx) : context(ctx) {};

        [[deprecated]]
        static shamrock::tree::ObjectCache build_neigh_cache(
            u32 start_offset,
            u32 obj_cnt,
            sycl::buffer<vec> &buf_xyz,
            sycl::buffer<flt> &buf_hpart,
            RadixTree<u_morton, vec> &tree,
            sycl::buffer<flt> &tree_field_hmax);

        [[deprecated]]
        static shamrock::tree::ObjectCache build_hiter_neigh_cache(
            u32 start_offset,
            u32 obj_cnt,
            sycl::buffer<vec> &buf_xyz,
            sycl::buffer<flt> &buf_hpart,
            RadixTree<u_morton, vec> &tree,
            flt h_tolerance);

        using GhostHandle        = sph::BasicSPHGhostHandler<vec>;
        using GhostHandleCache   = typename GhostHandle::CacheMap;
        using PreStepMergedField = typename GhostHandle::PreStepMergedField;

        using MergedPositions = shambase::DistributedData<PreStepMergedField>;
        using RTree           = RadixTree<u_morton, vec>;

        static shambase::DistributedData<RTree>
        make_merge_patch_trees(MergedPositions &merged_xyzh, u32 reduction_level) {
            shambase::DistributedData<RTree> trees
                = merged_xyzh.map<RTree>([&](u64 id, PreStepMergedField &merged) {
                      vec bmin = merged.bounds.lower;
                      vec bmax = merged.bounds.upper;

                      RTree tree(
                          shamsys::instance::get_compute_scheduler_ptr(),
                          {bmin, bmax},
                          merged.field_pos.get_buf(),
                          merged.field_pos.get_obj_cnt(),
                          reduction_level);

                      return tree;
                  });

            trees.for_each([&](u64 id, RTree &tree) {
                tree.compute_cell_ibounding_box(shamsys::instance::get_compute_queue());
                tree.convert_bounding_box(shamsys::instance::get_compute_queue());
            });

            return trees;
        }
    };

} // namespace shammodels
