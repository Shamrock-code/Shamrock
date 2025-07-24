// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file BuildTrees.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shammodels/sph/modules/BuildTrees.hpp"
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

        shambase::DistributedData<RTree> trees
            = merged_xyzh.template map<RTree>([&](u64 id, PreStepMergedField &merged) {
                  Tvec bmin = merged.bounds.lower;
                  Tvec bmax = merged.bounds.upper;

                  RTree tree(
                      shamsys::instance::get_compute_scheduler_ptr(),
                      {bmin, bmax},
                      merged.field_pos.get_buf(),
                      merged.field_pos.get_obj_cnt(),
                      solver_config.tree_reduction_level);

                  return tree;
              });

        trees.for_each([&](u64 id, RTree &tree) {
            tree.compute_cell_ibounding_box(shamsys::instance::get_compute_queue());
            tree.convert_bounding_box(shamsys::instance::get_compute_queue());
        });

        bool corect_boxes = solver_config.use_two_stage_search;
        if (corect_boxes) {

            trees.for_each([&](u64 id, RTree &tree) {
                u32 leaf_count          = tree.tree_reduced_morton_codes.tree_leaf_count;
                u32 internal_cell_count = tree.tree_struct.internal_cell_count;
                u32 tot_count           = leaf_count + internal_cell_count;

                sycl::buffer<Tvec> tmp_min_cell(tot_count);
                sycl::buffer<Tvec> tmp_max_cell(tot_count);

                sham::DeviceBuffer<Tvec> &buf_part_pos = merged_xyzh.get(id).field_pos.get_buf();

                sham::EventList depends_list;
                auto acc_pos = buf_part_pos.get_read_access(depends_list);

                sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

                auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                    shamrock::tree::ObjectIterator cell_looper(tree, cgh);

                    u32 leaf_offset = tree.tree_struct.internal_cell_count;

                    sycl::accessor comp_min{tmp_min_cell, cgh, sycl::write_only, sycl::no_init};
                    sycl::accessor comp_max{tmp_max_cell, cgh, sycl::write_only, sycl::no_init};

                    // how to lose a f****ing afternoon :
                    // 1 - code a nice algorithm that should optimize the code
                    // 2 - pass all the tests
                    // 3 - benchmark it and discover big loss in perf for no reasons
                    // 4 - change a parameter and discover a segfault (on GPU to have more fun ....)
                    // 5 - find that actually the core algorithm of the code create a bug in the new
                    // thing 6 - discover that every value in everything is wrong 7 - spent the
                    // whole night on it 8 - start putting prints everywhere 9 - isolate a bugged id
                    // 10 - try to understand why a f***ing leaf is as big as the root of the tree
                    // 11 - **** a few hours latter 12 - the goddam c++ standard define
                    // std::numeric_limits<float>::min() to be epsilon instead of -max 13 - road
                    // rage 14
                    // - open a bier alt f4 the ide

                    Tvec imin = shambase::VectorProperties<Tvec>::get_max();
                    Tvec imax = -shambase::VectorProperties<Tvec>::get_max();

                    shambase::parallel_for(cgh, leaf_count, "compute leaf boxes", [=](u64 leaf_id) {
                        Tvec min = imin;
                        Tvec max = imax;

                        cell_looper.iter_object_in_cell(leaf_id + leaf_offset, [&](u32 part_id) {
                            Tvec r = acc_pos[part_id];

                            min = sham::min(min, r);
                            max = sham::max(max, r);
                        });

                        comp_min[leaf_offset + leaf_id] = min;
                        comp_max[leaf_offset + leaf_id] = max;
                    });
                });

                buf_part_pos.complete_event_state(e);

                //{
                //    u32 leaf_offset = tree.tree_struct.internal_cell_count;
                //    sycl::host_accessor pos_min_cell  {tmp_min_cell};
                //    sycl::host_accessor pos_max_cell  {tmp_max_cell};
                //
                //    for (u32 i = 0; i < 1000; i++) {
                //            logger::raw_ln(i,pos_max_cell[i+leaf_offset] -
                //            pos_min_cell[i+leaf_offset]);
                //
                //    }
                //}

                auto ker_reduc_hmax = [&](sycl::handler &cgh) {
                    u32 offset_leaf = internal_cell_count;

                    sycl::accessor comp_min{tmp_min_cell, cgh, sycl::read_write};
                    sycl::accessor comp_max{tmp_max_cell, cgh, sycl::read_write};

                    sycl::accessor rchild_id{
                        shambase::get_check_ref(tree.tree_struct.buf_rchild_id),
                        cgh,
                        sycl::read_only};
                    sycl::accessor lchild_id{
                        shambase::get_check_ref(tree.tree_struct.buf_lchild_id),
                        cgh,
                        sycl::read_only};
                    sycl::accessor rchild_flag{
                        shambase::get_check_ref(tree.tree_struct.buf_rchild_flag),
                        cgh,
                        sycl::read_only};
                    sycl::accessor lchild_flag{
                        shambase::get_check_ref(tree.tree_struct.buf_lchild_flag),
                        cgh,
                        sycl::read_only};

                    shambase::parallel_for(cgh, internal_cell_count, "propagate up", [=](u64 gid) {
                        u32 lid = lchild_id[gid] + offset_leaf * lchild_flag[gid];
                        u32 rid = rchild_id[gid] + offset_leaf * rchild_flag[gid];

                        Tvec bminl = comp_min[lid];
                        Tvec bminr = comp_min[rid];
                        Tvec bmaxl = comp_max[lid];
                        Tvec bmaxr = comp_max[rid];

                        Tvec bmin = sham::min(bminl, bminr);
                        Tvec bmax = sham::max(bmaxl, bmaxr);

                        comp_min[gid] = bmin;
                        comp_max[gid] = bmax;
                    });
                };

                for (u32 i = 0; i < tree.tree_depth; i++) {
                    shamsys::instance::get_compute_queue().submit(ker_reduc_hmax);
                }

                sycl::buffer<Tvec> &tree_bmin
                    = shambase::get_check_ref(tree.tree_cell_ranges.buf_pos_min_cell_flt);
                sycl::buffer<Tvec> &tree_bmax
                    = shambase::get_check_ref(tree.tree_cell_ranges.buf_pos_max_cell_flt);

                shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
                    shamrock::tree::ObjectIterator cell_looper(tree, cgh);

                    u32 leaf_offset = tree.tree_struct.internal_cell_count;

                    sycl::accessor comp_bmin{tmp_min_cell, cgh, sycl::read_only};
                    sycl::accessor comp_bmax{tmp_max_cell, cgh, sycl::read_only};

                    sycl::accessor tree_buf_min{tree_bmin, cgh, sycl::read_write};
                    sycl::accessor tree_buf_max{tree_bmax, cgh, sycl::read_write};

                    shambase::parallel_for(cgh, tot_count, "write in tree range", [=](u64 nid) {
                        Tvec load_min = comp_bmin[nid];
                        Tvec load_max = comp_bmax[nid];

                        tree_buf_min[nid] = load_min;
                        tree_buf_max[nid] = load_max;
                    });
                });
            });
        }

        storage.merged_pos_trees.set(std::move(trees));
    };

} // namespace shammodels::sph::modules

using namespace shammath;

template class shammodels::sph::modules::BuildTrees<f64_3, M4>;
template class shammodels::sph::modules::BuildTrees<f64_3, M6>;
template class shammodels::sph::modules::BuildTrees<f64_3, M8>;
