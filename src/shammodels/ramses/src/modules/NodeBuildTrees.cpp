// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file NodeBuildTrees.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/time.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shammath/AABB.hpp"
#include "shammodels/ramses/modules/NodeBuildTrees.hpp"
#include "shamtree/TreeTraversal.hpp"

namespace {
    template<class Umorton, class TgridVec>
    void __internal_correct_tree_bb(
        RadixTree<Umorton, TgridVec> &tree,
        sham::DeviceBuffer<TgridVec> &block_min,
        sham::DeviceBuffer<TgridVec> &block_max) {

        u32 leaf_count          = tree.tree_reduced_morton_codes.tree_leaf_count;
        u32 internal_cell_count = tree.tree_struct.internal_cell_count;
        u32 tot_count           = leaf_count + internal_cell_count;

        sycl::buffer<TgridVec> tmp_min_cell(tot_count);
        sycl::buffer<TgridVec> tmp_max_cell(tot_count);

        sham::DeviceBuffer<TgridVec> &buf_cell_min = block_min;
        sham::DeviceBuffer<TgridVec> &buf_cell_max = block_max;

        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

        sham::EventList depends_list;
        auto acc_bmin = buf_cell_min.get_read_access(depends_list);
        auto acc_bmax = buf_cell_max.get_read_access(depends_list);

        auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
            shamrock::tree::ObjectIterator cell_looper(tree, cgh);

            u32 leaf_offset = tree.tree_struct.internal_cell_count;

            sycl::accessor comp_min{tmp_min_cell, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor comp_max{tmp_max_cell, cgh, sycl::write_only, sycl::no_init};

            TgridVec imin = shambase::VectorProperties<TgridVec>::get_max();
            TgridVec imax = shambase::VectorProperties<TgridVec>::get_min();

            shambase::parallel_for(cgh, leaf_count, "compute leaf boxes", [=](u64 leaf_id) {
                TgridVec min = imin;
                TgridVec max = imax;

                cell_looper.iter_object_in_cell(leaf_id + leaf_offset, [&](u32 block_id) {
                    TgridVec bmin = acc_bmin[block_id];
                    TgridVec bmax = acc_bmax[block_id];

                    min = sham::min(min, bmin);
                    max = sham::max(max, bmax);
                });

                comp_min[leaf_offset + leaf_id] = min;
                comp_max[leaf_offset + leaf_id] = max;
            });
        });

        buf_cell_min.complete_event_state(e);
        buf_cell_max.complete_event_state(e);

        auto ker_reduc_hmax = [&](sycl::handler &cgh) {
            u32 offset_leaf = internal_cell_count;

            sycl::accessor comp_min{tmp_min_cell, cgh, sycl::read_write};
            sycl::accessor comp_max{tmp_max_cell, cgh, sycl::read_write};

            sycl::accessor rchild_id{
                shambase::get_check_ref(tree.tree_struct.buf_rchild_id), cgh, sycl::read_only};
            sycl::accessor lchild_id{
                shambase::get_check_ref(tree.tree_struct.buf_lchild_id), cgh, sycl::read_only};
            sycl::accessor rchild_flag{
                shambase::get_check_ref(tree.tree_struct.buf_rchild_flag), cgh, sycl::read_only};
            sycl::accessor lchild_flag{
                shambase::get_check_ref(tree.tree_struct.buf_lchild_flag), cgh, sycl::read_only};

            shambase::parallel_for(cgh, internal_cell_count, "propagate up", [=](u64 gid) {
                u32 lid = lchild_id[gid] + offset_leaf * lchild_flag[gid];
                u32 rid = rchild_id[gid] + offset_leaf * rchild_flag[gid];

                TgridVec bminl = comp_min[lid];
                TgridVec bminr = comp_min[rid];
                TgridVec bmaxl = comp_max[lid];
                TgridVec bmaxr = comp_max[rid];

                TgridVec bmin = sham::min(bminl, bminr);
                TgridVec bmax = sham::max(bmaxl, bmaxr);

                comp_min[gid] = bmin;
                comp_max[gid] = bmax;
            });
        };

        for (u32 i = 0; i < tree.tree_depth; i++) {
            shamsys::instance::get_compute_queue().submit(ker_reduc_hmax);
        }

        sycl::buffer<TgridVec> &tree_bmin
            = shambase::get_check_ref(tree.tree_cell_ranges.buf_pos_min_cell_flt);
        sycl::buffer<TgridVec> &tree_bmax
            = shambase::get_check_ref(tree.tree_cell_ranges.buf_pos_max_cell_flt);

        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
            shamrock::tree::ObjectIterator cell_looper(tree, cgh);

            u32 leaf_offset = tree.tree_struct.internal_cell_count;

            sycl::accessor comp_bmin{tmp_min_cell, cgh, sycl::read_only};
            sycl::accessor comp_bmax{tmp_max_cell, cgh, sycl::read_only};

            sycl::accessor tree_buf_min{tree_bmin, cgh, sycl::read_write};
            sycl::accessor tree_buf_max{tree_bmax, cgh, sycl::read_write};

            shambase::parallel_for(cgh, tot_count, "write in tree range", [=](u64 nid) {
                TgridVec load_min = comp_bmin[nid];
                TgridVec load_max = comp_bmax[nid];

                // if(
                //     (!shambase::vec_equals(load_min,tree_buf_min[nid]))
                //   || !shambase::vec_equals(load_max,tree_buf_max[nid])
                //     )
                //{
                //
                //    sycl::ext::oneapi::experimental::printf(
                //        "%ld : (%ld %ld %ld) -> (%ld %ld %ld) & (%ld %ld %ld) -> (%ld %ld %ld)\n",
                //        nid,
                //        tree_buf_min[nid].x(),tree_buf_min[nid].y(),tree_buf_min[nid].z(),
                //        load_min.x(),load_min.y(),load_min.z(),
                //        tree_buf_max[nid].x(),tree_buf_max[nid].y(),tree_buf_max[nid].z(),
                //        load_max.x(),load_max.y(),load_max.z()
                //    );
                //
                //}

                tree_buf_min[nid] = load_min;
                tree_buf_max[nid] = load_max;
            });
        });
    }
} // namespace

namespace shammodels::basegodunov::modules {

    template<class Umorton, class TgridVec>
    void NodeBuildTrees<Umorton, TgridVec>::_impl_evaluate_internal() {
        auto edges = get_edges();

        auto &block_min = edges.block_min;
        auto &block_max = edges.block_max;

        const shambase::DistributedData<u32> &indexes_dd = edges.sizes.indexes;

        // TODO move the bounding box computation to another node.

        shambase::DistributedData<shammath::AABB<TgridVec>> bounds = {};

        bounds = indexes_dd.template map<shammath::AABB<TgridVec>>([&](u64 id, auto &merged) {
            TgridVec min_bound = block_min.get_field(id).compute_min();
            TgridVec max_bound = block_max.get_field(id).compute_max();

            // logger::raw_ln("AABB", id, min_bound, max_bound);

            return shammath::AABB<TgridVec>{min_bound, max_bound};
        });

        shambase::DistributedData<RTree> trees
            = indexes_dd.template map<RTree>([&](u64 id, auto &merged) {
                  shamlog_debug_ln("AMR", "compute tree for merged patch", id);

                  auto aabb = bounds.get(id);

                  TgridVec bmin = aabb.lower;
                  TgridVec bmax = aabb.upper;

                  TgridVec diff = bmax - bmin;
                  diff.x()      = shambase::roundup_pow2(diff.x());
                  diff.y()      = shambase::roundup_pow2(diff.y());
                  diff.z()      = shambase::roundup_pow2(diff.z());
                  bmax          = bmin + diff;

                  auto &field_pos = block_min.get_field(id);

                  RTree tree(
                      shamsys::instance::get_compute_scheduler_ptr(),
                      {bmin, bmax},
                      field_pos.get_buf(),
                      field_pos.get_obj_cnt(),
                      reduction_level);

                  return tree;
              });

        trees.for_each([](u64 id, RTree &tree) {
            tree.compute_cell_ibounding_box(shamsys::instance::get_compute_queue());
        });

        trees.for_each([](u64 id, RTree &tree) {
            tree.convert_bounding_box(shamsys::instance::get_compute_queue());
        });

        trees.for_each([&](u64 id, RTree &tree) {
            __internal_correct_tree_bb(
                tree, block_min.get_field(id).get_buf(), block_max.get_field(id).get_buf());
        });

        edges.trees.trees = std::move(trees);
    }

    template<class Umorton, class TgridVec>
    std::string NodeBuildTrees<Umorton, TgridVec>::_impl_get_tex() const {

        std::string sizes     = get_ro_edge_base(0).get_tex_symbol();
        std::string block_min = get_ro_edge_base(1).get_tex_symbol();
        std::string block_max = get_ro_edge_base(2).get_tex_symbol();
        std::string trees     = get_rw_edge_base(0).get_tex_symbol();

        std::string tex = R"tex(
            Build radix trees

            \begin{align}
            {trees}_{\rm patch} &= RadixTree(\{{block_min}_{\rm patch}\}_{\Omega_i}, \{{block_max}_{\rm patch}\}_{\Omega_i}) \\
            {\Omega_i} &= [0, {sizes}_{\rm patch} )
            \end{align}
        )tex";

        shambase::replace_all(tex, "{sizes}", sizes);
        shambase::replace_all(tex, "{block_min}", block_min);
        shambase::replace_all(tex, "{block_max}", block_max);
        shambase::replace_all(tex, "{trees}", trees);

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::NodeBuildTrees<u64, i64_3>;
