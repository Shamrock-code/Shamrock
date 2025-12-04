// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file FindBlockNeigh.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shammodels/ramses/modules/FindBlockNeigh.hpp"
#include "shamalgs/details/numeric/numeric.hpp"
#include "shammath/AABB.hpp"
#include "shammodels/ramses/modules/details/compute_neigh_graph.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamtree/TreeTraversal.hpp"

namespace shammodels::basegodunov::modules {

    template<class Tvec, class TgridVec, class Tmorton>
    class FindBlockNeigh<Tvec, TgridVec, Tmorton>::AMRBlockFinder {
        public:
        shamrock::tree::ObjectIterator<Tmorton, TgridVec> block_looper;

        sycl::accessor<TgridVec, 1, sycl::access::mode::read, sycl::target::device> acc_block_min;
        sycl::accessor<TgridVec, 1, sycl::access::mode::read, sycl::target::device> acc_block_max;

        TgridVec dir_offset;

        AMRBlockFinder(
            sycl::handler &cgh,
            const RTree &tree,
            sycl::buffer<TgridVec> &buf_block_min,
            sycl::buffer<TgridVec> &buf_block_max,
            TgridVec dir_offset)
            : block_looper(tree, cgh), acc_block_min{buf_block_min, cgh, sycl::read_only},
              acc_block_max{buf_block_max, cgh, sycl::read_only},
              dir_offset(std::move(dir_offset)) {}

        template<class IndexFunctor>
        void for_each_other_index(u32 id_a, IndexFunctor &&fct) const {

            // current block AABB
            shammath::AABB<TgridVec> block_aabb{acc_block_min[id_a], acc_block_max[id_a]};

            // The wanted AABB (the block we look for)
            shammath::AABB<TgridVec> check_aabb{
                block_aabb.lower + dir_offset, block_aabb.upper + dir_offset};

            block_looper.rtree_for(
                [&](u32 node_id, TgridVec bmin, TgridVec bmax) -> bool {
                    return shammath::AABB<TgridVec>{bmin, bmax}
                        .get_intersect(check_aabb)
                        .is_volume_not_null();
                },
                [&](u32 id_b) {
                    bool interact
                        = shammath::AABB<TgridVec>{acc_block_min[id_b], acc_block_max[id_b]}
                              .get_intersect(check_aabb)
                              .is_volume_not_null()
                          && id_b != id_a;

                    if (interact) {
                        fct(id_b);
                    }
                });
        }
    };

    template<class Tvec, class TgridVec, class Tmorton>
    void FindBlockNeigh<Tvec, TgridVec, Tmorton>::_impl_evaluate_internal() {
        auto edges = get_edges();

        edges.spans_block_min.check_sizes(edges.sizes.indexes);
        edges.spans_block_max.check_sizes(edges.sizes.indexes);

        shambase::DistributedData<OrientedAMRGraph> graph;

        edges.trees.trees.for_each([&](u64 id, const RTree &tree) {
            u32 leaf_count          = tree.tree_reduced_morton_codes.tree_leaf_count;
            u32 internal_cell_count = tree.tree_struct.internal_cell_count;
            u32 tot_count           = leaf_count + internal_cell_count;

            OrientedAMRGraph result;

            sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

            sycl::buffer<TgridVec> &tree_bmin
                = shambase::get_check_ref(tree.tree_cell_ranges.buf_pos_min_cell_flt);
            sycl::buffer<TgridVec> &tree_bmax
                = shambase::get_check_ref(tree.tree_cell_ranges.buf_pos_max_cell_flt);

            PatchDataField<TgridVec> &block_min = edges.spans_block_min.get_refs().get(id);
            PatchDataField<TgridVec> &block_max = edges.spans_block_max.get_refs().get(id);

            sycl::buffer<TgridVec> buf_block_min_sycl = block_min.get_buf().copy_to_sycl_buffer();
            sycl::buffer<TgridVec> buf_block_max_sycl = block_max.get_buf().copy_to_sycl_buffer();

            // Build the block graphs for the current patch in all 6 directions
            for (u32 dir = 0; dir < 6; dir++) {

                TgridVec dir_offset = result.offset_check[dir];

                // This build for the current patch in the direction [dir], the graph of its blocks.
                // So for each block in the patch we know its neighbors, and neccessary information
                // to access them
                AMRGraph rslt = details::compute_neigh_graph_deprecated<AMRBlockFinder>(
                    shamsys::instance::get_compute_scheduler_ptr(),
                    edges.sizes.indexes.get(id),
                    tree,
                    buf_block_min_sycl,
                    buf_block_max_sycl,
                    dir_offset);

                shamlog_debug_ln(
                    "AMR Block Graph", "Patch", id, "direction", dir, "link cnt", rslt.link_count);

                std::unique_ptr<AMRGraph> tmp_graph = std::make_unique<AMRGraph>(std::move(rslt));

                result.graph_links[dir] = std::move(tmp_graph);
            }

            graph.add_obj(id, std::move(result));
        });

        edges.block_neigh_graph.graph = std::move(graph);

        // possible unittest
        /*
        one patch with :
        sz = 1 << 4
        base = 4
        model.make_base_grid((0,0,0),(sz,sz,sz),(base*multx,base*multy,base*multz))

        make a grid of 4^3 blocks, which when merge with interface make 6^3 blocks.
        In each direction one slab will have no links, hence the number of links should always be
        6^3 - 6^2 = 180 which we get here on all directions
        */
    }

    template<class Tvec, class TgridVec, class Tmorton>
    std::string FindBlockNeigh<Tvec, TgridVec, Tmorton>::_impl_get_tex() const {

        std::string sizes             = get_ro_edge_base(0).get_tex_symbol();
        std::string block_min         = get_ro_edge_base(1).get_tex_symbol();
        std::string block_max         = get_ro_edge_base(2).get_tex_symbol();
        std::string trees             = get_ro_edge_base(3).get_tex_symbol();
        std::string block_neigh_graph = get_rw_edge_base(0).get_tex_symbol();

        std::string tex = R"tex(
            Find neighbour blocks

            \begin{align}
            {block_neigh_graph} = \text{FindBlockNeigh}({sizes}, {block_min}, {block_max}, {trees})
            \end{align}
        )tex";

        shambase::replace_all(tex, "{sizes}", sizes);
        shambase::replace_all(tex, "{block_min}", block_min);
        shambase::replace_all(tex, "{block_max}", block_max);
        shambase::replace_all(tex, "{trees}", trees);
        shambase::replace_all(tex, "{block_neigh_graph}", block_neigh_graph);

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::FindBlockNeigh<f64_3, i64_3, u64>;
