// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file BlockNeighToCellNeigh.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/exception.hpp"
#include "shamalgs/details/numeric/numeric.hpp"
#include "shambackends/EventList.hpp"
#include "shammath/AABB.hpp"
#include "shammodels/common/amr/AMRBlock.hpp"
#include "shammodels/common/amr/NeighGraph.hpp"
#include "shammodels/ramses/modules/BlockNeighToCellNeigh.hpp"
#include "shammodels/ramses/modules/details/compute_neigh_graph.hpp"
#include "shamrock/amr/AMRCell.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamsys/NodeInstance.hpp"
#include "shamtree/TreeTraversal.hpp"

namespace shammodels::basegodunov::modules {

    // here if we want to find the stencil instead of just the common face, we can just the change
    // lowering mask to the cube offseted in the wanted direction instead of just checking if we
    // have a common surface

    // like on the above case with block Nside 2 we get 12^3 - 12^2 = 1584 link count
    // it is a really good possible test

    template<class Tvec, class TgridVec, class Tmorton>
    template<class AMRBlock>
    class BlockNeighToCellNeigh<Tvec, TgridVec, Tmorton>::AMRLowering {
        public:
        AMRGraph &block_graph;
        sham::DeviceBuffer<TgridVec> &buf_block_min;
        sham::DeviceBuffer<TgridVec> &buf_block_max;
        TgridVec dir_offset;

        AMRLowering(
            AMRGraph &block_graph,
            sham::DeviceBuffer<TgridVec> &buf_block_min,
            sham::DeviceBuffer<TgridVec> &buf_block_max,
            TgridVec dir_offset)
            : block_graph(block_graph), buf_block_min(buf_block_min), buf_block_max(buf_block_max),
              dir_offset(dir_offset) {}

        struct ro_acces;
        inline ro_acces get_read_access(sham::EventList &e) {
            return {
                block_graph.get_read_access(e),
                buf_block_min.get_read_access(e),
                buf_block_max.get_read_access(e),
                dir_offset};
        }

        void complete_event_state(sycl::event &e) {
            block_graph.complete_event_state(e);
            buf_block_min.complete_event_state(e);
            buf_block_max.complete_event_state(e);
        }

        struct ro_acces {

            AMRGraph::ro_access graph_iter;

            const TgridVec *acc_block_min;
            const TgridVec *acc_block_max;

            TgridVec dir_offset;

            template<class IndexFunctor>
            void for_each_other_index_safe(u32 id_a, IndexFunctor &&fct) const {

                const u32 cell_global_id = (u32) id_a;

                const u32 block_id    = cell_global_id / AMRBlock::block_size;
                const u32 cell_loc_id = cell_global_id % AMRBlock::block_size;

                // fetch current block info
                const TgridVec cblock_min = acc_block_min[block_id];
                const TgridVec cblock_max = acc_block_max[block_id];
                const TgridVec delta_cell = (cblock_max - cblock_min) / AMRBlock::Nside;

                // Compute wanted neighbourg cell bounds
                auto get_cell_local_coord = [&]() -> TgridVec {
                    std::array<u32, 3> lcoord_arr = AMRBlock::get_coord(cell_loc_id);
                    return {lcoord_arr[0], lcoord_arr[1], lcoord_arr[2]};
                };

                TgridVec lcoord = get_cell_local_coord();

                shammath::AABB<TgridVec> current_cell_aabb
                    = {cblock_min + lcoord * delta_cell,
                       cblock_min + (lcoord + TgridVec{1, 1, 1}) * delta_cell};

                const shammath::AABB<TgridVec> current_cell_aabb_shifted
                    = {current_cell_aabb.lower + dir_offset, current_cell_aabb.upper + dir_offset};

                auto for_each_possible_blocks = [&](auto &&functor) {
                    functor(block_id);
                    graph_iter.for_each_object_link(block_id, [&](u32 block_b) {
                        functor(block_b);
                    });
                };

                for_each_possible_blocks([&](u32 block_b) {
                    TgridVec block_b_min = acc_block_min[block_b];
                    TgridVec block_b_max = acc_block_max[block_b];

                    const TgridVec delta_cell_b = (block_b_max - block_b_min) / AMRBlock::Nside;

                    for (u32 lx = 0; lx < AMRBlock::Nside; lx++) {
                        for (u32 ly = 0; ly < AMRBlock::Nside; ly++) {
                            for (u32 lz = 0; lz < AMRBlock::Nside; lz++) {

                                shammath::AABB<TgridVec> found_cell
                                    = {TgridVec{block_b_min + TgridVec{lx, ly, lz} * delta_cell_b},
                                       TgridVec{
                                           block_b_min
                                           + TgridVec{lx + 1, ly + 1, lz + 1} * delta_cell_b}};

                                u32 idx = block_b * AMRBlock::block_size
                                          + AMRBlock::get_index({lx, ly, lz});

                                bool overlap = found_cell.get_intersect(current_cell_aabb_shifted)
                                                   .is_volume_not_null()
                                               && id_a != idx;

                                if (overlap) {
                                    fct(idx);
                                }
                            }
                        }
                    }
                });
            }

            template<class IndexFunctor>
            void for_each_other_index_full(u32 id_a, IndexFunctor &&fct) const {

                const u32 cell_global_id = (u32) id_a;

                const u32 block_id    = cell_global_id / AMRBlock::block_size;
                const u32 cell_loc_id = cell_global_id % AMRBlock::block_size;

                // fetch current block info
                const TgridVec cblock_min = acc_block_min[block_id];
                const TgridVec cblock_max = acc_block_max[block_id];
                const TgridVec delta_cell = (cblock_max - cblock_min) / AMRBlock::Nside;

                // Compute wanted neighbourg cell bounds
                auto get_cell_local_coord = [&]() -> TgridVec {
                    std::array<u32, 3> lcoord_arr = AMRBlock::get_coord(cell_loc_id);
                    return {lcoord_arr[0], lcoord_arr[1], lcoord_arr[2]};
                };

                const TgridVec lcoord = get_cell_local_coord();

                const shammath::AABB<TgridVec> current_cell_aabb
                    = {cblock_min + lcoord * delta_cell,
                       cblock_min + (lcoord + TgridVec{1, 1, 1}) * delta_cell};

                const shammath::AABB<TgridVec> current_cell_aabb_shifted
                    = {current_cell_aabb.lower + dir_offset * delta_cell,
                       current_cell_aabb.upper + dir_offset * delta_cell};

                // by default we assume that we are in our block
                // the next function checks if the wanted block is in another blocks
                TgridVec wanted_block_min = cblock_min;
                TgridVec wanted_block_max = cblock_max;
                u32 wanted_block          = block_id;

                graph_iter.for_each_object_link(block_id, [&](u32 block_b) {
                    TgridVec int_wanted_block_min = acc_block_min[block_b];
                    TgridVec int_wanted_block_max = acc_block_max[block_b];

                    bool overlap
                        = shammath::AABB<TgridVec>{int_wanted_block_min, int_wanted_block_max}
                              .get_intersect(current_cell_aabb_shifted)
                              .is_volume_not_null();

                    if (overlap) {
                        wanted_block_min = int_wanted_block_min;
                        wanted_block_max = int_wanted_block_max;
                        wanted_block     = block_b;
                    }
                });

                bool overlap = shammath::AABB<TgridVec>{wanted_block_min, wanted_block_max}
                                   .get_intersect(current_cell_aabb_shifted)
                                   .is_volume_not_null();

                if (!overlap) {
                    return;
                }

                const TgridVec wanted_block_delta_cell
                    = (wanted_block_max - wanted_block_min) / AMRBlock::Nside;

                // at this point the block having the wanted neighbour is in `wanted_block`
                // now we need to find the local coordinates within wanted block of
                // `current_cell_aabb_shifted`, this will give away the indexes

                TgridVec wanted_block_current_cell_shifted
                    = current_cell_aabb_shifted.lower - wanted_block_min;

                std::array<u32, 3> wanted_block_index_range_min
                    = {u32(wanted_block_current_cell_shifted.x() / wanted_block_delta_cell.x()),
                       u32(wanted_block_current_cell_shifted.y() / wanted_block_delta_cell.y()),
                       u32(wanted_block_current_cell_shifted.z() / wanted_block_delta_cell.z())};

                std::array<u32, 3> wanted_block_index_range_max
                    = {u32((wanted_block_current_cell_shifted.x() + delta_cell.x())
                           / wanted_block_delta_cell.x()),
                       u32((wanted_block_current_cell_shifted.y() + delta_cell.y())
                           / wanted_block_delta_cell.y()),
                       u32((wanted_block_current_cell_shifted.z() + delta_cell.z())
                           / wanted_block_delta_cell.z())};

                // now if range size < 1  expand to 1 (case where wanted block is larger)
                if (wanted_block_index_range_max[0] - wanted_block_index_range_min[0] < 1)
                    wanted_block_index_range_max[0] = wanted_block_index_range_min[0] + 1;
                if (wanted_block_index_range_max[1] - wanted_block_index_range_min[1] < 1)
                    wanted_block_index_range_max[1] = wanted_block_index_range_min[1] + 1;
                if (wanted_block_index_range_max[2] - wanted_block_index_range_min[2] < 1)
                    wanted_block_index_range_max[2] = wanted_block_index_range_min[2] + 1;

                for (u32 x = wanted_block_index_range_min[0]; x < wanted_block_index_range_max[0];
                     x++) {
                    for (u32 y = wanted_block_index_range_min[1];
                         y < wanted_block_index_range_max[1];
                         y++) {
                        for (u32 z = wanted_block_index_range_min[2];
                             z < wanted_block_index_range_max[2];
                             z++) {

                            shammath::AABB<TgridVec> found_cell = {
                                TgridVec{wanted_block_min + TgridVec{x, y, z} * delta_cell},
                                TgridVec{
                                    wanted_block_min + TgridVec{x + 1, y + 1, z + 1} * delta_cell}};

                            bool overlap = found_cell.get_intersect(current_cell_aabb_shifted)
                                               .is_volume_not_null();

                            if (overlap) {
                                u32 idx = wanted_block * AMRBlock::block_size
                                          + AMRBlock::get_index({x, y, z});

                                fct(idx);
                            }
                        }
                    }
                }
            }

            template<class IndexFunctor>
            void for_each_other_index(u32 id_a, IndexFunctor &&fct) const {
                // Possible performance regression here, ideally i should fix the full mode for AMR
                // as i expect it to outperform the safe one

                // for_each_other_index_full(id_a, fct);
                for_each_other_index_safe(id_a, fct);
            }
        };
    };

    template<class Tvec, class TgridVec, class Tmorton>
    void BlockNeighToCellNeigh<Tvec, TgridVec, Tmorton>::_impl_evaluate_internal() {
        StackEntry stack_loc{};
        auto edges = get_edges();

        edges.spans_block_min.check_sizes(edges.sizes.indexes);
        edges.spans_block_max.check_sizes(edges.sizes.indexes);

        using AMRBlock = amr::AMRBlock<Tvec, TgridVec, 1>;

        if (block_nside_pow != 1) {
            shambase::throw_unimplemented("block_nside_pow != 1");
        }

        shambase::DistributedData<OrientedAMRGraph> cell_graph_links;

        edges.block_neigh_graph.graph.for_each([&](u64 id,
                                                   const OrientedAMRGraph &oriented_block_graph) {
            OrientedAMRGraph result;

            PatchDataField<TgridVec> &block_min = edges.spans_block_min.get_refs().get(id);
            PatchDataField<TgridVec> &block_max = edges.spans_block_max.get_refs().get(id);

            sham::DeviceBuffer<TgridVec> &buf_block_min = block_min.get_buf();
            sham::DeviceBuffer<TgridVec> &buf_block_max = block_max.get_buf();

            for (u32 dir = 0; dir < 6; dir++) {

                TgridVec dir_offset = result.offset_check[dir];

                AMRGraph &block_graph
                    = shambase::get_check_ref(oriented_block_graph.graph_links[dir]);

                u32 cell_count = (edges.sizes.indexes.get(id)) * AMRBlock::block_size;

                AMRGraph rslt = details::compute_neigh_graph<AMRLowering<AMRBlock>>(
                    shamsys::instance::get_compute_scheduler_ptr(),
                    cell_count,
                    block_graph,
                    buf_block_min,
                    buf_block_max,
                    dir_offset);

                shamlog_debug_ln(
                    "AMR Cell Graph", "Patch", id, "direction", dir, "link cnt", rslt.link_count);

                std::unique_ptr<AMRGraph> tmp_graph = std::make_unique<AMRGraph>(std::move(rslt));

                result.graph_links[dir] = std::move(tmp_graph);
            }

            cell_graph_links.add_obj(id, std::move(result));
        });

        shamlog_debug_ln("[AMR cell graph]", "compute antecedent map");
        cell_graph_links.for_each([&](u64 id, OrientedAMRGraph &oriented_block_graph) {
            auto ptr       = shamsys::instance::get_compute_scheduler_ptr();
            u32 cell_count = (edges.sizes.indexes.get(id)) * AMRBlock::block_size;
            for (u32 dir = 0; dir < 6; dir++) {
                oriented_block_graph.graph_links[dir]->compute_antecedent(ptr);
            }
        });

        edges.cell_neigh_graph.graph = std::move(cell_graph_links);
    }

    template<class Tvec, class TgridVec, class Tmorton>
    std::string BlockNeighToCellNeigh<Tvec, TgridVec, Tmorton>::_impl_get_tex() const {

        std::string sizes             = get_ro_edge_base(0).get_tex_symbol();
        std::string block_min         = get_ro_edge_base(1).get_tex_symbol();
        std::string block_max         = get_ro_edge_base(2).get_tex_symbol();
        std::string block_neigh_graph = get_ro_edge_base(3).get_tex_symbol();
        std::string cell_neigh_graph  = get_rw_edge_base(0).get_tex_symbol();

        std::string tex = R"tex(
            Find neighbour blocks

            \begin{align}
            {cell_neigh_graph} &= \text{BlockNeighToCellNeigh}({block_neigh_graph}, {sizes}, {block_min}, {block_max})
            \end{align}
        )tex";

        shambase::replace_all(tex, "{sizes}", sizes);
        shambase::replace_all(tex, "{block_min}", block_min);
        shambase::replace_all(tex, "{block_max}", block_max);
        shambase::replace_all(tex, "{block_neigh_graph}", block_neigh_graph);
        shambase::replace_all(tex, "{cell_neigh_graph}", cell_neigh_graph);

        return tex;
    }

} // namespace shammodels::basegodunov::modules

template class shammodels::basegodunov::modules::BlockNeighToCellNeigh<f64_3, i64_3, u64>;
