// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file AMRGraphGen.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "AMRGraphGen.hpp"
#include <utility>

/**
 * @brief Create a neighbour graph using a class that will list the ids of the found neighbourgh
 * NeighFindKernel will list the index and that function will run it twice to generate the graph
 *
 * @tparam NeighFindKernel the neigh find kernel
 * @tparam Args arguments that will be forwarded to the kernel
 * @param q the sycl queue
 * @param graph_nodes the number of graph nodes
 * @param args arguments that will be forwarded to the kernel
 * @return shammodels::basegodunov::modules::NeighGraph the neigh graph
 */
template<class NeighFindKernel, class... Args>
shammodels::basegodunov::modules::NeighGraph
compute_neigh_graph(sycl::queue &q, u32 graph_nodes, Args &&...args) {

    // [i] is the number of link for block i in mpdat (last value is 0)
    sycl::buffer<u32> link_counts(graph_nodes + 1);

    // fill buffer with number of link in the block graph
    q.submit([&](sycl::handler &cgh) {
        NeighFindKernel ker(cgh, std::forward<Args>(args)...);
        sycl::accessor link_cnt{link_counts, cgh, sycl::write_only, sycl::no_init};

        shambase::parralel_for(cgh, graph_nodes, "count block graph link", [=](u64 gid) {
            u32 id_a              = (u32) gid;
            u32 block_found_count = 0;

            ker.for_each_other_index(id_a, [&](u32 id_b) {
                block_found_count++;
            });

            link_cnt[id_a] = block_found_count;
        });
    });

    // set the last val to 0 so that the last slot after exclusive scan is the sum
    shamalgs::memory::set_element<u32>(q, link_counts, graph_nodes, 0);

    sycl::buffer<u32> link_cnt_offsets
        = shamalgs::numeric::exclusive_sum(q, link_counts, graph_nodes + 1);

    u32 link_cnt = shamalgs::memory::extract_element(q, link_cnt_offsets, graph_nodes);

    sycl::buffer<u32> ids_links(link_cnt);

    // find the neigh ids
    q.submit([&](sycl::handler &cgh) {
        NeighFindKernel ker(cgh, std::forward<Args>(args)...);
        sycl::accessor cnt_offsets{link_cnt_offsets, cgh, sycl::read_only};
        sycl::accessor links{ids_links, cgh, sycl::write_only, sycl::no_init};

        shambase::parralel_for(cgh, graph_nodes, "get ids block graph link", [=](u64 gid) {
            u32 id_a = (u32) gid;

            u32 next_link_idx = cnt_offsets[id_a];

            ker.for_each_other_index(id_a, [&](u32 id_b) {
                links[next_link_idx] = id_b;
                next_link_idx++;
            });
        });
    });

    using Graph = shammodels::basegodunov::modules::NeighGraph;
    return Graph(Graph{std::move(link_cnt_offsets), std::move(ids_links), link_cnt, graph_nodes});
};

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
// AMR block graph generation
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

template<class Tvec, class TgridVec>
class shammodels::basegodunov::modules::AMRGraphGen<Tvec, TgridVec>::AMRBlockFinder {
    public:
    using RTree = typename Storage::RTree;
    shamrock::tree::ObjectIterator<u_morton, TgridVec> block_looper;

    sycl::accessor<TgridVec, 1, sycl::access::mode::read, sycl::target::device> acc_block_min;
    sycl::accessor<TgridVec, 1, sycl::access::mode::read, sycl::target::device> acc_block_max;

    TgridVec dir_offset;

    AMRBlockFinder(
        sycl::handler &cgh,
        RTree &tree,
        sycl::buffer<TgridVec> &buf_block_min,
        sycl::buffer<TgridVec> &buf_block_max,
        TgridVec dir_offset)
        : block_looper(tree, cgh), acc_block_min{buf_block_min, cgh, sycl::read_only},
          acc_block_max{buf_block_max, cgh, sycl::read_only}, dir_offset(std::move(dir_offset)) {}

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
                bool interact = shammath::AABB<TgridVec>{acc_block_min[id_b], acc_block_max[id_b]}
                                    .get_intersect(check_aabb)
                                    .is_volume_not_null()
                                && id_b != id_a;

                if (interact) {
                    fct(id_b);
                }
            });
    }
};

template<class Tvec, class TgridVec>
shambase::DistributedData<shammodels::basegodunov::modules::OrientedAMRGraph<Tvec, TgridVec>>
shammodels::basegodunov::modules::AMRGraphGen<Tvec, TgridVec>::
    find_AMR_block_graph_links_common_face() {

    using MergedPDat = shamrock::MergedPatchData;
    using RTree      = typename Storage::RTree;

    StackEntry stack_loc{};

    shambase::DistributedData<OrientedAMRGraph> block_graph_links;

    storage.trees.get().for_each([&](u64 id, RTree &tree) {
        u32 leaf_count          = tree.tree_reduced_morton_codes.tree_leaf_count;
        u32 internal_cell_count = tree.tree_struct.internal_cell_count;
        u32 tot_count           = leaf_count + internal_cell_count;

        MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(id);

        OrientedAMRGraph result;

        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

        sycl::buffer<TgridVec> &tree_bmin
            = shambase::get_check_ref(tree.tree_cell_ranges.buf_pos_min_cell_flt);
        sycl::buffer<TgridVec> &tree_bmax
            = shambase::get_check_ref(tree.tree_cell_ranges.buf_pos_max_cell_flt);

        sham::DeviceBuffer<TgridVec> &buf_block_min = mpdat.pdat.get_field_buf_ref<TgridVec>(0);
        sham::DeviceBuffer<TgridVec> &buf_block_max = mpdat.pdat.get_field_buf_ref<TgridVec>(1);

        sycl::buffer<TgridVec> buf_block_min_sycl = buf_block_min.copy_to_sycl_buffer();
        sycl::buffer<TgridVec> buf_block_max_sycl = buf_block_max.copy_to_sycl_buffer();

        for (u32 dir = 0; dir < 6; dir++) {

            TgridVec dir_offset = result.offset_check[dir];

            AMRGraph rslt = compute_neigh_graph<AMRBlockFinder>(
                q.q,
                mpdat.total_elements,
                tree,
                buf_block_min_sycl,
                buf_block_max_sycl,
                dir_offset);

            logger::debug_ln(
                "AMR Block Graph", "Patch", id, "direction", dir, "link cnt", rslt.link_count);

            std::unique_ptr<AMRGraph> tmp_graph = std::make_unique<AMRGraph>(std::move(rslt));

            result.graph_links[dir] = std::move(tmp_graph);
        }

        block_graph_links.add_obj(id, std::move(result));
    });

    return block_graph_links;

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

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
// Lowering from block graph to cell graph
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

// here if we want to find the stencil instead of just the common face, we can just the change
// lowering mask to the cube offseted in the wanted direction instead of just checking if we have a
// common surface

// like on the above case with block Nside 2 we get 12^3 - 12^2 = 1584 link count
// it is a really good possible test

template<class Tvec, class TgridVec>
class shammodels::basegodunov::modules::AMRGraphGen<Tvec, TgridVec>::AMRLowering {
    public:
    AMRGraphLinkiterator graph_iter;

    sycl::accessor<TgridVec, 1, sycl::access::mode::read, sycl::target::device> acc_block_min;
    sycl::accessor<TgridVec, 1, sycl::access::mode::read, sycl::target::device> acc_block_max;

    TgridVec dir_offset;

    AMRLowering(
        sycl::handler &cgh,
        AMRGraph &block_graph,
        sycl::buffer<TgridVec> &buf_block_min,
        sycl::buffer<TgridVec> &buf_block_max,
        TgridVec dir_offset)
        : graph_iter{block_graph, cgh}, acc_block_min{buf_block_min, cgh, sycl::read_only},
          acc_block_max{buf_block_max, cgh, sycl::read_only}, dir_offset(std::move(dir_offset)) {}

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
                                   block_b_min + TgridVec{lx + 1, ly + 1, lz + 1} * delta_cell_b}};

                        u32 idx
                            = block_b * AMRBlock::block_size + AMRBlock::get_index({lx, ly, lz});

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

            bool overlap = shammath::AABB<TgridVec>{int_wanted_block_min, int_wanted_block_max}
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

        for (u32 x = wanted_block_index_range_min[0]; x < wanted_block_index_range_max[0]; x++) {
            for (u32 y = wanted_block_index_range_min[1]; y < wanted_block_index_range_max[1];
                 y++) {
                for (u32 z = wanted_block_index_range_min[2]; z < wanted_block_index_range_max[2];
                     z++) {

                    shammath::AABB<TgridVec> found_cell
                        = {TgridVec{wanted_block_min + TgridVec{x, y, z} * delta_cell},
                           TgridVec{wanted_block_min + TgridVec{x + 1, y + 1, z + 1} * delta_cell}};

                    bool overlap
                        = found_cell.get_intersect(current_cell_aabb_shifted).is_volume_not_null();

                    if (overlap) {
                        u32 idx
                            = wanted_block * AMRBlock::block_size + AMRBlock::get_index({x, y, z});

                        fct(idx);
                    }
                }
            }
        }
    }

    template<class IndexFunctor>
    void for_each_other_index(u32 id_a, IndexFunctor &&fct) const {
        // Possible performance regression here, ideally i should fix the full mode for AMR as i
        // expect it to outperform the safe one

        // for_each_other_index_full(id_a, fct);
        for_each_other_index_safe(id_a, fct);
    }
};

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::AMRGraphGen<Tvec, TgridVec>::
    lower_AMR_block_graph_to_cell_common_face_graph(
        shambase::DistributedData<OrientedAMRGraph> &oriented_blocks_graph_links) {

    StackEntry stack_loc{};

    using MergedPDat = shamrock::MergedPatchData;
    using RTree      = typename Storage::RTree;

    shambase::DistributedData<OrientedAMRGraph> cell_graph_links;

    oriented_blocks_graph_links.for_each([&](u64 id, OrientedAMRGraph &oriented_block_graph) {
        MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(id);

        OrientedAMRGraph result;

        sycl::queue &q = shamsys::instance::get_compute_queue();

        sham::DeviceBuffer<TgridVec> &buf_block_min = mpdat.pdat.get_field_buf_ref<TgridVec>(0);
        sham::DeviceBuffer<TgridVec> &buf_block_max = mpdat.pdat.get_field_buf_ref<TgridVec>(1);

        sycl::buffer<TgridVec> buf_block_min_sycl = buf_block_min.copy_to_sycl_buffer();
        sycl::buffer<TgridVec> buf_block_max_sycl = buf_block_max.copy_to_sycl_buffer();

        for (u32 dir = 0; dir < 6; dir++) {

            TgridVec dir_offset = result.offset_check[dir];

            AMRGraph &block_graph = shambase::get_check_ref(oriented_block_graph.graph_links[dir]);

            u32 cell_count = (mpdat.total_elements) * AMRBlock::block_size;

            AMRGraph rslt = compute_neigh_graph<AMRLowering>(
                q, cell_count, block_graph, buf_block_min_sycl, buf_block_max_sycl, dir_offset);

            logger::debug_ln(
                "AMR Cell Graph", "Patch", id, "direction", dir, "link cnt", rslt.link_count);

            std::unique_ptr<AMRGraph> tmp_graph = std::make_unique<AMRGraph>(std::move(rslt));

            result.graph_links[dir] = std::move(tmp_graph);
        }

        cell_graph_links.add_obj(id, std::move(result));
    });

    logger::debug_ln("[AMR cell graph]", "compute antecedent map");
    cell_graph_links.for_each([&](u64 id, OrientedAMRGraph &oriented_block_graph) {
        sycl::queue &q    = shamsys::instance::get_compute_queue();
        MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(id);
        u32 cell_count    = (mpdat.total_elements) * AMRBlock::block_size;
        for (u32 dir = 0; dir < 6; dir++) {
            oriented_block_graph.graph_links[dir]->compute_antecedent(q);
        }
    });

    storage.cell_link_graph.set(std::move(cell_graph_links));
}

template class shammodels::basegodunov::modules::AMRGraphGen<f64_3, i64_3>;
