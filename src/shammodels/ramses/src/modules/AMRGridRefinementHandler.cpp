// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file AMRGridRefinementHandler.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/DistributedData.hpp"
#include "shambase/aliases_int.hpp"
#include "shambase/memory.hpp"
#include "shamalgs/details/algorithm/algorithm.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/DeviceQueue.hpp"
#include "shambackends/EventList.hpp"
#include "shamcomm/logs.hpp"
#include "shammodels/ramses/modules/AMRGridRefinementHandler.hpp"
#include "shammodels/ramses/modules/AMRSortBlocks.hpp"
#include "shamsys/NodeInstance.hpp"
#include <shambackends/sycl.hpp>
#include <algorithm>
#include <stdexcept>
#include <utility>

template<class Tvec, class TgridVec>
template<class UserAcc, class... T>
void shammodels::basegodunov::modules::AMRGridRefinementHandler<Tvec, TgridVec>::
    gen_refine_block_changes(
        shambase::DistributedData<sham::DeviceBuffer<u32>> &refine_flags,
        shambase::DistributedData<sham::DeviceBuffer<u32>> &derefine_flags,
        T &&...args) {

    using namespace shamrock::patch;

    u64 tot_refine   = 0;
    u64 tot_derefine = 0;

    sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
    auto dev_sched       = shamsys::instance::get_compute_scheduler_ptr();

    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
        u64 id_patch = cur_p.id_patch;

        // create the refine and derefine flags buffers
        u32 obj_cnt = pdat.get_obj_cnt();

        sham::DeviceBuffer<u32> refine_flag(obj_cnt, dev_sched);
        sham::DeviceBuffer<u32> derefine_flag(obj_cnt, dev_sched);

        {
            sham::EventList depends_list;

            UserAcc uacc(depends_list, id_patch, cur_p, pdat, args...);

            auto refine_acc   = refine_flag.get_write_access(depends_list);
            auto derefine_acc = derefine_flag.get_write_access(depends_list);

            // fill in the flags
            auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                cgh.parallel_for(sycl::range<1>(obj_cnt), [=](sycl::item<1> gid) {
                    bool flag_refine   = false;
                    bool flag_derefine = false;
                    uacc.refine_criterion(gid.get_linear_id(), uacc, flag_refine, flag_derefine);

                    // This is just a safe guard to avoid this nonsensicall case
                    if (flag_refine && flag_derefine) {
                        flag_derefine = false;
                    }

                    refine_acc[gid]   = (flag_refine) ? 1 : 0;
                    derefine_acc[gid] = (flag_derefine) ? 1 : 0;
                });
            });

            sham::EventList resulting_events;
            resulting_events.add_event(e);

            refine_flag.complete_event_state(resulting_events);
            derefine_flag.complete_event_state(resulting_events);

            uacc.finalize(resulting_events, id_patch, cur_p, pdat, args...);
        }

        refine_flags.add_obj(id_patch, std::move(refine_flag));
        derefine_flags.add_obj(id_patch, std::move(derefine_flag));
    });
}

/**
 * @brief check and enforce 2:1 rule for refinement
 * @tparam Tvec
 * @tparam TgridVec
 * @param refine_list refinement mask
 * @param refine_idx_list refinement map
 */
template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::AMRGridRefinementHandler<Tvec, TgridVec>::
    enforce_two_to_one_for_refinement(
        shambase::DistributedData<sham::DeviceBuffer<u32>> &refine_flags,
        shambase::DistributedData<sham::DeviceBuffer<u32>> &refine_list) {

    using namespace shamrock::patch;
    using AMRGraph             = shammodels::basegodunov::modules::AMRGraph;
    using Direction_           = shammodels::basegodunov::modules::Direction;
    using AMRGraphLinkiterator = shammodels::basegodunov::modules::AMRGraph::ro_access;
    using TgridUint = typename std::make_unsigned<shambase::VecComponent<TgridVec>>::type;

    u64 tot_refine       = 0;
    sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
    auto dev_sched       = shamsys::instance::get_compute_scheduler_ptr();

    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
        u64 id_patch = cur_p.id_patch;

        sham::DeviceBuffer<u32> &refine_flags_buf = refine_flags.get(id_patch);
        u32 obj_cnt                               = pdat.get_obj_cnt();

        // blocks graph in each direction for the current patch
        AMRGraph &block_graph_neighs_xp = shambase::get_check_ref(storage.block_graph_edge)
                                              .get_refs_dir(Direction_::xp)
                                              .get(id_patch);
        AMRGraph &block_graph_neighs_xm = shambase::get_check_ref(storage.block_graph_edge)
                                              .get_refs_dir(Direction_::xm)
                                              .get(id_patch);
        AMRGraph &block_graph_neighs_yp = shambase::get_check_ref(storage.block_graph_edge)
                                              .get_refs_dir(Direction_::yp)
                                              .get(id_patch);
        AMRGraph &block_graph_neighs_ym = shambase::get_check_ref(storage.block_graph_edge)
                                              .get_refs_dir(Direction_::ym)
                                              .get(id_patch);
        AMRGraph &block_graph_neighs_zp = shambase::get_check_ref(storage.block_graph_edge)
                                              .get_refs_dir(Direction_::zp)
                                              .get(id_patch);
        AMRGraph &block_graph_neighs_zm = shambase::get_check_ref(storage.block_graph_edge)
                                              .get_refs_dir(Direction_::zm)
                                              .get(id_patch);

        // get levels in the current patch
        sham::DeviceBuffer<TgridUint> &buf_amr_block_levels
            = shambase::get_check_ref(storage.amr_block_levels).get_buf(id_patch);

        // propagate refinement until stability
        for (auto pass = 0; pass < 3; pass++) {

            sham::EventList depend_list;
            AMRGraphLinkiterator block_graph_xp
                = block_graph_neighs_xp.get_read_access(depend_list);
            AMRGraphLinkiterator block_graph_xm
                = block_graph_neighs_xm.get_read_access(depend_list);
            AMRGraphLinkiterator block_graph_yp
                = block_graph_neighs_yp.get_read_access(depend_list);
            AMRGraphLinkiterator block_graph_ym
                = block_graph_neighs_ym.get_read_access(depend_list);
            AMRGraphLinkiterator block_graph_zp
                = block_graph_neighs_zp.get_read_access(depend_list);
            AMRGraphLinkiterator block_graph_zm
                = block_graph_neighs_zm.get_read_access(depend_list);
            auto acc_amr_levels = buf_amr_block_levels.get_read_access(depend_list);
            auto acc_ref_flags  = refine_flags_buf.get_write_access(depend_list);

            auto e_all_dir = q.submit(depend_list, [&](sycl::handler &cgh) {
                cgh.parallel_for(sycl::range<1>(obj_cnt), [=](sycl::item<1> gid) {
                    u32 block_id = gid.get_linear_id();

                    // get refinement flag and amr level of the current block
                    u32 cur_ref_flag     = acc_ref_flags[block_id];
                    auto cur_block_level = acc_amr_levels[block_id];

                    if (cur_ref_flag) {
                        auto enforce_2_to_1_rule = [&](u32 neigh_block_id) {
                            if (0 <= neigh_block_id && neigh_block_id < obj_cnt) {
                                auto neigh_block_level = acc_amr_levels[neigh_block_id];
                                if (cur_block_level > neigh_block_level) {
                                    sycl::atomic_ref<
                                        u32,
                                        sycl::memory_order::relaxed,
                                        sycl::memory_scope::device>
                                        atomic_flag(acc_ref_flags[neigh_block_id]);
                                    atomic_flag.store(1);
                                }
                            }
                        };

                        block_graph_xp.for_each_object_link(block_id, enforce_2_to_1_rule);
                        block_graph_xm.for_each_object_link(block_id, enforce_2_to_1_rule);
                        block_graph_yp.for_each_object_link(block_id, enforce_2_to_1_rule);
                        block_graph_ym.for_each_object_link(block_id, enforce_2_to_1_rule);
                        block_graph_zp.for_each_object_link(block_id, enforce_2_to_1_rule);
                        block_graph_zm.for_each_object_link(block_id, enforce_2_to_1_rule);
                    }
                });
            });
            block_graph_neighs_xp.complete_event_state(e_all_dir);
            block_graph_neighs_xm.complete_event_state(e_all_dir);
            block_graph_neighs_yp.complete_event_state(e_all_dir);
            block_graph_neighs_ym.complete_event_state(e_all_dir);
            block_graph_neighs_zp.complete_event_state(e_all_dir);
            block_graph_neighs_zm.complete_event_state(e_all_dir);
            buf_amr_block_levels.complete_event_state(e_all_dir);
            refine_flags_buf.complete_event_state(e_all_dir);
        }
        ////////////////////////////////////////////////////////////////////////////////
        // refinement
        ////////////////////////////////////////////////////////////////////////////////

        // perform stream compactions on the refinement flags
        auto buf_refine = shamalgs::numeric::stream_compact(dev_sched, refine_flags_buf, obj_cnt);
        shamlog_debug_ln(
            "AMRGrid", "patch ", id_patch, buf_refine.get_size(), "marked for refinement + 2:1");
        tot_refine += buf_refine.get_size();
        // add the results to the map
        refine_list.add_obj(id_patch, std::move(buf_refine));
    });
    logger::info_ln("AMRGrid", "on this process", tot_refine, "blocks will be refined");
}

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::AMRGridRefinementHandler<Tvec, TgridVec>::
    check_geometrical_validity_for_derefinement(
        shambase::DistributedData<sham::DeviceBuffer<u32>> &derefine_flags,
        shambase::DistributedData<sham::DeviceBuffer<u32>> &refine_flags) {

    using namespace shamrock::patch;
    using TgridUint      = typename std::make_unsigned<shambase::VecComponent<TgridVec>>::type;
    sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
    auto dev_sched       = shamsys::instance::get_compute_scheduler_ptr();

    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
        u64 id_patch = cur_p.id_patch;

        sham::DeviceBuffer<u32> &derefine_flag_buf = derefine_flags.get(id_patch);
        sham::DeviceBuffer<u32> &refine_flag_buf   = refine_flags.get(id_patch);

        u32 obj_cnt = pdat.get_obj_cnt();

        // get the current buffer of block levels in the current patch
        sham::DeviceBuffer<TgridUint> &buf_amr_block_levels
            = shambase::get_check_ref(storage.amr_block_levels).get_buf(id_patch);

        sham::DeviceBuffer<TgridVec> &buf_cell_min = pdat.get_field_buf_ref<TgridVec>(0);
        sham::DeviceBuffer<TgridVec> &buf_cell_max = pdat.get_field_buf_ref<TgridVec>(1);

        sham::EventList depends_list;
        auto acc_min        = buf_cell_min.get_read_access(depends_list);
        auto acc_max        = buf_cell_max.get_read_access(depends_list);
        auto acc_amr_levels = buf_amr_block_levels.get_read_access(depends_list);
        auto acc_merge_flag = derefine_flag_buf.get_write_access(depends_list);
        auto acc_ref_flag   = refine_flag_buf.get_write_access(depends_list);

        auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::range<1>(obj_cnt), [=](sycl::item<1> gid) {
                u32 id = gid.get_linear_id();

                std::array<BlockCoord, split_count> blocks;
                bool do_merge       = true;
                bool all_same_level = true;

                // This avoid the case where we are in the last block of the buffer to
                // avoid the out-of-bound read
                if (id + split_count <= obj_cnt) {
                    bool all_want_to_merge = true;

                    /** get the local coordinate of a given block
                     */
                    auto get_coord = [](u32 i) -> std::array<u32, dim> {
                        constexpr u32 NsideBlockPow = 1;
                        constexpr u32 Nside         = 1U << NsideBlockPow;
                        constexpr u32 side_size     = Nside;
                        constexpr u32 block_size    = shambase::pow_constexpr<dim>(Nside);

                        if constexpr (dim == 3) {
                            const u32 tmp = i >> NsideBlockPow;
                            // This line is why derefinement never happens with th old
                            // implementation. return {i % Nside, (tmp) % Nside, (tmp ) >>
                            // NsideBlockPow};

                            // the following fix is what we should do. Is it only here should fix
                            // this ?
                            return {(tmp) >> NsideBlockPow, (tmp) % Nside, i % Nside};
                        }
                    };

                    auto get_split
                        = [=](BlockCoord target_block) -> std::array<BlockCoord, split_count> {
                        std::array<BlockCoord, split_count> ret;
                        auto bmin                   = target_block.bmin;
                        auto bmax                   = target_block.bmax;
                        auto split                  = bmin + (bmax - bmin) / 2;
                        std::array<TgridVec, 3> szs = {bmin, split, bmax};
                        for (u32 i = 0; i < split_count; i++) {
                            auto [lx, ly, lz] = get_coord(i);

                            ret[i].bmin = TgridVec{szs[lx].x(), szs[ly].y(), szs[lz].z()};
                            ret[i].bmax
                                = TgridVec{szs[lx + 1].x(), szs[ly + 1].y(), szs[lz + 1].z()};
                        }

                        return ret;
                    };

                    // check if all siblings want to merge and if the supposed sibling are at the
                    // same level (for ghost blocks)
                    for (u32 b_lid = 0; b_lid < split_count; b_lid++) {
                        blocks[b_lid]     = BlockCoord{acc_min[id + b_lid], acc_max[id + b_lid]};
                        all_want_to_merge = all_want_to_merge && acc_merge_flag[id + b_lid];
                        all_same_level
                            = all_same_level && (acc_amr_levels[id] == acc_amr_levels[id + b_lid]);
                    }

                    // check if the merge is geometrical valid
                    BlockCoord merged                            = BlockCoord::get_merge(blocks);
                    std::array<BlockCoord, split_count> splitted = get_split(merged);
                    for (u32 lid = 0; lid < split_count; lid++) {
                        do_merge = do_merge && sham::equals(blocks[lid].bmin, splitted[lid].bmin)
                                   && sham::equals(blocks[lid].bmax, splitted[lid].bmax);
                    }

                    // are all siblings want to merge,  mergeable and at same level ?
                    do_merge = do_merge && all_want_to_merge && all_same_level;

                    // if a block is marked both for refinement and derefinement then the last is
                    // aborted.
                    if (acc_ref_flag[id] && do_merge) {
                        do_merge = false;
                    }

                } else {
                    do_merge = false;
                }

                acc_merge_flag[id] = do_merge;
            });
        });
        buf_cell_min.complete_event_state(e);
        buf_cell_max.complete_event_state(e);
        buf_amr_block_levels.complete_event_state(e);
        refine_flag_buf.complete_event_state(e);
        derefine_flag_buf.complete_event_state(e);

        auto buf_derefine
            = shamalgs::numeric::stream_compact(dev_sched, derefine_flag_buf, obj_cnt);

        logger::raw_ln(
            " Count block's flag for derefinement [After geometry validity check and before 2:1 "
            "check] "
            "\t : ",
            "patch id \t",
            id_patch,
            "\t",
            buf_derefine.get_size(),
            "\n");
    });
}

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::AMRGridRefinementHandler<Tvec, TgridVec>::
    enforce_two_to_one_for_derefinement(
        shambase::DistributedData<sham::DeviceBuffer<u32>> &derefine_flags,
        shambase::DistributedData<sham::DeviceBuffer<u32>> &derefine_list) {

    using namespace shamrock::patch;
    using AMRGraph             = shammodels::basegodunov::modules::AMRGraph;
    using Direction_           = shammodels::basegodunov::modules::Direction;
    using AMRGraphLinkiterator = shammodels::basegodunov::modules::AMRGraph::ro_access;
    using TgridUint = typename std::make_unsigned<shambase::VecComponent<TgridVec>>::type;

    u64 tot_derefine     = 0;
    sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
    auto dev_sched       = shamsys::instance::get_compute_scheduler_ptr();

    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
        u64 id_patch = cur_p.id_patch;

        sham::DeviceBuffer<u32> &derefine_flags_buf = derefine_flags.get(id_patch);
        u32 obj_cnt                                 = pdat.get_obj_cnt();

        // blocks graph in each direction for the current patch
        AMRGraph &block_graph_neighs_xp = shambase::get_check_ref(storage.block_graph_edge)
                                              .get_refs_dir(Direction_::xp)
                                              .get(id_patch);
        AMRGraph &block_graph_neighs_xm = shambase::get_check_ref(storage.block_graph_edge)
                                              .get_refs_dir(Direction_::xm)
                                              .get(id_patch);
        AMRGraph &block_graph_neighs_yp = shambase::get_check_ref(storage.block_graph_edge)
                                              .get_refs_dir(Direction_::yp)
                                              .get(id_patch);
        AMRGraph &block_graph_neighs_ym = shambase::get_check_ref(storage.block_graph_edge)
                                              .get_refs_dir(Direction_::ym)
                                              .get(id_patch);
        AMRGraph &block_graph_neighs_zp = shambase::get_check_ref(storage.block_graph_edge)
                                              .get_refs_dir(Direction_::zp)
                                              .get(id_patch);
        AMRGraph &block_graph_neighs_zm = shambase::get_check_ref(storage.block_graph_edge)
                                              .get_refs_dir(Direction_::zm)
                                              .get(id_patch);

        // get the current buffer of block levels in the current patch
        sham::DeviceBuffer<TgridUint> &buf_amr_block_levels
            = shambase::get_check_ref(storage.amr_block_levels).get_buf(id_patch);

        sham::EventList depend_list;
        auto acc_amr_levels                 = buf_amr_block_levels.get_read_access(depend_list);
        AMRGraphLinkiterator block_graph_xp = block_graph_neighs_xp.get_read_access(depend_list);
        AMRGraphLinkiterator block_graph_xm = block_graph_neighs_xm.get_read_access(depend_list);
        AMRGraphLinkiterator block_graph_yp = block_graph_neighs_yp.get_read_access(depend_list);
        AMRGraphLinkiterator block_graph_ym = block_graph_neighs_ym.get_read_access(depend_list);
        AMRGraphLinkiterator block_graph_zp = block_graph_neighs_zp.get_read_access(depend_list);
        AMRGraphLinkiterator block_graph_zm = block_graph_neighs_zm.get_read_access(depend_list);

        auto acc_deref_flags = derefine_flags_buf.get_write_access(depend_list);

        auto e = q.submit(depend_list, [&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::range<1>(obj_cnt), [=](sycl::item<1> gid) {
                auto lid           = gid.get_linear_id();
                auto cur_block_lev = acc_amr_levels[lid];

                auto check_2_to_1_rule = [&](u32 nid) {
                    if (nid < obj_cnt && (acc_amr_levels[lid] < acc_amr_levels[nid])) {
                        // // No atomic operation is required. We only need to check whether the
                        // // rule is violated for at least one neighbor.
                        acc_deref_flags[lid] = 0;
                    }
                };

                if (acc_deref_flags[lid]) {

                    // // The 2:1 rule must be enforced at the parent level. However, since we do
                    // //  not have direct access to the neighbors of the parent,
                    // //  we instead check the rule against all neighbors of the eight siblings,
                    // // which collectively correspond to the neighbors of the parent block.
                    for (auto i = 0; i < AMRBlock::block_size; i++) {
                        block_graph_xp.for_each_object_link(lid + i, check_2_to_1_rule);
                        block_graph_xm.for_each_object_link(lid + i, check_2_to_1_rule);
                        block_graph_yp.for_each_object_link(lid + i, check_2_to_1_rule);
                        block_graph_ym.for_each_object_link(lid + i, check_2_to_1_rule);
                        block_graph_zp.for_each_object_link(lid + i, check_2_to_1_rule);
                        block_graph_zm.for_each_object_link(lid + i, check_2_to_1_rule);
                    }
                }
            });
        });

        block_graph_neighs_xp.complete_event_state(e);
        block_graph_neighs_xm.complete_event_state(e);
        block_graph_neighs_yp.complete_event_state(e);
        block_graph_neighs_ym.complete_event_state(e);
        block_graph_neighs_zp.complete_event_state(e);
        block_graph_neighs_zm.complete_event_state(e);
        buf_amr_block_levels.complete_event_state(e);
        derefine_flags_buf.complete_event_state(e);

        ////////////////////////////////////////////////////////////////////////////////
        // refinement
        ////////////////////////////////////////////////////////////////////////////////

        // perform stream compactions on the refinement flags
        auto buf_derefine
            = shamalgs::numeric::stream_compact(dev_sched, derefine_flags_buf, obj_cnt);

        logger::raw_ln(
            " Count block's flag for derefinement [After geometry validity check and after 2:1 "
            "check] \t : ",
            buf_derefine.get_size(),
            "\n");
        tot_derefine += buf_derefine.get_size();
        // add the results to the map
        derefine_list.add_obj(id_patch, std::move(buf_derefine));
    });
    logger::info_ln(
        "AMRGrid", "on this process", tot_derefine * split_count, "blocks will be derefined");
}

template<class Tvec, class TgridVec>
template<class UserAcc>
bool shammodels::basegodunov::modules::AMRGridRefinementHandler<Tvec, TgridVec>::
    internal_refine_grid(shambase::DistributedData<sham::DeviceBuffer<u32>> &&refine_list) {

    using namespace shamrock::patch;

    u64 sum_block_count = 0;

    bool new_cell_were_added = false;

    scheduler().for_each_patch_data([&](u64 id_patch, Patch cur_p, PatchDataLayer &pdat) {
        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

        u32 old_obj_cnt = pdat.get_obj_cnt();

        sham::DeviceBuffer<u32> &refine_flags = refine_list.get(id_patch);

        if (refine_flags.get_size() > 0) {

            // alloc memory for the new blocks to be created
            pdat.expand(refine_flags.get_size() * (split_count - 1));

            sham::DeviceBuffer<TgridVec> &buf_cell_min = pdat.get_field_buf_ref<TgridVec>(0);
            sham::DeviceBuffer<TgridVec> &buf_cell_max = pdat.get_field_buf_ref<TgridVec>(1);

            sham::EventList depends_list;
            auto block_bound_low  = buf_cell_min.get_write_access(depends_list);
            auto block_bound_high = buf_cell_max.get_write_access(depends_list);
            UserAcc uacc(depends_list, pdat);
            auto index_to_ref = refine_flags.get_read_access(depends_list);

            // Refine the block (set the positions) and fill the corresponding fields
            auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                u32 start_index_push = old_obj_cnt;

                constexpr u32 new_splits = split_count - 1;

                cgh.parallel_for(sycl::range<1>(refine_flags.get_size()), [=](sycl::item<1> gid) {
                    u32 tid = gid.get_linear_id();

                    u32 idx_to_refine = index_to_ref[tid];

                    // gen splits coordinates
                    BlockCoord cur_block{
                        block_bound_low[idx_to_refine], block_bound_high[idx_to_refine]};

                    std::array<BlockCoord, split_count> block_coords
                        = BlockCoord::get_split(cur_block.bmin, cur_block.bmax);

                    // generate index for the refined blocks
                    std::array<u32, split_count> blocks_ids;
                    blocks_ids[0] = idx_to_refine;

                    // generate index for the new blocks (the current index is reused for the first
                    // new block, the others are pushed at the end of the patchdata)
#pragma unroll
                    for (u32 pid = 0; pid < new_splits; pid++) {
                        blocks_ids[pid + 1] = start_index_push + tid * new_splits + pid;
                    }

                    // write coordinates

#pragma unroll
                    for (u32 pid = 0; pid < split_count; pid++) {
                        block_bound_low[blocks_ids[pid]]  = block_coords[pid].bmin;
                        block_bound_high[blocks_ids[pid]] = block_coords[pid].bmax;
                    }

                    // user lambda to fill the fields
                    uacc.apply_refine(idx_to_refine, cur_block, blocks_ids, block_coords, uacc);
                });
            });

            sham::EventList resulting_events{e};

            buf_cell_min.complete_event_state(resulting_events);
            buf_cell_max.complete_event_state(resulting_events);

            uacc.finalize(resulting_events, pdat);

            refine_flags.complete_event_state(resulting_events);
        }

        sum_block_count += pdat.get_obj_cnt();
        new_cell_were_added = new_cell_were_added || refine_flags.get_size() > 0;
    });

    logger::info_ln("AMRGrid", "process block count =", sum_block_count);

    return new_cell_were_added;
}

template<class Tvec, class TgridVec>
template<class UserAcc>
bool shammodels::basegodunov::modules::AMRGridRefinementHandler<Tvec, TgridVec>::
    internal_derefine_grid(shambase::DistributedData<sham::DeviceBuffer<u32>> &&derefine_list) {

    using namespace shamrock::patch;

    bool cell_were_removed = false;

    sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
    auto dev_sched       = shamsys::instance::get_compute_scheduler_ptr();

    scheduler().for_each_patch_data([&](u64 id_patch, Patch cur_p, PatchDataLayer &pdat) {
        u32 old_obj_cnt = pdat.get_obj_cnt();

        sham::DeviceBuffer<u32> &derefine_flags = derefine_list.get(id_patch);

        if (derefine_flags.get_size() > 0) {

            // init flag table
            sham::DeviceBuffer<u32> keep_block_flag(old_obj_cnt, dev_sched);
            keep_block_flag.fill(1);

            sham::DeviceBuffer<TgridVec> &buf_cell_min = pdat.get_field_buf_ref<TgridVec>(0);
            sham::DeviceBuffer<TgridVec> &buf_cell_max = pdat.get_field_buf_ref<TgridVec>(1);

            sham::EventList depends_list;
            auto block_bound_low  = buf_cell_min.get_write_access(depends_list);
            auto block_bound_high = buf_cell_max.get_write_access(depends_list);
            UserAcc uacc(depends_list, pdat);
            auto index_to_deref = derefine_flags.get_read_access(depends_list);
            auto flag_keep      = keep_block_flag.get_write_access(depends_list);

            // edit block content + make flag of blocks to keep
            auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                cgh.parallel_for(sycl::range<1>(derefine_flags.get_size()), [=](sycl::item<1> gid) {
                    u32 tid = gid.get_linear_id();

                    u32 idx_to_derefine = index_to_deref[gid];

                    // compute old block indexes
                    std::array<u32, split_count> old_indexes;
#pragma unroll
                    for (u32 pid = 0; pid < split_count; pid++) {
                        old_indexes[pid] = idx_to_derefine + pid;
                    }

                    // load block coords
                    std::array<BlockCoord, split_count> block_coords;
#pragma unroll
                    for (u32 pid = 0; pid < split_count; pid++) {
                        block_coords[pid] = BlockCoord{
                            block_bound_low[old_indexes[pid]], block_bound_high[old_indexes[pid]]};
                    }

                    // make new block coord
                    BlockCoord merged_block_coord = BlockCoord::get_merge(block_coords);

                    // write new coord
                    block_bound_low[idx_to_derefine]  = merged_block_coord.bmin;
                    block_bound_high[idx_to_derefine] = merged_block_coord.bmax;

// flag the old blocks for removal
#pragma unroll
                    for (u32 pid = 1; pid < split_count; pid++) {
                        flag_keep[idx_to_derefine + pid] = 0;
                    }

                    // user lambda to fill the fields
                    uacc.apply_derefine(
                        old_indexes, block_coords, idx_to_derefine, merged_block_coord, uacc);
                });
            });

            sham::EventList resulting_events{e};

            buf_cell_min.complete_event_state(resulting_events);
            buf_cell_max.complete_event_state(resulting_events);

            uacc.finalize(resulting_events, pdat);

            keep_block_flag.complete_event_state(resulting_events);
            derefine_flags.complete_event_state(resulting_events);

            // stream compact the flags
            auto buf_keep
                = shamalgs::numeric::stream_compact(dev_sched, keep_block_flag, old_obj_cnt);

            shamlog_debug_ln(
                "AMR Grid",
                "patch",
                id_patch,
                "derefine block count ",
                old_obj_cnt,
                "->",
                buf_keep.get_size());

            if (buf_keep.get_size() == 0) {
                throw std::runtime_error("buf keep must contain something at this point");
            }

            // remap pdat according to stream compact
            pdat.index_remap_resize(buf_keep, buf_keep.get_size());

            cell_were_removed = cell_were_removed || derefine_flags.get_size() > 0;
        }
    });

    return cell_were_removed;
}

template<class Tvec, class TgridVec>
template<class UserAccCrit, class UserAccSplit, class UserAccMerge>
void shammodels::basegodunov::modules::AMRGridRefinementHandler<Tvec, TgridVec>::
    internal_update_refinement() {

    // Ensure that the blocks are sorted before refinement
    AMRSortBlocks block_sorter(context, solver_config, storage);
    block_sorter.reorder_amr_blocks();

    // get refine and derefine list
    shambase::DistributedData<sham::DeviceBuffer<u32>> refine_list;
    shambase::DistributedData<sham::DeviceBuffer<u32>> derefine_list;

    gen_refine_block_changes<UserAccCrit>(refine_list, derefine_list);

    //////// apply refine ////////
    // Note that this only add new blocks at the end of the patchdata
    internal_refine_grid<UserAccSplit>(std::move(refine_list));

    //////// apply derefine ////////
    // Note that this will perform the merge then remove the old blocks
    // This is ok to call straight after the refine without edditing the index list in derefine_list
    // since no permutations were applied in internal_refine_grid and no cells can be both refined
    // and derefined in the same pass
    internal_derefine_grid<UserAccMerge>(std::move(derefine_list));
}

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::AMRGridRefinementHandler<Tvec, TgridVec>::
    update_refinement() {

    class RefineCritBlock {
        public:
        const TgridVec *block_low_bound;
        const TgridVec *block_high_bound;
        const Tscal *block_density_field;

        Tscal one_over_Nside = 1. / AMRBlock::Nside;

        Tscal dxfact;
        Tscal wanted_mass;

        RefineCritBlock(
            sham::EventList &depends_list,
            u64 id_patch,
            shamrock::patch::Patch p,
            shamrock::patch::PatchDataLayer &pdat,
            Tscal dxfact,
            Tscal wanted_mass)
            : dxfact(dxfact), wanted_mass(wanted_mass) {

            block_low_bound  = pdat.get_field<TgridVec>(0).get_buf().get_read_access(depends_list);
            block_high_bound = pdat.get_field<TgridVec>(1).get_buf().get_read_access(depends_list);
            block_density_field = pdat.get_field<Tscal>(pdat.pdl().get_field_idx<Tscal>("rho"))
                                      .get_buf()
                                      .get_read_access(depends_list);
        }

        void finalize(
            sham::EventList &resulting_events,
            u64 id_patch,
            shamrock::patch::Patch p,
            shamrock::patch::PatchDataLayer &pdat,
            Tscal dxfact,
            Tscal wanted_mass) {

            sham::DeviceBuffer<i64_3> &buf_cell_low_bound  = pdat.get_field<i64_3>(0).get_buf();
            sham::DeviceBuffer<i64_3> &buf_cell_high_bound = pdat.get_field<i64_3>(1).get_buf();

            buf_cell_low_bound.complete_event_state(resulting_events);
            buf_cell_high_bound.complete_event_state(resulting_events);
            pdat.get_field<Tscal>(pdat.pdl().get_field_idx<Tscal>("rho"))
                .get_buf()
                .complete_event_state(resulting_events);
        }

        void refine_criterion(
            u32 block_id, RefineCritBlock acc, bool &should_refine, bool &should_derefine) const {

            TgridVec low_bound  = acc.block_low_bound[block_id];
            TgridVec high_bound = acc.block_high_bound[block_id];

            Tvec lower_flt = low_bound.template convert<Tscal>() * dxfact;
            Tvec upper_flt = high_bound.template convert<Tscal>() * dxfact;

            Tvec block_cell_size = (upper_flt - lower_flt) * one_over_Nside;

            Tscal sum_mass = 0;
            for (u32 i = 0; i < AMRBlock::block_size; i++) {
                sum_mass += acc.block_density_field[i + block_id * AMRBlock::block_size];
            }
            sum_mass *= block_cell_size.x() * block_cell_size.y() * block_cell_size.z();

            if (sum_mass > wanted_mass * 8) {
                should_refine   = true;
                should_derefine = false;
            } else if (sum_mass < wanted_mass) {
                should_refine   = false;
                should_derefine = true;
            } else {
                should_refine   = false;
                should_derefine = false;
            }

            should_refine = should_refine && (high_bound.x() - low_bound.x() > AMRBlock::Nside);
            should_refine = should_refine && (high_bound.y() - low_bound.y() > AMRBlock::Nside);
            should_refine = should_refine && (high_bound.z() - low_bound.z() > AMRBlock::Nside);
        }
    };

    class RefineCellAccessor {
        public:
        f64 *rho;
        f64_3 *rho_vel;
        f64 *rhoE;

        RefineCellAccessor(sham::EventList &depends_list, shamrock::patch::PatchDataLayer &pdat) {

            rho     = pdat.get_field<f64>(2).get_buf().get_write_access(depends_list);
            rho_vel = pdat.get_field<f64_3>(3).get_buf().get_write_access(depends_list);
            rhoE    = pdat.get_field<f64>(4).get_buf().get_write_access(depends_list);
        }

        void finalize(sham::EventList &resulting_events, shamrock::patch::PatchDataLayer &pdat) {
            pdat.get_field<f64>(2).get_buf().complete_event_state(resulting_events);
            pdat.get_field<f64_3>(3).get_buf().complete_event_state(resulting_events);
            pdat.get_field<f64>(4).get_buf().complete_event_state(resulting_events);
        }

        void apply_refine(
            u32 cur_idx,
            BlockCoord cur_coords,
            std::array<u32, 8> new_blocks,
            std::array<BlockCoord, 8> new_block_coords,
            RefineCellAccessor acc) const {

            auto get_coord_ref = [](u32 i) -> std::array<u32, dim> {
                constexpr u32 NsideBlockPow = 1;
                constexpr u32 Nside         = 1U << NsideBlockPow;

                if constexpr (dim == 3) {
                    const u32 tmp = i >> NsideBlockPow;
                    return {i % Nside, (tmp) % Nside, (tmp) >> NsideBlockPow};
                }
            };

            auto get_index_block = [](std::array<u32, dim> coord) -> u32 {
                constexpr u32 NsideBlockPow = 1;
                constexpr u32 Nside         = 1U << NsideBlockPow;

                if constexpr (dim == 3) {
                    return coord[0] + Nside * coord[1] + Nside * Nside * coord[2];
                }
            };

            auto get_gid_write = [&](std::array<u32, dim> &glid) -> u32 {
                std::array<u32, dim> bid
                    = {glid[0] >> AMRBlock::NsideBlockPow,
                       glid[1] >> AMRBlock::NsideBlockPow,
                       glid[2] >> AMRBlock::NsideBlockPow};

                // logger::raw_ln(glid,bid);
                return new_blocks[get_index_block(bid)] * AMRBlock::block_size
                       + AMRBlock::get_index(
                           {glid[0] % AMRBlock::Nside,
                            glid[1] % AMRBlock::Nside,
                            glid[2] % AMRBlock::Nside});
            };

            std::array<f64, AMRBlock::block_size> old_rho_block;
            std::array<f64_3, AMRBlock::block_size> old_rho_vel_block;
            std::array<f64, AMRBlock::block_size> old_rhoE_block;

            // save old block
            for (u32 loc_id = 0; loc_id < AMRBlock::block_size; loc_id++) {

                auto [lx, ly, lz]         = get_coord_ref(loc_id);
                u32 old_cell_idx          = cur_idx * AMRBlock::block_size + loc_id;
                old_rho_block[loc_id]     = acc.rho[old_cell_idx];
                old_rho_vel_block[loc_id] = acc.rho_vel[old_cell_idx];
                old_rhoE_block[loc_id]    = acc.rhoE[old_cell_idx];
            }

            for (u32 loc_id = 0; loc_id < AMRBlock::block_size; loc_id++) {

                auto [lx, ly, lz] = get_coord_ref(loc_id);
                u32 old_cell_idx  = cur_idx * AMRBlock::block_size + loc_id;

                Tscal rho_block    = old_rho_block[loc_id];
                Tvec rho_vel_block = old_rho_vel_block[loc_id];
                Tscal rhoE_block   = old_rhoE_block[loc_id];
                for (u32 subdiv_lid = 0; subdiv_lid < 8; subdiv_lid++) {

                    auto [sx, sy, sz] = get_coord_ref(subdiv_lid);

                    std::array<u32, 3> glid = {lx * 2 + sx, ly * 2 + sy, lz * 2 + sz};

                    u32 new_cell_idx = get_gid_write(glid);
                    /*
                                        if (1627 == cur_idx) {
                                            logger::raw_ln(
                                                cur_idx,
                                                "set cell ",
                                                new_cell_idx,
                                                " from cell",
                                                old_cell_idx,
                                                "old",
                                                rho_block,
                                                rho_vel_block,
                                                rhoE_block);
                                        }
                                        */
                    acc.rho[new_cell_idx]     = rho_block;
                    acc.rho_vel[new_cell_idx] = rho_vel_block;
                    acc.rhoE[new_cell_idx]    = rhoE_block;
                }
            }
        }

        void apply_derefine(
            std::array<u32, 8> old_blocks,
            std::array<BlockCoord, 8> old_coords,
            u32 new_cell,
            BlockCoord new_coord,

            RefineCellAccessor acc) const {

            std::array<f64, AMRBlock::block_size> rho_block;
            std::array<f64_3, AMRBlock::block_size> rho_vel_block;
            std::array<f64, AMRBlock::block_size> rhoE_block;

            for (u32 cell_id = 0; cell_id < AMRBlock::block_size; cell_id++) {
                rho_block[cell_id]     = {};
                rho_vel_block[cell_id] = {};
                rhoE_block[cell_id]    = {};
            }

            for (u32 pid = 0; pid < 8; pid++) {
                for (u32 cell_id = 0; cell_id < AMRBlock::block_size; cell_id++) {
                    rho_block[cell_id] += acc.rho[old_blocks[pid] * AMRBlock::block_size + cell_id];
                    rho_vel_block[cell_id]
                        += acc.rho_vel[old_blocks[pid] * AMRBlock::block_size + cell_id];
                    rhoE_block[cell_id]
                        += acc.rhoE[old_blocks[pid] * AMRBlock::block_size + cell_id];
                }
            }

            for (u32 cell_id = 0; cell_id < AMRBlock::block_size; cell_id++) {
                rho_block[cell_id] /= 8;
                rho_vel_block[cell_id] /= 8;
                rhoE_block[cell_id] /= 8;
            }

            for (u32 cell_id = 0; cell_id < AMRBlock::block_size; cell_id++) {
                u32 newcell_idx          = new_cell * AMRBlock::block_size + cell_id;
                acc.rho[newcell_idx]     = rho_block[cell_id];
                acc.rho_vel[newcell_idx] = rho_vel_block[cell_id];
                acc.rhoE[newcell_idx]    = rhoE_block[cell_id];
            }
        }
    };

    using AMRmode_None         = typename AMRMode<Tvec, TgridVec>::None;
    using AMRmode_DensityBased = typename AMRMode<Tvec, TgridVec>::DensityBased;

    bool has_cell_order_changed = false;

    if (AMRmode_None *cfg = std::get_if<AMRmode_None>(&solver_config.amr_mode.config)) {
        // no refinment here turn around there is nothing to see
    } else if (
        AMRmode_DensityBased *cfg
        = std::get_if<AMRmode_DensityBased>(&solver_config.amr_mode.config)) {
        Tscal dxfact(solver_config.grid_coord_to_pos_fact);

        // get refine and derefine list
        shambase::DistributedData<sham::DeviceBuffer<u32>> refine_flags;
        shambase::DistributedData<sham::DeviceBuffer<u32>> derefine_flags;

        shambase::DistributedData<sham::DeviceBuffer<u32>> refine_list;
        shambase::DistributedData<sham::DeviceBuffer<u32>> derefine_list;

        gen_refine_block_changes<RefineCritBlock>(
            refine_flags, derefine_flags, dxfact, cfg->crit_mass);

        ///// enforce 2:1 for refinement ///////
        enforce_two_to_one_for_refinement(refine_flags, refine_list);

        ///// Check geometrical validity for derefinement ///////
        check_geometrical_validity_for_derefinement(derefine_flags, refine_flags);

        ////// 2:1 for derefinement  /////
        enforce_two_to_one_for_derefinement(derefine_flags, derefine_list);

        //////// apply refine ////////
        // Note that this only add new blocks at the end of the patchdata
        bool change_refine = internal_refine_grid<RefineCellAccessor>(std::move(refine_list));

        //////// apply derefine ////////
        // Note that this will perform the merge then remove the old blocks
        // This is ok to call straight after the refine without edditing the index list in
        // derefine_list since no permutations were applied in internal_refine_grid and no cells can
        // be both refined and derefined in the same pass
        bool change_derefine = false;
        //  internal_derefine_grid<RefineCellAccessor>(std::move(derefine_list));

        has_cell_order_changed = has_cell_order_changed || (change_refine || change_derefine);
    }

    if (has_cell_order_changed) {
        // Ensure that the blocks are sorted before refinement
        AMRSortBlocks block_sorter(context, solver_config, storage);
        block_sorter.reorder_amr_blocks();
    }
}

template class shammodels::basegodunov::modules::AMRGridRefinementHandler<f64_3, i64_3>;
