// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file AMRGridRefinementHandler.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Sewanou Leodasce (lsewanou@pxe.cbp.ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/aliases_int.hpp"
#include "shambase/memory.hpp"
#include "shamalgs/details/algorithm/algorithm.hpp"
#include "shambackends/EventList.hpp"
#include "shambackends/math.hpp"
#include "shambackends/typeAliasVec.hpp"
#include "shamcomm/logs.hpp"
#include "shammath/riemann.hpp"
#include "shammodels/common/amr/NeighGraph.hpp"
#include "shammodels/ramses/SolverConfig.hpp"
#include "shammodels/ramses/modules/AMRGridRefinementHandler.hpp"
#include "shammodels/ramses/modules/AMRSortBlocks.hpp"
#include "shammodels/ramses/modules/ComputeAMRLevel.hpp"
#include "shammodels/ramses/modules/InterpolationUtilities.hpp"
#include "shammodels/ramses/modules/SlopeLimitedGradientUtilities.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include <functional>
#include <stdexcept>

/**
 * @brief build refinement map (this is done patch per patch, shall we do it level per level?)
 * @tparam Tvec
 * @tparam TgridVec
 * @tparam UserAcc  User provided criterion for refinement
 * @tparam T        others template arguments (types for arguments needed by UserAcc)
 * @param refine_list refinement map
 * @param derefine_list derefinement map
 *
 */
template<class Tvec, class TgridVec>
template<class UserAcc, class... T>
void shammodels::basegodunov::modules::AMRGridRefinementHandler<Tvec, TgridVec>::
    gen_refine_block_changes(
        shambase::DistributedData<OptIndexList> &refine_list,
        shambase::DistributedData<OptIndexList> &derefine_list,
        T &&...args) {

    using namespace shamrock::patch;

    using AMRGraph             = shammodels::basegodunov::modules::AMRGraph;
    using Direction_           = shammodels::basegodunov::modules::Direction;
    using AMRGraphLinkiterator = shammodels::basegodunov::modules::AMRGraph::ro_access;
    using TgridUint = typename std::make_unsigned<shambase::VecComponent<TgridVec>>::type;

    // flag blocks for refinement or derefinement based on user-provided criterion
    u64 tot_refine   = 0;
    u64 tot_derefine = 0;

    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

        u64 id_patch = cur_p.id_patch;

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

        // create the refine and derefine flags buffers
        u32 obj_cnt = pdat.get_obj_cnt();

        sycl::buffer<u32> refine_flags(obj_cnt);
        sycl::buffer<u32> derefine_flags(obj_cnt);
        {
            sham::EventList depends_list;

            UserAcc uacc(depends_list, storage, id_patch, cur_p, pdat, args...);

            // fill in the flags
            auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                sycl::accessor refine_acc{refine_flags, cgh, sycl::write_only, sycl::no_init};
                sycl::accessor derefine_acc{derefine_flags, cgh, sycl::write_only, sycl::no_init};

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

            uacc.finalize(resulting_events, storage, id_patch, cur_p, pdat, args...);
        }
        sham::DeviceBuffer<TgridVec> &buf_cell_min = pdat.get_field_buf_ref<TgridVec>(0);
        sham::DeviceBuffer<TgridVec> &buf_cell_max = pdat.get_field_buf_ref<TgridVec>(1);

        sham::EventList depends_list;
        auto acc_min = buf_cell_min.get_read_access(depends_list);
        auto acc_max = buf_cell_max.get_read_access(depends_list);

        AMRGraphLinkiterator block_graph_xp = block_graph_neighs_xp.get_read_access(depends_list);
        AMRGraphLinkiterator block_graph_xm = block_graph_neighs_xm.get_read_access(depends_list);
        AMRGraphLinkiterator block_graph_yp = block_graph_neighs_yp.get_read_access(depends_list);
        AMRGraphLinkiterator block_graph_ym = block_graph_neighs_ym.get_read_access(depends_list);
        AMRGraphLinkiterator block_graph_zp = block_graph_neighs_zp.get_read_access(depends_list);
        AMRGraphLinkiterator block_graph_zm = block_graph_neighs_zm.get_read_access(depends_list);

        auto acc_amr_block_levels = buf_amr_block_levels.get_read_access(depends_list);

        // Enforce 2:1 restriction using blocks_neighborh graph and amr_levels
        auto e1 = q.submit(depends_list, [&](sycl::handler &cgh) {
            sycl::accessor acc_refine_flag{refine_flags, cgh, sycl::read_write};

            cgh.parallel_for(sycl::range<1>(obj_cnt), [=](sycl::item<1> gid) {
                u32 block_id                = gid.get_linear_id();
                u32 current_refinement_flag = acc_refine_flag[block_id];
                auto current_block_level    = acc_amr_block_levels[block_id];

                auto apply_to_each_neigh_block = [&](u32 b_id) {
                    u32 neigh_refine_flag  = acc_refine_flag[b_id];
                    auto neigh_block_level = acc_amr_block_levels[b_id];
                    /* current block (cur_b) is at level L1 and neighborh block (neigh_b) L2
                        such that L1 >= L2.
                        *
                        * Case 1 : cur_b and neigh_b are the same level L, cur_b is marked for
                        refinement but not neigh_b. We refine also neigh_b.
                        * Case 2 : cur_b and neigh_b are the same level L, cur_b and neigh_b
                        are both marked for refinement. We do nothing.
                        * Case 3 : cur_b is at level L1 and neigh at level L2 with (L1 - L2 =
                        1), cur_b is marked for refinement but not neigh_b. We refine also
                        neigh_b.
                        * Case 4 : cur_b is at level L1 and neigh at level L2 with (L1 - L2 =
                        1), cur_b and neigh_b are both marked for refinement. We do nothing.
                        * It's important to observe that we don't consider the case with
                        abs(L1-L2)>1 because the 2:1 condition should prevent this.
                    */

                    if (current_block_level - neigh_block_level == 1) {
                        sycl::
                            atomic_ref<u32, sycl::memory_order::relaxed, sycl::memory_scope::system>
                                atomic_flag(acc_refine_flag[b_id]);

                        // atomic_flag.fetch_or(1); // Atomically set flag to true
                        atomic_flag.exchange(1); // Atomically set flag to true
                    }
                    acc_refine_flag[b_id] = neigh_refine_flag;
                };

                if (current_refinement_flag) {
                    block_graph_xp.for_each_object_link(block_id, [&](u32 neigh_block_id) {
                        apply_to_each_neigh_block(neigh_block_id);
                    });
                    block_graph_xm.for_each_object_link(block_id, [&](u32 neigh_block_id) {
                        apply_to_each_neigh_block(neigh_block_id);
                    });
                    block_graph_yp.for_each_object_link(block_id, [&](u32 neigh_block_id) {
                        apply_to_each_neigh_block(neigh_block_id);
                    });
                    block_graph_ym.for_each_object_link(block_id, [&](u32 neigh_block_id) {
                        apply_to_each_neigh_block(neigh_block_id);
                    });
                    block_graph_zp.for_each_object_link(block_id, [&](u32 neigh_block_id) {
                        apply_to_each_neigh_block(neigh_block_id);
                    });
                    block_graph_zm.for_each_object_link(block_id, [&](u32 neigh_block_id) {
                        apply_to_each_neigh_block(neigh_block_id);
                    });
                }
            });
        });

        block_graph_neighs_xp.complete_event_state(e1);
        block_graph_neighs_xm.complete_event_state(e1);
        block_graph_neighs_yp.complete_event_state(e1);
        block_graph_neighs_ym.complete_event_state(e1);
        block_graph_neighs_zp.complete_event_state(e1);
        block_graph_neighs_zm.complete_event_state(e1);
        buf_amr_block_levels.complete_event_state(e1);

        // keep derefine flags on only if the eight cells want to merge and if they can
        auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
            sycl::accessor acc_merge_flag{derefine_flags, cgh, sycl::read_write};

            cgh.parallel_for(sycl::range<1>(obj_cnt), [=](sycl::item<1> gid) {
                u32 id = gid.get_linear_id();

                std::array<BlockCoord, split_count> blocks;
                bool do_merge = true;

                // This avoid the case where we are in the last block of the buffer to avoid the
                // out-of-bound read
                if (id + split_count <= obj_cnt) {
                    bool all_want_to_merge = true;

                    for (u32 lid = 0; lid < split_count; lid++) {
                        blocks[lid]       = BlockCoord{acc_min[gid + lid], acc_max[gid + lid]};
                        all_want_to_merge = all_want_to_merge && acc_merge_flag[gid + lid];
                    }

                    do_merge = all_want_to_merge && BlockCoord::are_mergeable(blocks);

                } else {
                    do_merge = false;
                }

                acc_merge_flag[gid] = do_merge;
            });
        });

        buf_cell_min.complete_event_state(e);
        buf_cell_max.complete_event_state(e);

        ////////////////////////////////////////////////////////////////////////////////
        // refinement
        ////////////////////////////////////////////////////////////////////////////////

        // perform stream compactions on the refinement flags
        auto [buf_refine, len_refine]
            = shamalgs::numeric::stream_compact(q.q, refine_flags, obj_cnt);

        shamlog_debug_ln("AMRGrid", "patch ", id_patch, len_refine, "marked for refinement ");

        tot_refine += len_refine;

        // add the results to the map
        refine_list.add_obj(id_patch, OptIndexList{std::move(buf_refine), len_refine});

        ////////////////////////////////////////////////////////////////////////////////
        // derefinement
        ////////////////////////////////////////////////////////////////////////////////

        // perform stream compactions on the derefinement flags
        auto [buf_derefine, len_derefine]
            = shamalgs::numeric::stream_compact(q.q, derefine_flags, obj_cnt);

        shamlog_debug_ln("AMRGrid", "patch ", id_patch, len_derefine, "marked for derefinement ");

        tot_derefine += len_derefine;

        // add the results to the map
        derefine_list.add_obj(id_patch, OptIndexList{std::move(buf_derefine), len_derefine});
    });

    logger::info_ln("AMRGrid", "on this process", tot_refine, "blocks will be refined");
    logger::info_ln(
        "AMRGrid", "on this process", tot_derefine * split_count, "blocks will be derefined");
}

/**
 * @brief Performed refinement
 *        For each flagged block for refinement build its 2^{dim} child blocks.
 *        for each patch, new blocks are temporary place at the end of its block list.
 *        More precisely the first block overwrites its parent and the rest of its siblings are
 * place at the end.
 * @tparam Tvec
 * @tparam TgridVec
 * @tparam UserAcc
 * @param refine_list refinement map of all patches on the current MPI process
 *
 */
template<class Tvec, class TgridVec>
template<class UserAcc>
void shammodels::basegodunov::modules::AMRGridRefinementHandler<Tvec, TgridVec>::
    internal_refine_grid(shambase::DistributedData<OptIndexList> &&refine_list) {

    using namespace shamrock::patch;

    u64 sum_block_count = 0;

    scheduler().for_each_patch_data([&](u64 id_patch, Patch cur_p, PatchDataLayer &pdat) {
        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

        u32 old_obj_cnt = pdat.get_obj_cnt();

        OptIndexList &refine_flags = refine_list.get(id_patch);

        if (refine_flags.count > 0) {

            // alloc memory for the new blocks to be created
            pdat.expand(refine_flags.count * (split_count - 1));

            sham::DeviceBuffer<TgridVec> &buf_cell_min = pdat.get_field_buf_ref<TgridVec>(0);
            sham::DeviceBuffer<TgridVec> &buf_cell_max = pdat.get_field_buf_ref<TgridVec>(1);

            sham::EventList depends_list;
            auto block_bound_low  = buf_cell_min.get_write_access(depends_list);
            auto block_bound_high = buf_cell_max.get_write_access(depends_list);
            // UserAcc uacc(depends_list, pdat);

            UserAcc uacc(depends_list, storage, id_patch, pdat);

            // Refine the block (set the positions) and fill the corresponding fields
            auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                sycl::accessor index_to_ref{*refine_flags.idx, cgh, sycl::read_only};

                u32 start_index_push = old_obj_cnt;

                constexpr u32 new_splits = split_count - 1;

                cgh.parallel_for(sycl::range<1>(refine_flags.count), [=](sycl::item<1> gid) {
                    u32 tid = gid.get_linear_id();

                    u32 idx_to_refine = index_to_ref[gid];

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

            sham::EventList resulting_events;

            buf_cell_min.complete_event_state(resulting_events);
            buf_cell_max.complete_event_state(resulting_events);

            uacc.finalize(resulting_events, storage, id_patch, pdat);
        }
        shamlog_debug_ln("AMRGrid", "patch ", id_patch, "new block count = ", pdat.get_obj_cnt());

        sum_block_count += pdat.get_obj_cnt();
    });

    logger::info_ln("AMRGrid", "process block count =", sum_block_count);
}

/**
 * @brief Performed derefinement and remove old blocks
 * @tparam Tvec
 * @tparam TgridVec
 * @tparam UserAcc
 * @param derefine_list derefinement map of all patches on the current MPI process
 *
 */
template<class Tvec, class TgridVec>
template<class UserAcc>
void shammodels::basegodunov::modules::AMRGridRefinementHandler<Tvec, TgridVec>::
    internal_derefine_grid(shambase::DistributedData<OptIndexList> &&derefine_list) {

    using namespace shamrock::patch;

    scheduler().for_each_patch_data([&](u64 id_patch, Patch cur_p, PatchDataLayer &pdat) {
        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

        // old  block count in the patch
        u32 old_obj_cnt = pdat.get_obj_cnt();

        // derefinement map of the patch (ids of blocks needed to be merged)
        OptIndexList &derefine_flags = derefine_list.get(id_patch);

        if (derefine_flags.count > 0) {

            // init flag table
            sycl::buffer<u32> keep_block_flag
                = shamalgs::algorithm::gen_buffer_device(q.q, old_obj_cnt, [](u32 i) -> u32 {
                      return 1;
                  });

            sham::DeviceBuffer<TgridVec> &buf_cell_min = pdat.get_field_buf_ref<TgridVec>(0);
            sham::DeviceBuffer<TgridVec> &buf_cell_max = pdat.get_field_buf_ref<TgridVec>(1);

            sham::EventList depends_list;
            auto block_bound_low  = buf_cell_min.get_write_access(depends_list);
            auto block_bound_high = buf_cell_max.get_write_access(depends_list);
            // UserAcc uacc(depends_list, pdat);
            UserAcc uacc(depends_list, storage, id_patch, pdat);

            // edit block content + make flag of blocks to keep
            auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                sycl::accessor index_to_deref{*derefine_flags.idx, cgh, sycl::read_only};

                sycl::accessor flag_keep{keep_block_flag, cgh, sycl::read_write};

                cgh.parallel_for(sycl::range<1>(derefine_flags.count), [=](sycl::item<1> gid) {
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

            sham::EventList resulting_events;

            buf_cell_min.complete_event_state(resulting_events);
            buf_cell_max.complete_event_state(resulting_events);

            uacc.finalize(resulting_events, storage, id_patch, pdat);

            // stream compact the flags (get new block ids map after merged)
            auto [opt_buf, len]
                = shamalgs::numeric::stream_compact(q.q, keep_block_flag, old_obj_cnt);

            // shamlog_debug_ln(
            //     "AMR Grid", "patch", id_patch, "derefine block count ", old_obj_cnt, "->", len);

            logger::info_ln(
                "AMR Grid",
                "patch",
                id_patch,
                "derefine block count = ",
                old_obj_cnt - len,
                "new block count = ",
                len);

            // shamlog_debug_ln(
            //     "AMR Grid",
            //     "patch",
            //     id_patch,
            //     "derefine block count = ",
            //     old_obj_cnt - len,
            //     "new block count = ",
            //     len);

            if (!opt_buf) {
                throw std::runtime_error("opt buf must contain something at this point");
            }

            // remap pdat according to stream compact (for each field in patchdataleyer resize
            // according to new block ids map)
            pdat.index_remap_resize(*opt_buf, len);
        }
    });
}

template<class Tvec, class TgridVec>
template<class UserAccCrit, class UserAccSplit, class UserAccMerge>
void shammodels::basegodunov::modules::AMRGridRefinementHandler<Tvec, TgridVec>::
    internal_update_refinement() {

    // Ensure that the blocks are sorted before refinement
    AMRSortBlocks block_sorter(context, solver_config, storage);
    block_sorter.reorder_amr_blocks();

    // get refine and derefine list
    shambase::DistributedData<OptIndexList> refine_list;
    shambase::DistributedData<OptIndexList> derefine_list;

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
            Storage &storage,
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
            Storage &storage,
            u64 id_patch,
            shamrock::patch::Patch p,
            shamrock::patch::PatchDataLayer &pdat,
            Tscal dxfact,
            Tscal wanted_mass) {

            pdat.get_field<TgridVec>(0).get_buf().complete_event_state(resulting_events);
            pdat.get_field<TgridVec>(1).get_buf().complete_event_state(resulting_events);

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
        using AMRGraph             = shammodels::basegodunov::modules::AMRGraph;
        using Direction_           = shammodels::basegodunov::modules::Direction;
        using AMRGraphLinkiterator = shammodels::basegodunov::modules::AMRGraph::ro_access;

        f64 *rho;
        f64_3 *rho_vel;
        f64 *rhoE;
        u64 p_id;
        f64 *cell_size;
        f64_3 *block_aabb_lower;
        f64 *press;
        f64_3 *vel;

        AMRGraphLinkiterator cell_graph_xp;
        AMRGraphLinkiterator cell_graph_xm;
        AMRGraphLinkiterator cell_graph_yp;
        AMRGraphLinkiterator cell_graph_ym;
        AMRGraphLinkiterator cell_graph_zp;
        AMRGraphLinkiterator cell_graph_zm;

        RefineCellAccessor(
            sham::EventList &depends_list,
            Storage &storage,
            u64 &id_patch,
            shamrock::patch::PatchDataLayer &pdat)
            : cell_graph_xp(
                  shambase::get_check_ref(storage.cell_graph_edge)
                      .get_refs_dir(Direction_::xp)
                      .get(id_patch)
                      .get()
                      .get_read_access(depends_list)),
              cell_graph_xm(
                  shambase::get_check_ref(storage.cell_graph_edge)
                      .get_refs_dir(Direction_::xm)
                      .get(id_patch)
                      .get()
                      .get_read_access(depends_list)),
              cell_graph_yp(
                  shambase::get_check_ref(storage.cell_graph_edge)
                      .get_refs_dir(Direction_::yp)
                      .get(id_patch)
                      .get()
                      .get_read_access(depends_list)),
              cell_graph_ym(
                  shambase::get_check_ref(storage.cell_graph_edge)
                      .get_refs_dir(Direction_::ym)
                      .get(id_patch)
                      .get()
                      .get_read_access(depends_list)),
              cell_graph_zp(
                  shambase::get_check_ref(storage.cell_graph_edge)
                      .get_refs_dir(Direction_::zp)
                      .get(id_patch)
                      .get()
                      .get_read_access(depends_list)),
              cell_graph_zm(
                  shambase::get_check_ref(storage.cell_graph_edge)
                      .get_refs_dir(Direction_::zm)
                      .get(id_patch)
                      .get()
                      .get_read_access(depends_list))

        {
            p_id      = id_patch;
            rho       = pdat.get_field<f64>(2).get_buf().get_write_access(depends_list);
            rho_vel   = pdat.get_field<f64_3>(3).get_buf().get_write_access(depends_list);
            rhoE      = pdat.get_field<f64>(4).get_buf().get_write_access(depends_list);
            cell_size = shambase::get_check_ref(storage.block_cell_sizes)
                            .get_buf(id_patch)
                            .get_write_access(depends_list);
            block_aabb_lower = shambase::get_check_ref(storage.cell0block_aabb_lower)
                                   .get_buf(id_patch)
                                   .get_write_access(depends_list);
            press = shambase::get_check_ref(storage.press)
                                   .get_buf(id_patch)
                                   .get_write_access(depends_list);
            vel = shambase::get_check_ref(storage.vel)
                                   .get_buf(id_patch)
                                   .get_write_access(depends_list);
        }

        void finalize(
            sham::EventList &resulting_events,
            Storage &storage,
            u64 &id_patch,
            shamrock::patch::PatchDataLayer &pdat) {
            pdat.get_field<f64>(2).get_buf().complete_event_state(resulting_events);
            pdat.get_field<f64_3>(3).get_buf().complete_event_state(resulting_events);
            pdat.get_field<f64>(4).get_buf().complete_event_state(resulting_events);
            shambase::get_check_ref(storage.block_cell_sizes)
                .get_buf(id_patch)
                .complete_event_state(resulting_events);
            shambase::get_check_ref(storage.cell0block_aabb_lower)
                .get_buf(id_patch)
                .complete_event_state(resulting_events);

            shambase::get_check_ref(storage.cell_graph_edge)
                .get_refs_dir(Direction_::xp)
                .get(id_patch)
                .get()
                .complete_event_state(resulting_events);
            shambase::get_check_ref(storage.cell_graph_edge)
                .get_refs_dir(Direction_::xm)
                .get(id_patch)
                .get()
                .complete_event_state(resulting_events);
            shambase::get_check_ref(storage.cell_graph_edge)
                .get_refs_dir(Direction_::yp)
                .get(id_patch)
                .get()
                .complete_event_state(resulting_events);
            shambase::get_check_ref(storage.cell_graph_edge)
                .get_refs_dir(Direction_::ym)
                .get(id_patch)
                .get()
                .complete_event_state(resulting_events);
            shambase::get_check_ref(storage.cell_graph_edge)
                .get_refs_dir(Direction_::zp)
                .get(id_patch)
                .get()
                .complete_event_state(resulting_events);
            shambase::get_check_ref(storage.cell_graph_edge)
                .get_refs_dir(Direction_::zm)
                .get(id_patch)
                .get()
                .complete_event_state(resulting_events);
            
            shambase::get_check_ref(storage.press)
                                   .get_buf(id_patch)
                                   .complete_event_state(resulting_events);

           shambase::get_check_ref(storage.vel)
                                   .get_buf(id_patch)
                                   .complete_event_state(resulting_events);
            
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

            std::array<f64_3, AMRBlock::block_size> old_vel_block;
            std::array<f64, AMRBlock::block_size> old_press_block;

            // save old block
            for (u32 loc_id = 0; loc_id < AMRBlock::block_size; loc_id++) {

                /* local integer coordinate in the block ( 0 -> (0,0,0) ; 1 -> (1,0,0); 2 ->
                   (0,1,0); 3 -> (1,1,0) ; 4 -> (0,0,1) ; 5 -> (1,0,1); 6 -> (0,1,1) ; 7 -> (1,1,1)
                */
                auto [lx, ly, lz] = get_coord_ref(loc_id);

                // global child id of the cell in the patch
                u32 old_cell_idx          = cur_idx * AMRBlock::block_size + loc_id;
                old_rho_block[loc_id]     = acc.rho[old_cell_idx];
                old_rho_vel_block[loc_id] = acc.rho_vel[old_cell_idx];
                old_rhoE_block[loc_id]    = acc.rhoE[old_cell_idx];
                old_vel_block[loc_id] = acc.vel[old_cell_idx];
                old_press_block[loc_id]    = acc.press[old_cell_idx];
            }

            for (u32 loc_id = 0; loc_id < AMRBlock::block_size; loc_id++) {

/*


                // local integer coordinate in the block ( 0 -> (0,0,0) ; 1 -> (1,0,0); 2 -> (0,1,0)
                //
                auto [lx, ly, lz] = get_coord_ref(loc_id);

                // global child id of the cell in the refined block
                u32 old_cell_idx = cur_idx * AMRBlock::block_size + loc_id;

                // cell size in the refined block
                Tscal delta_cell = cell_size[cur_idx];
                Tscal c_offset   = delta_cell / 4;

                Tscal rho_block    = old_rho_block[loc_id];
                Tvec rho_vel_block = old_rho_vel_block[loc_id];
                Tscal rhoE_block   = old_rhoE_block[loc_id];
                Tvec vel_block = old_vel_block[loc_id];
                Tscal press_block   = old_press_block[loc_id];

                // logger::raw_ln("id ", old_cell_idx, "rho_block ", rho_block);

                // for old_cell_idx in the refined block fill fields for each of its eight child
                // cells
                //              . . . . .
                //   (1,0) <--  . x . x . -->(1,1)
                //              . . X . .
                //   (0,0) <--  . x . x . -->(0,1)
                //              . . . . .
                //

                std::array<f64_3, AMRBlock::block_size> child_center_offsets;
                child_center_offsets[0] = {-c_offset, -c_offset, -c_offset}; // 0 : (0,0,0)
                child_center_offsets[1] = {c_offset, -c_offset, -c_offset};  // 1 : (1,0,0)
                child_center_offsets[2] = {-c_offset, c_offset, -c_offset};  // 2 : (0,1,0)
                child_center_offsets[3] = {c_offset, c_offset, -c_offset};   // 3 : (1,1,0)

                child_center_offsets[4] = {-c_offset, -c_offset, c_offset}; // 4 : (0,0,1)
                child_center_offsets[5] = {c_offset, -c_offset, c_offset};  // 5 : (1,0,1)
                child_center_offsets[6] = {-c_offset, c_offset, c_offset};  // 6 : (0,1,1)
                child_center_offsets[7] = {c_offset, c_offset, c_offset};   // 7 : (1,1,1)

                constexp
                static_assert(AMRBlock::get_coord(0) == std::array<u32, dim>{0,0,0} , "");
                

                // // limited slopes of the refined cells
                // // TODO : generalize to other slope limiter modes and arbitrary nvar
                auto result_rho = get_3d_grad<Tscal, Tvec, Minmod>(
                    old_cell_idx,
                    delta_cell,
                    cell_graph_xp,
                    cell_graph_xm,
                    cell_graph_yp,
                    cell_graph_ym,
                    cell_graph_zp,
                    cell_graph_zm,
                    [=](u32 id) {
                        return acc.rho[id];
                    });
                // logger::raw_ln("rho_grad ", old_cell_idx, result_rho[0], result_rho[1],
                // result_rho[2],"\n");

                // auto result_rhoe = get_3d_grad<Tscal, Tvec, Minmod>(
                //     old_cell_idx,
                //     delta_cell,
                //     cell_graph_xp,
                //     cell_graph_xm,
                //     cell_graph_yp,
                //     cell_graph_ym,
                //     cell_graph_zp,
                //     cell_graph_zm,
                //     [=](u32 id) {
                //         return acc.rhoE[id];
                //     });

                
                auto result_press = get_3d_grad<Tscal, Tvec, Minmod>(
                    old_cell_idx,
                    delta_cell,
                    cell_graph_xp,
                    cell_graph_xm,
                    cell_graph_yp,
                    cell_graph_ym,
                    cell_graph_zp,
                    cell_graph_zm,
                    [=](u32 id) {
                        return acc.press[id];
                    });
                // logger::raw_ln("rhoe_grad ", old_cell_idx, result_rhoe[0], result_rhoe[1],
                // result_rhoe[2],"\n");

                // auto result_rhov = get_3d_grad<Tvec, Tvec, Minmod>(
                //     old_cell_idx,
                //     delta_cell,
                //     cell_graph_xp,
                //     cell_graph_xm,
                //     cell_graph_yp,
                //     cell_graph_ym,
                //     cell_graph_zp,
                //     cell_graph_zm,
                //     [=](u32 id) {
                //         return acc.rho_vel[id];
                //     });
                
                auto result_vel = get_3d_grad<Tvec, Tvec, Minmod>(
                    old_cell_idx,
                    delta_cell,
                    cell_graph_xp,
                    cell_graph_xm,
                    cell_graph_yp,
                    cell_graph_ym,
                    cell_graph_zp,
                    cell_graph_zm,
                    [=](u32 id) {
                        return acc.vel[id];
                    });
                // logger::raw_ln("rhov_grad [x] ", old_cell_idx, result_rhov[0][0],
                // result_rhov[0][1], result_rhov[0][2],   "\n"); logger::raw_ln("rhov_grad [y] ",
                // old_cell_idx, result_rhov[1][0], result_rhov[1][1], result_rhov[1][2],   "\n");
                // logger::raw_ln("rhov_grad [z] ", old_cell_idx, result_rhov[2][0],
                // result_rhov[2][1], result_rhov[2][2],   "\n");

                shammath::ConsState<Tvec> mean_cons_var{0., 0, {0., 0., 0.}};
                Tscal rho_interpolate   = 0;
                Tvec rhovel_interpolate = {0., 0., 0.};
                Tscal rhoe_interpolate  = 0.0;
                Tvec vel_interpolate = {0., 0., 0.};
                Tscal press_interpolate  = 0.0;

                Tscal check_rho = 0, check_rhoe = 0;
                Tvec check_rhov  = {0., 0., 0.};

                Tscal rho_interp = 0, rhoe_interp = 0;
                Tvec rhov_interp = {0., 0., 0.};

                Tscal  press_interp = 0;
                Tvec v_interp = {0., 0., 0.};

                Tscal mean_rho = 0, mean_rhoe = 0;
                Tvec mean_rhov = {0., 0., 0.};

                for (u32 subdiv_lid = 0; subdiv_lid < 8; subdiv_lid++) {

                    auto [sx, sy, sz] = get_coord_ref(subdiv_lid);

                    // global coordinate in the patch for the (subdiv_lid + 1)-th child of
                    // old_cell_idx
                    std::array<u32, 3> glid = {lx * 2 + sx, ly * 2 + sy, lz * 2 + sz};

                    // global id in the patch for the (subdiv_lid + 1)-th child of old_cell_idx
                    u32 new_cell_idx = get_gid_write(glid);

                    // linear interpolation
                    Tscal rho_dx    = result_rho[0] * child_center_offsets[subdiv_lid][0];
                    // Tscal rhoe_dx   = result_rhoe[0] * child_center_offsets[subdiv_lid][0];
                    // Tvec rho_vel_dx = result_rhov[0] * child_center_offsets[subdiv_lid][0];

                    Tscal press_dx   = result_press[0] * child_center_offsets[subdiv_lid][0];
                    Tvec vel_dx = result_vel[0] * child_center_offsets[subdiv_lid][0];

                    Tscal rho_dy    = result_rho[1] * child_center_offsets[subdiv_lid][1];

                    // Tscal rhoe_dy   = result_rhoe[1] * child_center_offsets[subdiv_lid][1];
                    // Tvec rho_vel_dy = result_rhov[1] * child_center_offsets[subdiv_lid][1];


                    Tscal press_dy   = result_press[1] * child_center_offsets[subdiv_lid][1];
                    Tvec vel_dy = result_vel[1] * child_center_offsets[subdiv_lid][1];

                    Tscal rho_dz    = result_rho[2] * child_center_offsets[subdiv_lid][2];
                    // Tscal rhoe_dz   = result_rhoe[2] * child_center_offsets[subdiv_lid][2];
                    // Tvec rho_vel_dz = result_rhov[2] * child_center_offsets[subdiv_lid][2];

                    Tscal press_dz   = result_press[2] * child_center_offsets[subdiv_lid][2];
                    Tvec vel_dz = result_vel[2] * child_center_offsets[subdiv_lid][2];

                    rho_interp  = rho_block + (rho_dx + rho_dy + rho_dz);

                    // rhoe_interp = rhoE_block + (rhoe_dx + rhoe_dy + rhoe_dz);
                    // rhov_interp = rho_vel_block + (rho_vel_dx + rho_vel_dy + rho_vel_dz);

                    press_interp = press_block + (press_dx + press_dy + press_dz);
                    v_interp = vel_block + (vel_dx + vel_dy + vel_dz);

                    // logger::raw_ln("rho_interp ", rho_interp, rho_block, rho_dx, rho_dy, rho_dz);

                    // compute mean quatities
                    mean_rho += rho_interp;
                    mean_rhoe += rhoe_interp;
                    mean_rhov += rhov_interp;

                    // acc.rho[new_cell_idx]     = rho_block;
                    // acc.rho_vel[new_cell_idx] = rho_vel_block;
                    // acc.rhoE[new_cell_idx]    = rhoE_block;
                }

                // average
                mean_rho *= (1. / 8.);
                mean_rhoe *= (1. / 8.);
                mean_rhov *= (1. / 8.);

                // apply correction
                for (u32 subdiv_lid = 0; subdiv_lid < 8; subdiv_lid++) {

                    auto [sx, sy, sz] = get_coord_ref(subdiv_lid);

                    // global coordinate in the patch for the (subdiv_lid + 1)-th child of
                    // old_cell_idx
                    std::array<u32, 3> glid = {lx * 2 + sx, ly * 2 + sy, lz * 2 + sz};

                    // global id in the patch for the (subdiv_lid + 1)-th child of old_cell_idx
                    u32 new_cell_idx = get_gid_write(glid);

                    acc.rho[new_cell_idx]     = rho_interp + (rho_block - mean_rho);
                    acc.rho_vel[new_cell_idx] = rhov_interp + (rho_vel_block - mean_rhov);
                    acc.rhoE[new_cell_idx]    = rhoe_interp + (rhoe_interp - mean_rhoe);

                    // check_rho += rho_interp + (rho_block - mean_rho);
                    // check_rhov += rhov_interp + (rho_vel_block - mean_rhov);
                    // check_rhoe += rhoe_interp + (rhoE_block - mean_rhoe);
                }

                // logger::raw_ln("check mass conservation ", check_rho/8, rho_block);
                // logger::raw_ln("check total energy conservation ", check_rhoe/8, rhoE_block);
                // logger::raw_ln("check momemtum(x) conservation ", check_rhov[0]/8,
                // rho_vel_block[0]); logger::raw_ln("check momemtum(y) conservation ",
                // check_rhov[1]/8, rho_vel_block[1]); logger::raw_ln("check momemtum(z)
                // conservation ", check_rhov[2]/8, rho_vel_block[2]);

*/

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

    class RefineCritNormalizedSlopeAccessor {
        public:
        using AMRGraph             = shammodels::basegodunov::modules::AMRGraph;
        using Direction_           = shammodels::basegodunov::modules::Direction;
        using AMRGraphLinkiterator = shammodels::basegodunov::modules::AMRGraph::ro_access;

        const TgridVec *block_low_bound;
        const TgridVec *block_high_bound;
        const Tscal *block_rho;
        Tscal minimum_refine;

        u64 p_id;

        AMRGraphLinkiterator cell_graph_xp;
        AMRGraphLinkiterator cell_graph_xm;
        AMRGraphLinkiterator cell_graph_yp;
        AMRGraphLinkiterator cell_graph_ym;
        AMRGraphLinkiterator cell_graph_zp;
        AMRGraphLinkiterator cell_graph_zm;

        RefineCritNormalizedSlopeAccessor(
            sham::EventList &depend_list,
            Storage &storage,
            u64 &id_patch,
            shamrock::patch::Patch p,
            shamrock::patch::PatchDataLayer &pdat,
            Tscal min_refine)
            : minimum_refine(min_refine), cell_graph_xp(
                                              shambase::get_check_ref(storage.cell_graph_edge)
                                                  .get_refs_dir(Direction_::xp)
                                                  .get(id_patch)
                                                  .get()
                                                  .get_read_access(depend_list)),
              cell_graph_xm(
                  shambase::get_check_ref(storage.cell_graph_edge)
                      .get_refs_dir(Direction_::xm)
                      .get(id_patch)
                      .get()
                      .get_read_access(depend_list)),
              cell_graph_yp(
                  shambase::get_check_ref(storage.cell_graph_edge)
                      .get_refs_dir(Direction_::yp)
                      .get(id_patch)
                      .get()
                      .get_read_access(depend_list)),
              cell_graph_ym(
                  shambase::get_check_ref(storage.cell_graph_edge)
                      .get_refs_dir(Direction_::ym)
                      .get(id_patch)
                      .get()
                      .get_read_access(depend_list)),
              cell_graph_zp(
                  shambase::get_check_ref(storage.cell_graph_edge)
                      .get_refs_dir(Direction_::zp)
                      .get(id_patch)
                      .get()
                      .get_read_access(depend_list)),
              cell_graph_zm(
                  shambase::get_check_ref(storage.cell_graph_edge)
                      .get_refs_dir(Direction_::zm)
                      .get(id_patch)
                      .get()
                      .get_read_access(depend_list)) {
            block_low_bound  = pdat.get_field<TgridVec>(0).get_buf().get_read_access(depend_list);
            block_high_bound = pdat.get_field<TgridVec>(1).get_buf().get_read_access(depend_list);
            p_id             = id_patch;
            block_rho        = pdat.get_field<f64>(2).get_buf().get_write_access(depend_list);
        }

        void finalize(
            sham::EventList &resulting_events,
            Storage &storage,
            u64 &id_patch,
            shamrock::patch::Patch p,
            shamrock::patch::PatchDataLayer &pdat,
            Tscal min_refine) {
            pdat.get_field<i64_3>(0).get_buf().complete_event_state(resulting_events);
            pdat.get_field<i64_3>(1).get_buf().complete_event_state(resulting_events);
            pdat.get_field<f64>(2).get_buf().complete_event_state(resulting_events);

            shambase::get_check_ref(storage.cell_graph_edge)
                .get_refs_dir(Direction_::xp)
                .get(id_patch)
                .get()
                .complete_event_state(resulting_events);
            shambase::get_check_ref(storage.cell_graph_edge)
                .get_refs_dir(Direction_::xm)
                .get(id_patch)
                .get()
                .complete_event_state(resulting_events);
            shambase::get_check_ref(storage.cell_graph_edge)
                .get_refs_dir(Direction_::yp)
                .get(id_patch)
                .get()
                .complete_event_state(resulting_events);
            shambase::get_check_ref(storage.cell_graph_edge)
                .get_refs_dir(Direction_::ym)
                .get(id_patch)
                .get()
                .complete_event_state(resulting_events);
            shambase::get_check_ref(storage.cell_graph_edge)
                .get_refs_dir(Direction_::zp)
                .get(id_patch)
                .get()
                .complete_event_state(resulting_events);
            shambase::get_check_ref(storage.cell_graph_edge)
                .get_refs_dir(Direction_::zm)
                .get(id_patch)
                .get()
                .complete_event_state(resulting_events);
        }

        void refine_criterion(
            u32 block_id,
            RefineCritNormalizedSlopeAccessor acc,
            bool &should_refine,
            bool &should_derefine) const {

            TgridVec low_bound  = acc.block_low_bound[block_id];
            TgridVec high_bound = acc.block_high_bound[block_id];

            auto get_normalized_slope_dir = [&](auto &cell_graph_links_left,
                                                auto &cell_graph_links_right,
                                                u32 cell_global_id) -> Tscal {
                Tscal center_field      = block_rho[cell_global_id];
                Tscal left_neigh_field  = shambase::VectorProperties<Tscal>::get_zero();
                Tscal right_neigh_field = shambase::VectorProperties<Tscal>::get_zero();
                cell_graph_links_left.for_each_object_link(cell_global_id, [&](u32 neigh_id) {
                    left_neigh_field += block_rho[neigh_id];
                });
                cell_graph_links_right.for_each_object_link(cell_global_id, [&](u32 neigh_id) {
                    right_neigh_field += block_rho[neigh_id];
                });
                Tscal res = sham::details::g_sycl_abs(
                    (right_neigh_field - left_neigh_field)
                    / (2.0 * sham::details::g_sycl_max(center_field, 1e-7)));
                return res;
            };

            auto get_normalized_slope = [&](u32 cell_global_id) -> Tscal {
                Tscal res = shambase::VectorProperties<Tscal>::get_zero();
                res       = sham::details::g_sycl_max(
                    res, get_normalized_slope_dir(cell_graph_xm, cell_graph_xp, cell_global_id));
                res = sham::details::g_sycl_max(
                    res, get_normalized_slope_dir(cell_graph_ym, cell_graph_yp, cell_global_id));
                res = sham::details::g_sycl_max(
                    res, get_normalized_slope_dir(cell_graph_zm, cell_graph_zp, cell_global_id));
                return res;
            };

            Tscal block_norm_slope = shambase::VectorProperties<Tscal>::get_zero();
            for (u32 i = 0; i < AMRBlock::block_size; i++) {
                block_norm_slope = sham::details::g_sycl_max(
                    block_norm_slope, get_normalized_slope(i + block_id * AMRBlock::block_size));
            }

            should_refine   = false;
            should_derefine = false;
            if (block_norm_slope > minimum_refine) {
                should_refine = true;
            }

            should_refine = should_refine && (high_bound.x() - low_bound.x() > AMRBlock::Nside);
            should_refine = should_refine && (high_bound.y() - low_bound.y() > AMRBlock::Nside);
            should_refine = should_refine && (high_bound.z() - low_bound.z() > AMRBlock::Nside);
        }
    };

    class RefineCritPseudoGradientAccessor {
        public:
        using AMRGraph             = shammodels::basegodunov::modules::AMRGraph;
        using Direction_           = shammodels::basegodunov::modules::Direction;
        using AMRGraphLinkiterator = shammodels::basegodunov::modules::AMRGraph::ro_access;

        Tscal one_over_Nside = 1. / AMRBlock::Nside;
        const TgridVec *block_low_bound;
        const TgridVec *block_high_bound;
        const Tscal *block_rho;
        const f64 *block_pressure;
    
        Tscal error_min;
        Tscal error_max;
        u64 p_id;

        AMRGraphLinkiterator cell_graph_xp;
        AMRGraphLinkiterator cell_graph_xm;
        AMRGraphLinkiterator cell_graph_yp;
        AMRGraphLinkiterator cell_graph_ym;
        AMRGraphLinkiterator cell_graph_zp;
        AMRGraphLinkiterator cell_graph_zm;

        RefineCritPseudoGradientAccessor(
            sham::EventList &depend_list,
            Storage &storage,
            u64 &id_patch,
            shamrock::patch::Patch p,
            shamrock::patch::PatchDataLayer &pdat,
            Tscal err_min,
            Tscal err_max)
            :  error_min(err_min), error_max(err_max),
              cell_graph_xp(
                  shambase::get_check_ref(storage.cell_graph_edge)
                      .get_refs_dir(Direction_::xp)
                      .get(id_patch)
                      .get()
                      .get_read_access(depend_list)),
              cell_graph_xm(
                  shambase::get_check_ref(storage.cell_graph_edge)
                      .get_refs_dir(Direction_::xm)
                      .get(id_patch)
                      .get()
                      .get_read_access(depend_list)),
              cell_graph_yp(
                  shambase::get_check_ref(storage.cell_graph_edge)
                      .get_refs_dir(Direction_::yp)
                      .get(id_patch)
                      .get()
                      .get_read_access(depend_list)),
              cell_graph_ym(
                  shambase::get_check_ref(storage.cell_graph_edge)
                      .get_refs_dir(Direction_::ym)
                      .get(id_patch)
                      .get()
                      .get_read_access(depend_list)),
              cell_graph_zp(
                  shambase::get_check_ref(storage.cell_graph_edge)
                      .get_refs_dir(Direction_::zp)
                      .get(id_patch)
                      .get()
                      .get_read_access(depend_list)),
              cell_graph_zm(
                  shambase::get_check_ref(storage.cell_graph_edge)
                      .get_refs_dir(Direction_::zm)
                      .get(id_patch)
                      .get()
                      .get_read_access(depend_list)) {
            block_low_bound  = pdat.get_field<TgridVec>(0).get_buf().get_read_access(depend_list);
            block_high_bound = pdat.get_field<TgridVec>(1).get_buf().get_read_access(depend_list);
            p_id             = id_patch;
            block_rho        = pdat.get_field<f64>(2).get_buf().get_write_access(depend_list);
            // block_rho_vel    = pdat.get_field<f64_3>(3).get_buf().get_write_access(depend_list);
            // block_rhoE       = pdat.get_field<f64>(4).get_buf().get_write_access(depend_list);
             block_pressure   = shambase::get_check_ref(storage.press)
                                 .get_buf(id_patch)
                                 .get_read_access(depend_list);
        }

        void finalize(
            sham::EventList &resulting_events,
            Storage &storage,
            u64 &id_patch,
            shamrock::patch::Patch p,
            shamrock::patch::PatchDataLayer &pdat,
            Tscal err_min,
            Tscal err_max) {
            pdat.get_field<i64_3>(0).get_buf().complete_event_state(resulting_events);
            pdat.get_field<i64_3>(1).get_buf().complete_event_state(resulting_events);
            pdat.get_field<f64>(2).get_buf().complete_event_state(resulting_events);
            // pdat.get_field<f64_3>(3).get_buf().complete_event_state(resulting_events);
            // pdat.get_field<f64>(4).get_buf().complete_event_state(resulting_events);

            shambase::get_check_ref(storage.cell_graph_edge)
                .get_refs_dir(Direction_::xp)
                .get(id_patch)
                .get()
                .complete_event_state(resulting_events);
            shambase::get_check_ref(storage.cell_graph_edge)
                .get_refs_dir(Direction_::xm)
                .get(id_patch)
                .get()
                .complete_event_state(resulting_events);
            shambase::get_check_ref(storage.cell_graph_edge)
                .get_refs_dir(Direction_::yp)
                .get(id_patch)
                .get()
                .complete_event_state(resulting_events);
            shambase::get_check_ref(storage.cell_graph_edge)
                .get_refs_dir(Direction_::ym)
                .get(id_patch)
                .get()
                .complete_event_state(resulting_events);
            shambase::get_check_ref(storage.cell_graph_edge)
                .get_refs_dir(Direction_::zp)
                .get(id_patch)
                .get()
                .complete_event_state(resulting_events);
            shambase::get_check_ref(storage.cell_graph_edge)
                .get_refs_dir(Direction_::zm)
                .get(id_patch)
                .get()
                .complete_event_state(resulting_events);

            
            shambase::get_check_ref(storage.press)
                .get_buf(id_patch)
                .complete_event_state(resulting_events);
        }

        void refine_criterion(
            u32 block_id,
            RefineCritPseudoGradientAccessor acc,
            bool &should_refine,
            bool &should_derefine) const {
            TgridVec low_bound  = acc.block_low_bound[block_id];
            TgridVec high_bound = acc.block_high_bound[block_id];

            // Tvec lower_flt = low_bound.template convert<Tscal>() * dxfact;
            // Tvec upper_flt = high_bound.template convert<Tscal>() * dxfact;

            // Tvec block_cell_size = (upper_flt - lower_flt) * one_over_Nside;

            Tscal block_rho_grad = shambase::VectorProperties<Tscal>::get_zero();
            for (u32 i = 0; i < AMRBlock::block_size; i++) {
                block_rho_grad = sham::details::g_sycl_max(
                    block_rho_grad,
                    get_pseudo_grad<Tscal, Tvec>(
                        i + block_id * AMRBlock::block_size,
                        cell_graph_xp,
                        cell_graph_xm,
                        cell_graph_yp,
                        cell_graph_ym,
                        cell_graph_zp,
                        cell_graph_zm,
                        [=](u32 id) {
                            return block_rho[id];
                        }));
            }
            // block_rho_grad *= block_cell_size.x() * block_cell_size.y() * block_cell_size.z();

            Tscal block_press_grad = shambase::VectorProperties<Tscal>::get_zero();
            for (u32 i = 0; i < AMRBlock::block_size; i++) {
                    block_press_grad = sham::details::g_sycl_max(
                    block_press_grad,
                    get_pseudo_grad<Tscal, Tvec>(
                        i + block_id * AMRBlock::block_size,
                        cell_graph_xp,
                        cell_graph_xm,
                        cell_graph_yp,
                        cell_graph_ym,
                        cell_graph_zp,
                        cell_graph_zm,
                        [=](u32 id) {
                            return block_pressure[id];
                        }));
            }

        Tscal error = sham::details::g_sycl_max(block_rho_grad, block_press_grad);
            should_refine   = false;
            should_derefine = false;
            if (error > error_max) {
                should_refine   = true;
                // should_derefine = false;
            } else if (error < 0.5 * error_max) {
                should_refine   = false;
                // should_derefine = true;
            }

            should_refine = should_refine && (high_bound.x() - low_bound.x() > AMRBlock::Nside);
            should_refine = should_refine && (high_bound.y() - low_bound.y() > AMRBlock::Nside);
            should_refine = should_refine && (high_bound.z() - low_bound.z() > AMRBlock::Nside);
        }
    };

    class RefineCritSecondOrderDerivativeAccessor {
        public:
        using AMRGraph             = shammodels::basegodunov::modules::AMRGraph;
        using Direction_           = shammodels::basegodunov::modules::Direction;
        using AMRGraphLinkiterator = shammodels::basegodunov::modules::AMRGraph::ro_access;

        Tscal one_over_Nside = 1. / AMRBlock::Nside;
        // Tscal dxfact;
        const TgridVec *block_low_bound;
        const TgridVec *block_high_bound;
        const Tscal *block_rho;
        const f64 *block_pressure;
        Tscal error_min;
        Tscal error_max;
        u64 p_id;

        AMRGraphLinkiterator cell_graph_xp;
        AMRGraphLinkiterator cell_graph_xm;
        AMRGraphLinkiterator cell_graph_yp;
        AMRGraphLinkiterator cell_graph_ym;
        AMRGraphLinkiterator cell_graph_zp;
        AMRGraphLinkiterator cell_graph_zm;

        RefineCritSecondOrderDerivativeAccessor(
            sham::EventList &depend_list,
            Storage &storage,
            u64 &id_patch,
            shamrock::patch::Patch p,
            shamrock::patch::PatchDataLayer &pdat,
            Tscal err_min,
            Tscal err_max)
            : error_min(err_min), error_max(err_max),
              cell_graph_xp(
                  shambase::get_check_ref(storage.cell_graph_edge)
                      .get_refs_dir(Direction_::xp)
                      .get(id_patch)
                      .get()
                      .get_read_access(depend_list)),
              cell_graph_xm(
                  shambase::get_check_ref(storage.cell_graph_edge)
                      .get_refs_dir(Direction_::xm)
                      .get(id_patch)
                      .get()
                      .get_read_access(depend_list)),
              cell_graph_yp(
                  shambase::get_check_ref(storage.cell_graph_edge)
                      .get_refs_dir(Direction_::yp)
                      .get(id_patch)
                      .get()
                      .get_read_access(depend_list)),
              cell_graph_ym(
                  shambase::get_check_ref(storage.cell_graph_edge)
                      .get_refs_dir(Direction_::ym)
                      .get(id_patch)
                      .get()
                      .get_read_access(depend_list)),
              cell_graph_zp(
                  shambase::get_check_ref(storage.cell_graph_edge)
                      .get_refs_dir(Direction_::zp)
                      .get(id_patch)
                      .get()
                      .get_read_access(depend_list)),
              cell_graph_zm(
                  shambase::get_check_ref(storage.cell_graph_edge)
                      .get_refs_dir(Direction_::zm)
                      .get(id_patch)
                      .get()
                      .get_read_access(depend_list)) {
            block_low_bound  = pdat.get_field<TgridVec>(0).get_buf().get_read_access(depend_list);
            block_high_bound = pdat.get_field<TgridVec>(1).get_buf().get_read_access(depend_list);
            p_id             = id_patch;
            block_rho        = pdat.get_field<f64>(2).get_buf().get_write_access(depend_list);
            block_pressure   = shambase::get_check_ref(storage.press)
                                 .get_buf(id_patch)
                                 .get_read_access(depend_list);
        }

        void finalize(
            sham::EventList &resulting_events,
            Storage &storage,
            u64 &id_patch,
            shamrock::patch::Patch p,
            shamrock::patch::PatchDataLayer &pdat,
            Tscal err_min,
            Tscal err_max) {
            pdat.get_field<i64_3>(0).get_buf().complete_event_state(resulting_events);
            pdat.get_field<i64_3>(1).get_buf().complete_event_state(resulting_events);
            pdat.get_field<f64>(2).get_buf().complete_event_state(resulting_events);

            shambase::get_check_ref(storage.cell_graph_edge)
                .get_refs_dir(Direction_::xp)
                .get(id_patch)
                .get()
                .complete_event_state(resulting_events);
            shambase::get_check_ref(storage.cell_graph_edge)
                .get_refs_dir(Direction_::xm)
                .get(id_patch)
                .get()
                .complete_event_state(resulting_events);
            shambase::get_check_ref(storage.cell_graph_edge)
                .get_refs_dir(Direction_::yp)
                .get(id_patch)
                .get()
                .complete_event_state(resulting_events);
            shambase::get_check_ref(storage.cell_graph_edge)
                .get_refs_dir(Direction_::ym)
                .get(id_patch)
                .get()
                .complete_event_state(resulting_events);
            shambase::get_check_ref(storage.cell_graph_edge)
                .get_refs_dir(Direction_::zp)
                .get(id_patch)
                .get()
                .complete_event_state(resulting_events);
            shambase::get_check_ref(storage.cell_graph_edge)
                .get_refs_dir(Direction_::zm)
                .get(id_patch)
                .get()
                .complete_event_state(resulting_events);

            shambase::get_check_ref(storage.press)
                .get_buf(id_patch)
                .complete_event_state(resulting_events);
        }

        void refine_criterion(
            u32 block_id,
            RefineCritSecondOrderDerivativeAccessor acc,
            bool &should_refine,
            bool &should_derefine) const {
            TgridVec low_bound  = acc.block_low_bound[block_id];
            TgridVec high_bound = acc.block_high_bound[block_id];
            // Tvec lower_flt = low_bound.template convert<Tscal>() * dxfact;
            // Tvec upper_flt = high_bound.template convert<Tscal>() * dxfact;

            // Tvec block_cell_size = (upper_flt - lower_flt) * one_over_Nside;

            /*
                    Tscal block_rho_grad = shambase::VectorProperties<Tscal>::get_zero();
                    for (u32 i = 0; i < AMRBlock::block_size; i++) {
                        block_rho_grad = sham::details::g_sycl_max(
                            block_rho_grad,
                            modif_second_derivative<Tscal, Tvec>(
                                i + block_id * AMRBlock::block_size,
                                cell_graph_xp,
                                cell_graph_xm,
                                cell_graph_yp,
                                cell_graph_ym,
                                cell_graph_zp,
                                cell_graph_zm,
                                [=](u32 id) {
                                    return block_rho[id];
                                }));
                    }
                    block_rho_grad *=block_cell_size.x() * block_cell_size.y() *
               block_cell_size.z();

                    Tscal block_press_grad = shambase::VectorProperties<Tscal>::get_zero();
                    for(u32 i = 0; i < AMRBlock::block_size; i++){
                        block_press_grad = sham::details::g_sycl_max(block_press_grad,
               modif_second_derivative<Tscal, Tvec>( i + block_id * AMRBlock::block_size,
                                cell_graph_xp,
                                cell_graph_xm,
                                cell_graph_yp,
                                cell_graph_ym,
                                cell_graph_zp,
                                cell_graph_zm,
                                [=](u32 id) {
                                    return block_pressure[id];
                                }));
                    }
                    */

            Tscal block_rho_grad = shambase::VectorProperties<Tscal>::get_zero();
            for (u32 i = 0; i < AMRBlock::block_size; i++) {
                block_rho_grad = sham::details::g_sycl_max(
                    block_rho_grad,
                    get_Lohner<Tscal, Tvec>(
                        i + block_id * AMRBlock::block_size,
                        cell_graph_xp,
                        cell_graph_xm,
                        cell_graph_yp,
                        cell_graph_ym,
                        cell_graph_zp,
                        cell_graph_zm,
                        [=](u32 id) {
                            return block_rho[id];
                        }));
            }
            // block_rho_grad *=block_cell_size.x() * block_cell_size.y() * block_cell_size.z();

            Tscal block_press_grad = shambase::VectorProperties<Tscal>::get_zero();
            for (u32 i = 0; i < AMRBlock::block_size; i++) {
                block_press_grad = sham::details::g_sycl_max(
                    block_press_grad,
                    get_Lohner<Tscal, Tvec>(
                        i + block_id * AMRBlock::block_size,
                        cell_graph_xp,
                        cell_graph_xm,
                        cell_graph_yp,
                        cell_graph_ym,
                        cell_graph_zp,
                        cell_graph_zm,
                        [=](u32 id) {
                            return block_pressure[id];
                        }));
            }

            should_refine   = false;
            should_derefine = false;
            if (sham::details::g_sycl_max(block_press_grad, block_rho_grad)
                > error_max * error_max) {
                should_refine   = true;
                // should_derefine = false;
            } else if (
                sham::details::g_sycl_max(block_press_grad, block_rho_grad)
                < 0.25 * error_max * error_max) {
                should_refine   = false;
                // should_derefine = true;
            }

            should_refine = should_refine && (high_bound.x() - low_bound.x() > AMRBlock::Nside);
            should_refine = should_refine && (high_bound.y() - low_bound.y() > AMRBlock::Nside);
            should_refine = should_refine && (high_bound.z() - low_bound.z() > AMRBlock::Nside);
        }
    };

    // Ensure that the blocks are sorted before refinement
    AMRSortBlocks block_sorter(context, solver_config, storage);
    block_sorter.reorder_amr_blocks();

    using AMRmode_None                = typename AMRMode<Tvec, TgridVec>::None;
    using AMRmode_DensityBased        = typename AMRMode<Tvec, TgridVec>::DensityBased;
    using AMRmode_SlopeBased          = typename AMRMode<Tvec, TgridVec>::SlopeBased;
    using AMRmode_PseudoGradientBased = typename AMRMode<Tvec, TgridVec>::PseudoGradientBased;
    using AMRmode_SecondOrderDerivative =
        typename AMRMode<Tvec, TgridVec>::SecondOrderDerivativeBased;

    if (AMRmode_None *cfg = std::get_if<AMRmode_None>(&solver_config.amr_mode.config)) {
        // no refinment here turn around there is nothing to see
    } else if (
        AMRmode_DensityBased *cfg
        = std::get_if<AMRmode_DensityBased>(&solver_config.amr_mode.config)) {
        Tscal dxfact(solver_config.grid_coord_to_pos_fact);

        // get refine and derefine list
        shambase::DistributedData<OptIndexList> refine_list;
        shambase::DistributedData<OptIndexList> derefine_list;

        gen_refine_block_changes<RefineCritBlock>(
            refine_list, derefine_list, dxfact, cfg->crit_mass);

        //////// apply refine ////////
        // Note that this only add new blocks at the end of the patchdata
        internal_refine_grid<RefineCellAccessor>(std::move(refine_list));

        //////// apply derefine ////////
        // Note that this will perform the merge then remove the old blocks
        // This is ok to call straight after the refine without edditing the index list in
        // derefine_list since no permutations were applied in internal_refine_grid and no cells can
        // be both refined and derefined in the same pass
        internal_derefine_grid<RefineCellAccessor>(std::move(derefine_list));
    } else if (
        AMRmode_SlopeBased *cfg = std::get_if<AMRmode_SlopeBased>(&solver_config.amr_mode.config)) {

        // get refine and derefine list
        shambase::DistributedData<OptIndexList> refine_list;
        shambase::DistributedData<OptIndexList> derefine_list;
        gen_refine_block_changes<RefineCritNormalizedSlopeAccessor>(
            refine_list, derefine_list, cfg->crit_smooth);
        //////// apply refine ////////
        // Note that this only add new blocks at the end of the patchdata
        internal_refine_grid<RefineCellAccessor>(std::move(refine_list));

        //////// apply derefine ////////
        // Note that this will perform the merge then remove the old blocks
        // This is ok to call straight after the refine without edditing the index list in
        // derefine_list since no permutations were applied in internal_refine_grid and no cells can
        // be both refined and derefined in the same pass
        internal_derefine_grid<RefineCellAccessor>(std::move(derefine_list));

    } else if (
        AMRmode_PseudoGradientBased *cfg
        = std::get_if<AMRmode_PseudoGradientBased>(&solver_config.amr_mode.config)) {
        Tscal dxfact(solver_config.grid_coord_to_pos_fact);
        // get refine and derefine list
        shambase::DistributedData<OptIndexList> refine_list;
        shambase::DistributedData<OptIndexList> derefine_list;
        gen_refine_block_changes<RefineCritPseudoGradientAccessor>(
            refine_list, derefine_list, cfg->error_min, cfg->error_max);
        //////// apply refine ////////
        // Note that this only add new blocks at the end of the patchdata
        internal_refine_grid<RefineCellAccessor>(std::move(refine_list));

        //////// apply derefine ////////
        // Note that this will perform the merge then remove the old blocks
        // This is ok to call straight after the refine without edditing the index list in
        // derefine_list since no permutations were applied in internal_refine_grid and no cells can
        // be both refined and derefined in the same pass
        internal_derefine_grid<RefineCellAccessor>(std::move(derefine_list));

    }

    else if (
        AMRmode_SecondOrderDerivative *cfg
        = std::get_if<AMRmode_SecondOrderDerivative>(&solver_config.amr_mode.config)) {

        // get refine and derefine list
        shambase::DistributedData<OptIndexList> refine_list;
        shambase::DistributedData<OptIndexList> derefine_list;
        gen_refine_block_changes<RefineCritSecondOrderDerivativeAccessor>(
            refine_list, derefine_list, cfg->crit_min, cfg->crit_max);
        //////// apply refine ////////
        // Note that this only add new blocks at the end of the patchdata
        internal_refine_grid<RefineCellAccessor>(std::move(refine_list));

        //////// apply derefine ////////
        // Note that this will perform the merge then remove the old blocks
        // This is ok to call straight after the refine without edditing the index list in
        // derefine_list since no permutations were applied in internal_refine_grid and no cells can
        // be both refined and derefined in the same pass
        internal_derefine_grid<RefineCellAccessor>(std::move(derefine_list));

    }

    else {
        throw std::invalid_argument("Unsupported AMRMode ");
    }
}

template class shammodels::basegodunov::modules::AMRGridRefinementHandler<f64_3, i64_3>;
