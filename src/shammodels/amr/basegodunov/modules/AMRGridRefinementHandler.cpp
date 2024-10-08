// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file AMRGridRefinementHandler.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shammodels/amr/basegodunov/modules/AMRGridRefinementHandler.hpp"
#include "shammodels/amr/basegodunov/modules/AMRSortBlocks.hpp"

template<class Tvec, class TgridVec>
template<class UserAcc, class Fct, class... T>
void shammodels::basegodunov::modules::AMRGridRefinementHandler<Tvec, TgridVec>::
    gen_refine_block_changes(
        shambase::DistributedData<OptIndexList> &refine_list,
        shambase::DistributedData<OptIndexList> &derefine_list,
        Fct &&flag_refine_derefine_functor,
        T &&...args) {

    using namespace shamrock::patch;

    u64 tot_refine   = 0;
    u64 tot_derefine = 0;

    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
        sycl::queue &q = shamsys::instance::get_compute_queue();

        u64 id_patch = cur_p.id_patch;

        // create the refine and derefine flags buffers
        u32 obj_cnt = pdat.get_obj_cnt();

        sycl::buffer<u32> refine_flags(obj_cnt);
        sycl::buffer<u32> derefine_flags(obj_cnt);

        // fill in the flags
        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
            sycl::accessor refine_acc{refine_flags, cgh, sycl::write_only, sycl::no_init};
            sycl::accessor derefine_acc{derefine_flags, cgh, sycl::write_only, sycl::no_init};

            UserAcc uacc(cgh, id_patch, cur_p, pdat, args...);

            cgh.parallel_for(sycl::range<1>(obj_cnt), [=](sycl::item<1> gid) {
                bool flag_refine   = false;
                bool flag_derefine = false;
                flag_refine_derefine_functor(gid.get_linear_id(), uacc, flag_refine, flag_derefine);

                // This is just a safe guard to avoid this nonsensicall case
                if (flag_refine && flag_derefine) {
                    flag_derefine = false;
                }

                refine_acc[gid]   = (flag_refine) ? 1 : 0;
                derefine_acc[gid] = (flag_derefine) ? 1 : 0;
            });
        });

        // keep only derefine flags on only if the eight cells want to merge and if they can
        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
            sycl::accessor acc_min{*pdat.get_field<TgridVec>(0).get_buf(), cgh, sycl::read_only};
            sycl::accessor acc_max{*pdat.get_field<TgridVec>(1).get_buf(), cgh, sycl::read_only};

            sycl::accessor acc_merge_flag{derefine_flags, cgh, sycl::read_write};

            cgh.parallel_for(sycl::range<1>(obj_cnt), [=](sycl::item<1> gid) {
                u32 id = gid.get_linear_id();

                std::array<BlockCoord, split_count> blocks;

                bool all_want_to_merge = true;
                for (u32 lid = 0; lid < split_count; lid++) {
                    blocks[lid]       = BlockCoord{acc_min[gid + lid], acc_max[gid + lid]};
                    all_want_to_merge = all_want_to_merge && acc_merge_flag[gid + lid];
                }

                acc_merge_flag[gid] = all_want_to_merge && BlockCoord::are_mergeable(blocks);
            });
        });

        ////////////////////////////////////////////////////////////////////////////////
        // refinement
        ////////////////////////////////////////////////////////////////////////////////

        // perform stream compactions on the refinement flags
        auto [buf_refine, len_refine] = shamalgs::numeric::stream_compact(q, refine_flags, obj_cnt);

        logger::debug_ln("AMRGrid", "patch ", id_patch, "refine block count = ", len_refine);

        tot_refine += len_refine;

        // add the results to the map
        refine_list.add_obj(id_patch, OptIndexList{std::move(buf_refine), len_refine});

        ////////////////////////////////////////////////////////////////////////////////
        // derefinement
        ////////////////////////////////////////////////////////////////////////////////

        // perform stream compactions on the derefinement flags
        auto [buf_derefine, len_derefine]
            = shamalgs::numeric::stream_compact(q, derefine_flags, obj_cnt);

        logger::debug_ln("AMRGrid", "patch ", id_patch, "merge block count = ", len_derefine);

        tot_derefine += len_derefine;

        // add the results to the map
        derefine_list.add_obj(id_patch, OptIndexList{std::move(buf_derefine), len_derefine});
    });

    logger::info_ln("AMRGrid", "on this process", tot_refine, "blocks were refined");
    logger::info_ln(
        "AMRGrid", "on this process", tot_derefine * split_count, "blocks were derefined");
}

template<class Tvec, class TgridVec>
auto shammodels::basegodunov::modules::AMRGridRefinementHandler<Tvec, TgridVec>::update_refinement()
    -> CellToUpdate {

    // Ensure that the blocks are sorted before refinement
    AMRSortBlocks block_sorter(context, solver_config, storage);
    block_sorter.reorder_amr_blocks();

    class RefineCritBlockAccessor {
        public:
        sycl::accessor<u64_3, 1, sycl::access::mode::read, sycl::target::device> block_low_bound;
        sycl::accessor<u64_3, 1, sycl::access::mode::read, sycl::target::device> block_high_bound;

        RefineCritBlockAccessor(
            sycl::handler &cgh,
            u64 id_patch,
            shamrock::patch::Patch p,
            shamrock::patch::PatchData &pdat)
            : block_low_bound{*pdat.get_field<u64_3>(0).get_buf(), cgh, sycl::read_only},
              block_high_bound{*pdat.get_field<u64_3>(1).get_buf(), cgh, sycl::read_only} {}
    };

    // get refine and derefine list
    shambase::DistributedData<OptIndexList> refine_list;
    shambase::DistributedData<OptIndexList> derefine_list;

    gen_refine_block_changes<RefineCritBlockAccessor>(
        refine_list,
        derefine_list,
        [](u32 block_id, RefineCritBlockAccessor acc, bool &should_refine, bool &should_derefine) {
            u64_3 low_bound  = acc.block_low_bound[block_id];
            u64_3 high_bound = acc.block_high_bound[block_id];

            u64_3 block_size = high_bound - low_bound;
            u64 block_sz     = block_size.x();

            // refine based on x position
            u64 wanted_sz = block_sz;
            if (low_bound[0] < 4)
                wanted_sz = 4;
            if (low_bound[0] < 8)
                wanted_sz = 8;
            if (low_bound[0] < 16)
                wanted_sz = 16;

            should_refine   = (block_sz > 1) && (block_sz > wanted_sz);
            should_derefine = (block_sz < wanted_sz);
        });

    class RefineCellAccessor {
        public:
        sycl::accessor<u32, 1, sycl::access::mode::read_write, sycl::target::device> field;

        RefineCellAccessor(sycl::handler &cgh, shamrock::patch::PatchData &pdat)
            : field{*pdat.get_field<u32>(2).get_buf(), cgh, sycl::read_write} {}
    };

    internal_refine_grid<RefineCellAccessor>(
        std::move(refine_list),

        [](u32 cur_idx,
           BlockCoord cur_coords,
           std::array<u32, 8> new_cells,
           std::array<BlockCoord, 8> new_cells_coords,
           RefineCellAccessor acc) {
            u32 val = acc.field[cur_idx];

#pragma unroll
            for (u32 pid = 0; pid < 8; pid++) {
                acc.field[new_cells[pid]] = val;
            }
        }

    );

    internal_derefine_grid<RefineCellAccessor>(
        std::move(derefine_list),

        [](std::array<u32, 8> old_cells,
           std::array<BlockCoord, 8> old_coords,
           u32 new_cell,
           BlockCoord new_coord,

           RefineCellAccessor acc) {
            u32 accum = 0;

#pragma unroll
            for (u32 pid = 0; pid < 8; pid++) {
                accum += acc.field[old_cells[pid]];
            }

            acc.field[new_cell] = accum / 8;
        }

    );

    return {};
}

template<class Tvec, class TgridVec>
template<class UserAcc, class Fct>
void shammodels::basegodunov::modules::AMRGridRefinementHandler<Tvec, TgridVec>::
    internal_refine_grid(shambase::DistributedData<OptIndexList> &&refine_list, Fct &&lambd) {

    using namespace shamrock::patch;

    u64 sum_block_count = 0;

    scheduler().for_each_patch_data([&](u64 id_patch, Patch cur_p, PatchData &pdat) {
        sycl::queue &q = shamsys::instance::get_compute_queue();

        u32 old_obj_cnt = pdat.get_obj_cnt();

        OptIndexList &refine_flags = refine_list.get(id_patch);

        if (refine_flags.count > 0) {

            // alloc memory for the new blocks to be created
            pdat.expand(refine_flags.count * (split_count - 1));

            // Refine the block (set the positions) and fill the corresponding fields
            shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
                sycl::accessor index_to_ref{*refine_flags.idx, cgh, sycl::read_only};

                sycl::accessor block_bound_low{
                    *pdat.get_field<TgridVec>(0).get_buf(), cgh, sycl::read_write};
                sycl::accessor block_bound_high{
                    *pdat.get_field<TgridVec>(1).get_buf(), cgh, sycl::read_write};

                u32 start_index_push = old_obj_cnt;

                constexpr u32 new_splits = split_count - 1;

                UserAcc uacc(cgh, pdat);

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
                    lambd(idx_to_refine, cur_block, blocks_ids, block_coords, uacc);
                });
            });
        }

        sum_block_count += pdat.get_obj_cnt();
    });

    logger::info_ln("AMRGrid", "process block count =", sum_block_count);
}

template<class Tvec, class TgridVec>
template<class UserAcc, class Fct>
void shammodels::basegodunov::modules::AMRGridRefinementHandler<Tvec, TgridVec>::
    internal_derefine_grid(shambase::DistributedData<OptIndexList> &&derefine_list, Fct &&lambd) {

    using namespace shamrock::patch;

    scheduler().for_each_patch_data([&](u64 id_patch, Patch cur_p, PatchData &pdat) {
        sycl::queue &q = shamsys::instance::get_compute_queue();

        u32 old_obj_cnt = pdat.get_obj_cnt();

        OptIndexList &derefine_flags = derefine_list.get(id_patch);

        if (derefine_flags.count > 0) {

            // init flag table
            sycl::buffer<u32> keep_block_flag
                = shamalgs::algorithm::gen_buffer_device(q, old_obj_cnt, [](u32 i) -> u32 {
                      return 1;
                  });

            // edit block content + make flag of blocks to keep
            shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
                sycl::accessor index_to_deref{*derefine_flags.idx, cgh, sycl::read_only};

                sycl::accessor block_bound_low{
                    *pdat.get_field<TgridVec>(0).get_buf(), cgh, sycl::read_write};
                sycl::accessor block_bound_high{
                    *pdat.get_field<TgridVec>(1).get_buf(), cgh, sycl::read_write};

                sycl::accessor flag_keep{keep_block_flag, cgh, sycl::read_write};

                UserAcc uacc(cgh, pdat);

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
                    lambd(old_indexes, block_coords, idx_to_derefine, merged_block_coord, uacc);
                });
            });

            // stream compact the flags
            auto [opt_buf, len] = shamalgs::numeric::stream_compact(
                shamsys::instance::get_compute_queue(), keep_block_flag, old_obj_cnt);

            logger::debug_ln(
                "AMR Grid", "patch", id_patch, "derefine block count ", old_obj_cnt, "->", len);

            if (!opt_buf) {
                throw std::runtime_error("opt buf must contain something at this point");
            }

            // remap pdat according to stream compact
            pdat.index_remap_resize(*opt_buf, len);
        }
    });
}

template class shammodels::basegodunov::modules::AMRGridRefinementHandler<f64_3, i64_3>;
