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

template<class Tvec, class TgridVec>
template<class UserAcc, class Fct>
void shammodels::basegodunov::modules::AMRGridRefinementHandler<Tvec, TgridVec>::
    internal_refine_grid(shambase::DistributedData<OptIndexList> refine_list, Fct &&f) {

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

    logger::info_ln("AMRGrid", "process cell count =", sum_block_count);
}

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::AMRGridRefinementHandler<Tvec, TgridVec>::refine_grid(
    shambase::DistributedData<OptIndexList> refine_list) {}

template class shammodels::basegodunov::modules::AMRGridRefinementHandler<f64_3, i64_3>;
