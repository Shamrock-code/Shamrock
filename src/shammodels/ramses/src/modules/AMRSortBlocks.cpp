// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file AMRSortBlocks.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shammodels/ramses/modules/AMRSortBlocks.hpp"

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::AMRSortBlocks<Tvec, TgridVec>::reorder_amr_blocks() {

    using MortonBuilder = RadixTreeMortonBuilder<u64, TgridVec, 3>;
    using namespace shamrock::patch;

    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
        std::unique_ptr<sycl::buffer<u64>> out_buf_morton;
        std::unique_ptr<sycl::buffer<u32>> out_buf_particle_index_map;

        MortonBuilder::build(
            shamsys::instance::get_compute_scheduler_ptr(),
            scheduler().get_sim_box().template patch_coord_to_domain<TgridVec>(cur_p),
            pdat.get_field<TgridVec>(0).get_buf(),
            pdat.get_obj_cnt(),
            out_buf_morton,
            out_buf_particle_index_map);

        // apply list permut on patch

        u32 pre_merge_obj_cnt = pdat.get_obj_cnt();

        pdat.index_remap(*out_buf_particle_index_map, pre_merge_obj_cnt);
    });
}

template class shammodels::basegodunov::modules::AMRSortBlocks<f64_3, i64_3>;
