// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file AMRSortBlocks.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shammodels/amr/basegodunov/modules/AMRSortBlocks.hpp"

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::AMRSortBlocks<Tvec, TgridVec>::reorder_amr_blocks() {

    using MortonBuilder = RadixTreeMortonBuilder<u64, TgridVec, 3>;
    using namespace shamrock::patch;

    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
        std::unique_ptr<sycl::buffer<u64>> out_buf_morton;
        std::unique_ptr<sycl::buffer<u32>> out_buf_particle_index_map;

        MortonBuilder::build(
            shamsys::instance::get_compute_queue(),
            scheduler().get_sim_box().template patch_coord_to_domain<TgridVec>(cur_p),
            *pdat.get_field<TgridVec>(0).get_buf(),
            pdat.get_obj_cnt(),
            out_buf_morton,
            out_buf_particle_index_map);

        // apply list permut on patch

        u32 pre_merge_obj_cnt = pdat.get_obj_cnt();

        pdat.index_remap(*out_buf_particle_index_map, pre_merge_obj_cnt);
    });
}

template class shammodels::basegodunov::modules::AMRSortBlocks<f64_3, i64_3>;
