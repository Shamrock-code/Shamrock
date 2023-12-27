// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file WriteBack.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief 
 * 
 */
 
#include "shammodels/amr/zeus/modules/WriteBack.hpp"

template<class Tvec, class TgridVec>
using Module = shammodels::zeus::modules::WriteBack<Tvec, TgridVec>;

template<class Tvec, class TgridVec>
void Module<Tvec, TgridVec>::write_back_merged_data(){

    StackEntry stack_loc{};


    using namespace shamrock::patch;
    using namespace shamrock;

    using Block = typename Config::AMRBlock;

    PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 irho_interf                                = ghost_layout.get_field_idx<Tscal>("rho");
    u32 ieint_interf                                = ghost_layout.get_field_idx<Tscal>("eint");
    u32 ivel_interf                                = ghost_layout.get_field_idx<Tvec>("vel");

    scheduler().for_each_patchdata_nonempty([&](Patch p, PatchData &pdat) {

        using MergedPDat = shamrock::MergedPatchData;
        MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(p.id_patch);

        sycl::buffer<Tscal> &rho_merged = mpdat.pdat.get_field_buf_ref<Tscal>(irho_interf);
        sycl::buffer<Tscal> &eint_merged = mpdat.pdat.get_field_buf_ref<Tscal>(ieint_interf);
        sycl::buffer<Tvec> &vel_merged = mpdat.pdat.get_field_buf_ref<Tvec>(ivel_interf);

        PatchData &patch_dest = scheduler().patch_data.get_pdat(p.id_patch);
        sycl::buffer<Tscal> &rho_dest = patch_dest.get_field_buf_ref<Tscal>(irho_interf);
        sycl::buffer<Tscal> &eint_dest = patch_dest.get_field_buf_ref<Tscal>(ieint_interf);
        sycl::buffer<Tvec> &vel_dest = patch_dest.get_field_buf_ref<Tvec>(ivel_interf);

        shamsys::instance::get_compute_queue().submit([&](sycl::handler & cgh){

            sycl::accessor acc_rho_src{rho_merged, cgh, sycl::read_only};
            sycl::accessor acc_eint_src{eint_merged, cgh, sycl::read_only};
            sycl::accessor acc_vel_src{vel_merged, cgh, sycl::read_only};

            sycl::accessor acc_rho_dest{rho_dest, cgh, sycl::write_only};
            sycl::accessor acc_eint_dest{eint_dest, cgh, sycl::write_only};
            sycl::accessor acc_vel_dest{vel_dest, cgh, sycl::write_only};

            shambase::parralel_for(cgh, mpdat.original_elements*Block::block_size, "copy_back", [=](u32 id){
                acc_rho_dest[id] = acc_rho_src[id];
                acc_eint_dest[id] = acc_eint_src[id];
                acc_vel_dest[id] = acc_vel_src[id];
            });
        });

        if (mpdat.pdat.has_nan()) {
            logger::err_ln("[Zeus]", "nan detected in write back");
            throw shambase::make_except_with_loc<std::runtime_error>("detected nan");
        }
        
    });

    
}


template class shammodels::zeus::modules::WriteBack<f64_3, i64_3>;