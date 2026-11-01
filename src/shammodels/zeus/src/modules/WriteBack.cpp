// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file WriteBack.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shammodels/zeus/modules/WriteBack.hpp"

template<class Tvec, class TgridVec>
void shammodels::zeus::modules::WriteBack<Tvec, TgridVec>::write_back_merged_data() {

    StackEntry stack_loc{};

    using namespace shamrock::patch;
    using namespace shamrock;

    using Block = typename Config::AMRBlock;

    PatchDataLayerLayout &ghost_layout = shambase::get_check_ref(storage.ghost_layout.get());
    u32 irho_interf                    = ghost_layout.get_field_idx<Tscal>("rho");
    u32 ieint_interf                   = ghost_layout.get_field_idx<Tscal>("eint");
    u32 ivel_interf                    = ghost_layout.get_field_idx<Tvec>("vel");

    scheduler().for_each_patchdata_nonempty([&](Patch p, PatchDataLayer &pdat) {
        using MergedPDat  = shamrock::MergedPatchData;
        MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(p.id_patch);

        sham::DeviceBuffer<Tscal> &rho_merged  = mpdat.pdat.get_field_buf_ref<Tscal>(irho_interf);
        sham::DeviceBuffer<Tscal> &eint_merged = mpdat.pdat.get_field_buf_ref<Tscal>(ieint_interf);
        sham::DeviceBuffer<Tvec> &vel_merged   = mpdat.pdat.get_field_buf_ref<Tvec>(ivel_interf);

        PatchDataLayer &patch_dest           = scheduler().patch_data.get_pdat(p.id_patch);
        sham::DeviceBuffer<Tscal> &rho_dest  = patch_dest.get_field_buf_ref<Tscal>(irho_interf);
        sham::DeviceBuffer<Tscal> &eint_dest = patch_dest.get_field_buf_ref<Tscal>(ieint_interf);
        sham::DeviceBuffer<Tvec> &vel_dest   = patch_dest.get_field_buf_ref<Tvec>(ivel_interf);

        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

        sham::kernel_call(
            q,
            sham::MultiRef{rho_merged, eint_merged, vel_merged},
            sham::MultiRef{rho_dest, eint_dest, vel_dest},
            mpdat.original_elements * Block::block_size,
            [](u32 id,
               const Tscal *__restrict acc_rho_src,
               const Tscal *__restrict acc_eint_src,
               const Tvec *__restrict acc_vel_src,
               Tscal *__restrict acc_rho_dest,
               Tscal *__restrict acc_eint_dest,
               Tvec *__restrict acc_vel_dest) {
                acc_rho_dest[id]  = acc_rho_src[id];
                acc_eint_dest[id] = acc_eint_src[id];
                acc_vel_dest[id]  = acc_vel_src[id];
            });

        if (mpdat.pdat.has_nan()) {
            logger::err_ln("[Zeus]", "nan detected in write back");
            throw shambase::make_except_with_loc<std::runtime_error>("detected nan");
        }
    });
}

template class shammodels::zeus::modules::WriteBack<f64_3, i64_3>;
