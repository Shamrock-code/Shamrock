// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file GhostCommunicator.cpp
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief MPI ghost field exchange implementation
 */

#include "shambase/stacktrace.hpp"
#include "shambase/time.hpp"
#include "shammodels/gsph/modules/GhostCommunicator.hpp"
#include "shammodels/sph/BasicSPHGhosts.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"
#include "shamrock/solvergraph/Field.hpp"

namespace shammodels::gsph::modules {

    template<class Tvec, template<class> class SPHKernel>
    void GhostCommunicator<Tvec, SPHKernel>::communicate_merge_ghosts_fields() {
        StackEntry stack_loc{};

        shambase::Timer timer_interf;
        timer_interf.start();

        using namespace shamrock;
        using namespace shamrock::patch;

        PatchDataLayerLayout &pdl = scheduler().pdl();
        const u32 ixyz            = pdl.template get_field_idx<Tvec>("xyz");
        const u32 ivxyz           = pdl.template get_field_idx<Tvec>("vxyz");
        const u32 ihpart          = pdl.template get_field_idx<Tscal>("hpart");

        const bool has_uint  = solver_config.has_field_uint();
        const u32 iuint      = has_uint ? pdl.template get_field_idx<Tscal>("uint") : 0;
        const bool has_pmass = solver_config.has_field_pmass();
        const u32 ipmass     = has_pmass ? pdl.template get_field_idx<Tscal>("pmass") : 0;

        auto ghost_layout_ptr              = storage.ghost_layout.get();
        PatchDataLayerLayout &ghost_layout = shambase::get_check_ref(ghost_layout_ptr);
        u32 ihpart_interf                  = ghost_layout.template get_field_idx<Tscal>("hpart");
        u32 ivxyz_interf                   = ghost_layout.template get_field_idx<Tvec>("vxyz");
        u32 iomega_interf                  = ghost_layout.template get_field_idx<Tscal>("omega");
        u32 idensity_interf                = ghost_layout.template get_field_idx<Tscal>("density");
        u32 iuint_interf  = has_uint ? ghost_layout.template get_field_idx<Tscal>("uint") : 0;
        u32 ipmass_interf = has_pmass ? ghost_layout.template get_field_idx<Tscal>("pmass") : 0;

        const bool has_grads = solver_config.requires_gradients();
        u32 igrad_d_interf
            = has_grads ? ghost_layout.template get_field_idx<Tvec>("grad_density") : 0;
        u32 igrad_p_interf
            = has_grads ? ghost_layout.template get_field_idx<Tvec>("grad_pressure") : 0;
        u32 igrad_vx_interf = has_grads ? ghost_layout.template get_field_idx<Tvec>("grad_vx") : 0;
        u32 igrad_vy_interf = has_grads ? ghost_layout.template get_field_idx<Tvec>("grad_vy") : 0;
        u32 igrad_vz_interf = has_grads ? ghost_layout.template get_field_idx<Tvec>("grad_vz") : 0;

        using InterfaceBuildInfos = typename sph::BasicSPHGhostHandler<Tvec>::InterfaceBuildInfos;

        sph::BasicSPHGhostHandler<Tvec> &ghost_handle = storage.ghost_handler.get();
        shamrock::solvergraph::Field<Tscal> &omega    = shambase::get_check_ref(storage.omega);
        shamrock::solvergraph::Field<Tscal> &density  = shambase::get_check_ref(storage.density);

        shamrock::solvergraph::Field<Tvec> *grad_density_ptr
            = has_grads ? &shambase::get_check_ref(storage.grad_density) : nullptr;
        shamrock::solvergraph::Field<Tvec> *grad_pressure_ptr
            = has_grads ? &shambase::get_check_ref(storage.grad_pressure) : nullptr;
        shamrock::solvergraph::Field<Tvec> *grad_vx_ptr
            = has_grads ? &shambase::get_check_ref(storage.grad_vx) : nullptr;
        shamrock::solvergraph::Field<Tvec> *grad_vy_ptr
            = has_grads ? &shambase::get_check_ref(storage.grad_vy) : nullptr;
        shamrock::solvergraph::Field<Tvec> *grad_vz_ptr
            = has_grads ? &shambase::get_check_ref(storage.grad_vz) : nullptr;

        auto pdat_interf = ghost_handle.template build_interface_native<PatchDataLayer>(
            storage.ghost_patch_cache.get(),
            [&](u64 sender,
                u64,
                InterfaceBuildInfos binfo,
                sham::DeviceBuffer<u32> &buf_idx,
                u32 cnt) {
                PatchDataLayer pdat(ghost_layout_ptr);
                pdat.reserve(cnt);
                return pdat;
            });

        ghost_handle.template modify_interface_native<PatchDataLayer>(
            storage.ghost_patch_cache.get(),
            pdat_interf,
            [&](u64 sender,
                u64,
                InterfaceBuildInfos binfo,
                sham::DeviceBuffer<u32> &buf_idx,
                u32 cnt,
                PatchDataLayer &pdat) {
                PatchDataLayer &sender_patch          = scheduler().patch_data.get_pdat(sender);
                PatchDataField<Tscal> &sender_omega   = omega.get(sender);
                PatchDataField<Tscal> &sender_density = density.get(sender);

                sender_patch.template get_field<Tscal>(ihpart).append_subset_to(
                    buf_idx, cnt, pdat.template get_field<Tscal>(ihpart_interf));
                sender_patch.template get_field<Tvec>(ivxyz).append_subset_to(
                    buf_idx, cnt, pdat.template get_field<Tvec>(ivxyz_interf));
                sender_omega.append_subset_to(
                    buf_idx, cnt, pdat.template get_field<Tscal>(iomega_interf));
                sender_density.append_subset_to(
                    buf_idx, cnt, pdat.template get_field<Tscal>(idensity_interf));

                if (has_uint) {
                    sender_patch.template get_field<Tscal>(iuint).append_subset_to(
                        buf_idx, cnt, pdat.template get_field<Tscal>(iuint_interf));
                }

                if (has_pmass) {
                    sender_patch.template get_field<Tscal>(ipmass).append_subset_to(
                        buf_idx, cnt, pdat.template get_field<Tscal>(ipmass_interf));
                }

                if (has_grads) {
                    grad_density_ptr->get(sender).append_subset_to(
                        buf_idx, cnt, pdat.template get_field<Tvec>(igrad_d_interf));
                    grad_pressure_ptr->get(sender).append_subset_to(
                        buf_idx, cnt, pdat.template get_field<Tvec>(igrad_p_interf));
                    grad_vx_ptr->get(sender).append_subset_to(
                        buf_idx, cnt, pdat.template get_field<Tvec>(igrad_vx_interf));
                    grad_vy_ptr->get(sender).append_subset_to(
                        buf_idx, cnt, pdat.template get_field<Tvec>(igrad_vy_interf));
                    grad_vz_ptr->get(sender).append_subset_to(
                        buf_idx, cnt, pdat.template get_field<Tvec>(igrad_vz_interf));
                }
            });

        ghost_handle.template modify_interface_native<PatchDataLayer>(
            storage.ghost_patch_cache.get(),
            pdat_interf,
            [&](u64 sender,
                u64,
                InterfaceBuildInfos binfo,
                sham::DeviceBuffer<u32> &buf_idx,
                u32 cnt,
                PatchDataLayer &pdat) {
                if (sycl::length(binfo.offset_speed) > 0) {
                    pdat.template get_field<Tvec>(ivxyz_interf).apply_offset(binfo.offset_speed);
                }
            });

        shambase::DistributedDataShared<PatchDataLayer> interf_pdat
            = ghost_handle.communicate_pdat(ghost_layout_ptr, std::move(pdat_interf));

        std::map<u64, u64> sz_interf_map;
        interf_pdat.for_each([&](u64 s, u64 r, PatchDataLayer &pdat_interf) {
            sz_interf_map[r] += pdat_interf.get_obj_cnt();
        });

        storage.merged_patchdata_ghost.set(
            ghost_handle.template merge_native<PatchDataLayer, PatchDataLayer>(
                std::move(interf_pdat),
                [&](const Patch p, PatchDataLayer &pdat) {
                    PatchDataLayer pdat_new(ghost_layout_ptr);

                    u32 or_elem = pdat.get_obj_cnt();
                    pdat_new.reserve(or_elem + sz_interf_map[p.id_patch]);

                    PatchDataField<Tscal> &cur_omega   = omega.get(p.id_patch);
                    PatchDataField<Tscal> &cur_density = density.get(p.id_patch);

                    pdat_new.template get_field<Tscal>(ihpart_interf)
                        .insert(pdat.template get_field<Tscal>(ihpart));
                    pdat_new.template get_field<Tvec>(ivxyz_interf)
                        .insert(pdat.template get_field<Tvec>(ivxyz));
                    pdat_new.template get_field<Tscal>(iomega_interf).insert(cur_omega);
                    pdat_new.template get_field<Tscal>(idensity_interf).insert(cur_density);

                    if (has_uint) {
                        pdat_new.template get_field<Tscal>(iuint_interf)
                            .insert(pdat.template get_field<Tscal>(iuint));
                    }

                    if (has_pmass) {
                        pdat_new.template get_field<Tscal>(ipmass_interf)
                            .insert(pdat.template get_field<Tscal>(ipmass));
                    }

                    if (has_grads) {
                        pdat_new.template get_field<Tvec>(igrad_d_interf)
                            .insert(grad_density_ptr->get(p.id_patch));
                        pdat_new.template get_field<Tvec>(igrad_p_interf)
                            .insert(grad_pressure_ptr->get(p.id_patch));
                        pdat_new.template get_field<Tvec>(igrad_vx_interf)
                            .insert(grad_vx_ptr->get(p.id_patch));
                        pdat_new.template get_field<Tvec>(igrad_vy_interf)
                            .insert(grad_vy_ptr->get(p.id_patch));
                        pdat_new.template get_field<Tvec>(igrad_vz_interf)
                            .insert(grad_vz_ptr->get(p.id_patch));
                    }

                    pdat_new.check_field_obj_cnt_match();
                    return pdat_new;
                },
                [](PatchDataLayer &pdat, PatchDataLayer &pdat_interf) {
                    pdat.insert_elements(pdat_interf);
                }));

        timer_interf.end();
        storage.timings_details.interface += timer_interf.elasped_sec();
    }

    // Explicit instantiations
    template class GhostCommunicator<f64_3, shammath::M4>;
    template class GhostCommunicator<f64_3, shammath::M6>;
    template class GhostCommunicator<f64_3, shammath::M8>;
    template class GhostCommunicator<f64_3, shammath::C2>;
    template class GhostCommunicator<f64_3, shammath::C4>;
    template class GhostCommunicator<f64_3, shammath::C6>;
    template class GhostCommunicator<f64_3, shammath::TGauss3>;

} // namespace shammodels::gsph::modules
