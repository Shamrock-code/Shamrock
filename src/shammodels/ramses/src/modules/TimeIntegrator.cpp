// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file TimeIntegrator.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */
#include "shammodels/ramses/modules/TimeIntegrator.hpp"

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::TimeIntegrator<Tvec, TgridVec>::forward_euler(Tscal dt) {

    StackEntry stack_loc{};

    using namespace shamrock::patch;
    using namespace shamrock;
    using namespace shammath;

    shamrock::ComputeField<Tscal> &cfield_dtrho  = storage.dtrho.get();
    shamrock::ComputeField<Tvec> &cfield_dtrhov  = storage.dtrhov.get();
    shamrock::ComputeField<Tscal> &cfield_dtrhoe = storage.dtrhoe.get();

    // load layout info
    PatchDataLayerLayout &pdl = scheduler().pdl();

    const u32 icell_min = pdl.get_field_idx<TgridVec>("cell_min");
    const u32 icell_max = pdl.get_field_idx<TgridVec>("cell_max");
    const u32 irho      = pdl.get_field_idx<Tscal>("rho");
    const u32 irhoetot  = pdl.get_field_idx<Tscal>("rhoetot");
    const u32 irhovel   = pdl.get_field_idx<Tvec>("rhovel");

    scheduler().for_each_patchdata_nonempty(
        [&, dt](const shamrock::patch::Patch p, shamrock::patch::PatchDataLayer &pdat) {
            shamlog_debug_ln("[AMR Flux]", "forward euler integration patch", p.id_patch);

            sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
            u32 id               = p.id_patch;

            sham::DeviceBuffer<Tscal> &dt_rho_patch  = cfield_dtrho.get_buf_check(id);
            sham::DeviceBuffer<Tvec> &dt_rhov_patch  = cfield_dtrhov.get_buf_check(id);
            sham::DeviceBuffer<Tscal> &dt_rhoe_patch = cfield_dtrhoe.get_buf_check(id);

            u32 cell_count = pdat.get_obj_cnt() * AMRBlock::block_size;

            sham::DeviceBuffer<Tscal> &buf_rho  = pdat.get_field_buf_ref<Tscal>(irho);
            sham::DeviceBuffer<Tvec> &buf_rhov  = pdat.get_field_buf_ref<Tvec>(irhovel);
            sham::DeviceBuffer<Tscal> &buf_rhoe = pdat.get_field_buf_ref<Tscal>(irhoetot);

            sham::EventList depends_list;
            auto acc_dt_rho_patch  = dt_rho_patch.get_read_access(depends_list);
            auto acc_dt_rhov_patch = dt_rhov_patch.get_read_access(depends_list);
            auto acc_dt_rhoe_patch = dt_rhoe_patch.get_read_access(depends_list);

            auto rho  = buf_rho.get_write_access(depends_list);
            auto rhov = buf_rhov.get_write_access(depends_list);
            auto rhoe = buf_rhoe.get_write_access(depends_list);

            auto e = q.submit(depends_list, [&, dt](sycl::handler &cgh) {
                shambase::parallel_for(cgh, cell_count, "accumulate fluxes", [=](u32 id_a) {
                    const u32 cell_global_id = (u32) id_a;

                    rho[id_a] += dt * acc_dt_rho_patch[id_a];
                    rhov[id_a] += dt * acc_dt_rhov_patch[id_a];
                    rhoe[id_a] += dt * acc_dt_rhoe_patch[id_a];
                });
            });

            dt_rho_patch.complete_event_state(e);
            dt_rhov_patch.complete_event_state(e);
            dt_rhoe_patch.complete_event_state(e);

            buf_rho.complete_event_state(e);
            buf_rhov.complete_event_state(e);
            buf_rhoe.complete_event_state(e);
        });

    if (solver_config.is_dust_on()) {

        shamrock::ComputeField<Tscal> &cfield_dtrho_dust = storage.dtrho_dust.get();
        shamrock::ComputeField<Tvec> &cfield_dtrhov_dust = storage.dtrhov_dust.get();

        const u32 irho_dust    = pdl.get_field_idx<Tscal>("rho_dust");
        const u32 irhovel_dust = pdl.get_field_idx<Tvec>("rhovel_dust");

        scheduler().for_each_patchdata_nonempty([&, dt](
                                                    const shamrock::patch::Patch p,
                                                    shamrock::patch::PatchDataLayer &pdat) {
            shamlog_debug_ln(
                "[AMR Flux]", "forward euler integration patch for dust fields", p.id_patch);

            sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
            u32 id               = p.id_patch;

            sham::DeviceBuffer<Tscal> &dt_rho_dust_patch = cfield_dtrho_dust.get_buf_check(id);
            sham::DeviceBuffer<Tvec> &dt_rhov_dust_patch = cfield_dtrhov_dust.get_buf_check(id);

            u32 cell_count = pdat.get_obj_cnt() * AMRBlock::block_size;
            u32 ndust      = solver_config.dust_config.ndust;

            sham::DeviceBuffer<Tscal> &buf_rho_dust = pdat.get_field_buf_ref<Tscal>(irho_dust);
            sham::DeviceBuffer<Tvec> &buf_rhov_dust = pdat.get_field_buf_ref<Tvec>(irhovel_dust);

            sham::EventList depends_list;
            auto acc_dt_rho_dust_patch  = dt_rho_dust_patch.get_read_access(depends_list);
            auto acc_dt_rhov_dust_patch = dt_rhov_dust_patch.get_read_access(depends_list);

            auto rho_dust  = buf_rho_dust.get_write_access(depends_list);
            auto rhov_dust = buf_rhov_dust.get_write_access(depends_list);

            auto e = q.submit(depends_list, [&, dt](sycl::handler &cgh) {
                shambase::parallel_for(cgh, ndust * cell_count, "accumulate fluxes", [=](u32 id_a) {
                    rho_dust[id_a] += dt * acc_dt_rho_dust_patch[id_a];
                    rhov_dust[id_a] += dt * acc_dt_rhov_dust_patch[id_a];
                });
            });

            dt_rho_dust_patch.complete_event_state(e);
            dt_rhov_dust_patch.complete_event_state(e);
            buf_rho_dust.complete_event_state(e);
            buf_rhov_dust.complete_event_state(e);
        });
    }

    if (solver_config.is_gravity_on()) {
        shamrock::solvergraph::DDPatchDataFieldRef<Tscal> &phi_new
            = shambase::get_check_ref(storage.refs_phi).get_refs();
        const u32 iphi = pdl.get_field_idx<Tscal>("phi");

        scheduler().for_each_patchdata_nonempty(
            [&, dt](const shamrock::patch::Patch p, shamrock::patch::PatchDataLayer &pdat) {
                shamlog_debug_ln(
                    "[Self-gravity]", "Gravitational potential saveback patch", p.id_patch);

                sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
                u32 id               = p.id_patch;

                sham::DeviceBuffer<Tscal> &phi_new_patch = phi_new.get(id).get().get_buf();

                u32 cell_count = pdat.get_obj_cnt() * AMRBlock::block_size;

                sham::DeviceBuffer<Tscal> &phi_old = pdat.get_field_buf_ref<Tscal>(iphi);

                sham::EventList depends_list;
                auto acc_phi_new = phi_new_patch.get_read_access(depends_list);
                auto acc_phi_old = phi_old.get_write_access(depends_list);

                // auto inc = 2;

                auto e = q.submit(depends_list, [&, dt](sycl::handler &cgh) {
                    shambase::parallel_for(cgh, cell_count, "saveback", [=](u32 id_a) {
                        acc_phi_old[id_a] = acc_phi_new[id_a];
                    });
                });
                phi_new_patch.complete_event_state(e);
                phi_old.complete_event_state(e);
            });
    }
}

template class shammodels::basegodunov::modules::TimeIntegrator<f64_3, i64_3>;
