// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
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
#include "shambackends/kernel_call.hpp"

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::TimeIntegrator<Tvec, TgridVec>::forward_euler(Tscal dt) {

    StackEntry stack_loc{};

    using namespace shamrock::patch;
    using namespace shamrock;
    using namespace shammath;

    shamrock::solvergraph::Field<Tscal> &cfield_dtrho  = shambase::get_check_ref(storage.dtrho);
    shamrock::solvergraph::Field<Tvec> &cfield_dtrhov  = shambase::get_check_ref(storage.dtrhov);
    shamrock::solvergraph::Field<Tscal> &cfield_dtrhoe = shambase::get_check_ref(storage.dtrhoe);

    // load layout info
    PatchDataLayerLayout &pdl = scheduler().pdl_old();

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

            sham::DeviceBuffer<Tscal> &dt_rho_patch  = cfield_dtrho.get_buf(id);
            sham::DeviceBuffer<Tvec> &dt_rhov_patch  = cfield_dtrhov.get_buf(id);
            sham::DeviceBuffer<Tscal> &dt_rhoe_patch = cfield_dtrhoe.get_buf(id);

            u32 cell_count = pdat.get_obj_cnt() * AMRBlock::block_size;

            sham::DeviceBuffer<Tscal> &buf_rho  = pdat.get_field_buf_ref<Tscal>(irho);
            sham::DeviceBuffer<Tvec> &buf_rhov  = pdat.get_field_buf_ref<Tvec>(irhovel);
            sham::DeviceBuffer<Tscal> &buf_rhoe = pdat.get_field_buf_ref<Tscal>(irhoetot);

            sham::kernel_call(
                q,
                sham::MultiRef{dt_rho_patch, dt_rhov_patch, dt_rhoe_patch},
                sham::MultiRef{buf_rho, buf_rhov, buf_rhoe},
                cell_count,
                [dt](
                    u32 id_a,
                    const Tscal *__restrict acc_dt_rho_patch,
                    const Tvec *__restrict acc_dt_rhov_patch,
                    const Tscal *__restrict acc_dt_rhoe_patch,
                    Tscal *__restrict rho,
                    Tvec *__restrict rhov,
                    Tscal *__restrict rhoe) {
                    const u32 cell_global_id = (u32) id_a;

                    rho[id_a] += dt * acc_dt_rho_patch[id_a];
                    rhov[id_a] += dt * acc_dt_rhov_patch[id_a];
                    rhoe[id_a] += dt * acc_dt_rhoe_patch[id_a];
                });
        });

    if (solver_config.is_dust_on()) {

        shamrock::solvergraph::Field<Tscal> &cfield_dtrho_dust
            = shambase::get_check_ref(storage.dtrho_dust);
        shamrock::solvergraph::Field<Tvec> &cfield_dtrhov_dust
            = shambase::get_check_ref(storage.dtrhov_dust);

        const u32 irho_dust    = pdl.get_field_idx<Tscal>("rho_dust");
        const u32 irhovel_dust = pdl.get_field_idx<Tvec>("rhovel_dust");

        scheduler().for_each_patchdata_nonempty(
            [&, dt](const shamrock::patch::Patch p, shamrock::patch::PatchDataLayer &pdat) {
                shamlog_debug_ln(
                    "[AMR Flux]", "forward euler integration patch for dust fields", p.id_patch);

                sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();
                u32 id               = p.id_patch;

                sham::DeviceBuffer<Tscal> &dt_rho_dust_patch = cfield_dtrho_dust.get_buf(id);
                sham::DeviceBuffer<Tvec> &dt_rhov_dust_patch = cfield_dtrhov_dust.get_buf(id);

                u32 cell_count = pdat.get_obj_cnt() * AMRBlock::block_size;
                u32 ndust      = solver_config.dust_config.ndust;

                sham::DeviceBuffer<Tscal> &buf_rho_dust = pdat.get_field_buf_ref<Tscal>(irho_dust);
                sham::DeviceBuffer<Tvec> &buf_rhov_dust
                    = pdat.get_field_buf_ref<Tvec>(irhovel_dust);

                sham::kernel_call(
                    q,
                    sham::MultiRef{dt_rho_dust_patch, dt_rhov_dust_patch},
                    sham::MultiRef{buf_rho_dust, buf_rhov_dust},
                    ndust * cell_count,
                    [dt](
                        u32 id_a,
                        const Tscal *__restrict acc_dt_rho_dust_patch,
                        const Tvec *__restrict acc_dt_rhov_dust_patch,
                        Tscal *__restrict rho_dust,
                        Tvec *__restrict rhov_dust) {
                        rho_dust[id_a] += dt * acc_dt_rho_dust_patch[id_a];
                        rhov_dust[id_a] += dt * acc_dt_rhov_dust_patch[id_a];
                    });
            });
    }
}

template class shammodels::basegodunov::modules::TimeIntegrator<f64_3, i64_3>;
