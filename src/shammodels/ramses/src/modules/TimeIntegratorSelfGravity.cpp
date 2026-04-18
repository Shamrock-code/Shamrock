// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file TimeIntegratorSelfGravity.cpp
 * @author Léodasce Sewanou (leodasce.sewanou@ens-lyon.fr)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */
#include "shambase/memory.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shambackends/sycl_utils.hpp"
#include "shamcomm/logs.hpp"
#include "shammodels/ramses/modules/TimeIntegratorSelfGravity.hpp"
#include "shamrock/patch/PatchDataLayer.hpp"

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::TimeIntegratorSelfGravity<Tvec, TgridVec>::forward_euler(
    Tscal dt) {

    StackEntry stack_loc{};

    using namespace shamrock::patch;
    using namespace shamrock;
    using namespace shammath;

    // load layout info
    PatchDataLayerLayout &pdl = scheduler().pdl_old();

    const u32 irho     = pdl.get_field_idx<Tscal>("rho");
    const u32 irhoetot = pdl.get_field_idx<Tscal>("rhoetot");
    const u32 irhovel  = pdl.get_field_idx<Tvec>("rhovel");
    const u32 iphi     = pdl.get_field_idx<Tscal>("phi");

    {

        auto &rho_next  = shambase::get_check_ref(storage.refs_rho_next);
        auto &rhov_next = shambase::get_check_ref(storage.refs_rhov_next);
        auto &rhoe_next = shambase::get_check_ref(storage.refs_rhoe_next);

        scheduler().for_each_patchdata_nonempty([&](const shamrock::patch::Patch p,
                                                    shamrock::patch::PatchDataLayer &pdat) {
            shamlog_debug_ln(
                "[AMR Flux]", "forward euler integration-self-gravity patch", p.id_patch);

            sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

            sham::DeviceBuffer<Tscal> &rho_next_patch  = rho_next.get_buf(p.id_patch);
            sham::DeviceBuffer<Tvec> &rhov_next_patch  = rhov_next.get_buf(p.id_patch);
            sham::DeviceBuffer<Tscal> &rhoe_next_patch = rhoe_next.get_buf(p.id_patch);

            u32 cell_count = pdat.get_obj_cnt() * AMRBlock::block_size;

            sham::DeviceBuffer<Tscal> &buf_rho  = pdat.get_field_buf_ref<Tscal>(irho);
            sham::DeviceBuffer<Tvec> &buf_rhov  = pdat.get_field_buf_ref<Tvec>(irhovel);
            sham::DeviceBuffer<Tscal> &buf_rhoe = pdat.get_field_buf_ref<Tscal>(irhoetot);

            sham::EventList depends_list;
            auto acc_rho_next_patch  = rho_next_patch.get_read_access(depends_list);
            auto acc_rhov_next_patch = rhov_next_patch.get_read_access(depends_list);
            auto acc_rhoe_next_patch = rhoe_next_patch.get_read_access(depends_list);

            auto rho_old  = buf_rho.get_write_access(depends_list);
            auto rhov_old = buf_rhov.get_write_access(depends_list);
            auto rhoe_old = buf_rhoe.get_write_access(depends_list);

            auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                shambase::parallel_for(cgh, cell_count, "next-times-step-cons-var", [=](u32 id_a) {
                    rho_old[id_a]  = acc_rho_next_patch[id_a];
                    rhov_old[id_a] = acc_rhov_next_patch[id_a];
                    rhoe_old[id_a] = acc_rhoe_next_patch[id_a];

                    // logger::raw(rho_old[id_a]);
                });
            });

            rho_next_patch.complete_event_state(e);
            rhov_next_patch.complete_event_state(e);
            rhoe_next_patch.complete_event_state(e);

            buf_rho.complete_event_state(e);
            buf_rhov.complete_event_state(e);
            buf_rhoe.complete_event_state(e);
        });
    }

    if (solver_config.is_gravity_on() && (!solver_config.is_coupling_gravity_on())) {
        logger::raw_ln("Self-gravity -- [NO COUPLING MODE] \n");

        shamrock::solvergraph::DDPatchDataFieldRef<Tscal> &phi_new
            = shambase::get_check_ref(storage.refs_phi).get_refs();
        scheduler().for_each_patchdata_nonempty(
            [&, dt](const shamrock::patch::Patch p, shamrock::patch::PatchDataLayer &pdat) {
                shamlog_debug_ln(
                    "[Self-gravity]", "Gravitational potential saveback patch", p.id_patch);

                sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

                sham::DeviceBuffer<Tscal> &phi_new_patch = phi_new.get(p.id_patch).get().get_buf();

                u32 cell_count = pdat.get_obj_cnt() * AMRBlock::block_size;

                sham::DeviceBuffer<Tscal> &phi_old = pdat.get_field_buf_ref<Tscal>(iphi);

                sham::EventList depends_list;
                auto acc_phi_new = phi_new_patch.get_read_access(depends_list);
                auto acc_phi_old = phi_old.get_write_access(depends_list);

                auto e = q.submit(depends_list, [&, dt](sycl::handler &cgh) {
                    shambase::parallel_for(cgh, cell_count, "saveback", [=](u32 id_a) {
                        acc_phi_old[id_a] = acc_phi_new[id_a];
                    });
                });
                phi_new_patch.complete_event_state(e);
                phi_old.complete_event_state(e);
            });
    }

    if (solver_config.is_gravity_on() && solver_config.is_coupling_gravity_on()) {
        logger::raw_ln("Self-gravity -- [COUPLING MODE] \n");
        auto &phi_next = shambase::get_check_ref(storage.refs_phi_new);

        scheduler().for_each_patchdata_nonempty(
            [&, dt](const shamrock::patch::Patch p, shamrock::patch::PatchDataLayer &pdat) {
                shamlog_debug_ln(
                    "[Self-gravity]", "Gravitational potential saveback patch", p.id_patch);

                sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

                sham::DeviceBuffer<Tscal> &phi_new_patch = phi_next.get(p.id_patch).get_buf();

                u32 cell_count = pdat.get_obj_cnt() * AMRBlock::block_size;

                sham::DeviceBuffer<Tscal> &phi_old = pdat.get_field_buf_ref<Tscal>(iphi);

                sham::EventList depends_list;
                auto acc_phi_new = phi_new_patch.get_read_access(depends_list);
                auto acc_phi_old = phi_old.get_write_access(depends_list);

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

template class shammodels::basegodunov::modules::TimeIntegratorSelfGravity<f64_3, i64_3>;
