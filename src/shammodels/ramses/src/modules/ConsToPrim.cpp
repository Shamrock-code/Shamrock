// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ConsToPrim.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shammodels/ramses/modules/ConsToPrim.hpp"
#include "shammath/riemann.hpp"
#include "shammath/riemann_dust.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"
#include <utility>

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::ConsToPrim<Tvec, TgridVec>::cons_to_prim() {

    StackEntry stack_loc{};

    using MergedPDat = shamrock::MergedPatchData;

    shamrock::SchedulerUtility utility(scheduler());

    shamrock::ComputeField<Tvec> v_ghost
        = utility.make_compute_field<Tvec>("vel", AMRBlock::block_size, [&](u64 id) {
              return storage.merged_patchdata_ghost.get().get(id).total_elements;
          });

    shamrock::ComputeField<Tscal> P_ghost
        = utility.make_compute_field<Tscal>("P", AMRBlock::block_size, [&](u64 id) {
              return storage.merged_patchdata_ghost.get().get(id).total_elements;
          });

    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 irho_ghost                                 = ghost_layout.get_field_idx<Tscal>("rho");
    u32 irhov_ghost                                = ghost_layout.get_field_idx<Tvec>("rhovel");
    u32 irhoe_ghost                                = ghost_layout.get_field_idx<Tscal>("rhoetot");

    storage.merged_patchdata_ghost.get().for_each([&](u64 id, MergedPDat &mpdat) {
        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

        sham::DeviceBuffer<TgridVec> &buf_block_min = mpdat.pdat.get_field_buf_ref<TgridVec>(0);
        sham::DeviceBuffer<TgridVec> &buf_block_max = mpdat.pdat.get_field_buf_ref<TgridVec>(1);

        sham::DeviceBuffer<Tscal> &buf_rho  = mpdat.pdat.get_field_buf_ref<Tscal>(irho_ghost);
        sham::DeviceBuffer<Tvec> &buf_rhov  = mpdat.pdat.get_field_buf_ref<Tvec>(irhov_ghost);
        sham::DeviceBuffer<Tscal> &buf_rhoe = mpdat.pdat.get_field_buf_ref<Tscal>(irhoe_ghost);

        sham::DeviceBuffer<Tvec> &buf_vel = v_ghost.get_buf(id);
        sham::DeviceBuffer<Tscal> &buf_P  = P_ghost.get_buf(id);

        sham::EventList depends_list;

        auto rho    = buf_rho.get_read_access(depends_list);
        auto rhovel = buf_rhov.get_read_access(depends_list);
        auto rhoe   = buf_rhoe.get_read_access(depends_list);

        auto vel = buf_vel.get_write_access(depends_list);
        auto P   = buf_P.get_write_access(depends_list);

        auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
            Tscal gamma = solver_config.eos_gamma;

            u32 cell_count = (mpdat.total_elements) * AMRBlock::block_size;

            shambase::parralel_for(cgh, cell_count, "cons_to_prim", [=](u64 gid) {
                auto conststate = shammath::ConsState<Tvec>{rho[gid], rhoe[gid], rhovel[gid]};

                auto prim_state = shammath::cons_to_prim(conststate, gamma);

                vel[gid] = prim_state.vel;
                P[gid]   = prim_state.press;
            });
        });

        buf_rho.complete_event_state(e);
        buf_rhov.complete_event_state(e);
        buf_rhoe.complete_event_state(e);
        buf_vel.complete_event_state(e);
        buf_P.complete_event_state(e);
    });

    storage.vel.set(std::move(v_ghost));
    storage.press.set(std::move(P_ghost));

    if (solver_config.is_dust_on()) {
        u32 ndust                                      = solver_config.dust_config.ndust;
        shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();

        shamrock::ComputeField<Tvec> v_dust_ghost = utility.make_compute_field<Tvec>(
            "vel_dust", ndust * AMRBlock::block_size, [&](u64 id) {
                return storage.merged_patchdata_ghost.get().get(id).total_elements;
            });
        u32 irho_dust_ghost  = ghost_layout.get_field_idx<Tscal>("rho_dust");
        u32 irhov_dust_ghost = ghost_layout.get_field_idx<Tvec>("rhovel_dust");

        storage.merged_patchdata_ghost.get().for_each([&](u64 id, MergedPDat &mpdat) {
            sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

            sham::DeviceBuffer<TgridVec> &buf_block_min = mpdat.pdat.get_field_buf_ref<TgridVec>(0);
            sham::DeviceBuffer<TgridVec> &buf_block_max = mpdat.pdat.get_field_buf_ref<TgridVec>(1);

            sham::DeviceBuffer<Tscal> &buf_rho_dust
                = mpdat.pdat.get_field_buf_ref<Tscal>(irho_dust_ghost);
            sham::DeviceBuffer<Tvec> &buf_rhov_dust
                = mpdat.pdat.get_field_buf_ref<Tvec>(irhov_dust_ghost);

            sham::DeviceBuffer<Tvec> &buf_vel_dust = v_dust_ghost.get_buf(id);

            sham::EventList depends_list;

            auto rho_dust    = buf_rho_dust.get_read_access(depends_list);
            auto rhovel_dust = buf_rhov_dust.get_read_access(depends_list);

            auto vel_dust = buf_vel_dust.get_write_access(depends_list);

            auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
                u32 cell_count = (mpdat.total_elements) * AMRBlock::block_size;

                u32 nvar_dust = ndust;
                shambase::parralel_for(
                    cgh, nvar_dust * cell_count, "cons_to_prim_dust", [=](u64 gid) {
                        auto d_conststate
                            = shammath::DustConsState<Tvec>{rho_dust[gid], rhovel_dust[gid]};

                        auto d_prim_state = shammath::d_cons_to_prim(d_conststate);

                        vel_dust[gid] = d_prim_state.vel;
                    });
            });

            buf_rho_dust.complete_event_state(e);
            buf_rhov_dust.complete_event_state(e);
            buf_vel_dust.complete_event_state(e);
        });
        storage.vel_dust.set(std::move(v_dust_ghost));
    }
}

template class shammodels::basegodunov::modules::ConsToPrim<f64_3, i64_3>;
