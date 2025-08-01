// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ComputeCFL.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shammodels/ramses/modules/ComputeCFL.hpp"
#include "fmt/core.h"
#include "shammath/riemann.hpp"
#include "shammath/riemann_dust.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"

template<class Tvec, class TgridVec>
auto shammodels::basegodunov::modules::ComputeCFL<Tvec, TgridVec>::compute_cfl() -> Tscal {

    StackEntry stack_loc{};

    using namespace shamrock::patch;
    using namespace shamrock;
    using namespace shammath;

    SchedulerUtility utility(scheduler());
    ComputeField<Tscal> cfl_dt = utility.make_compute_field<Tscal>("cfl_dt", AMRBlock::block_size);

    // load layout info
    PatchDataLayout &pdl = scheduler().pdl;

    const u32 icell_min = pdl.get_field_idx<TgridVec>("cell_min");
    const u32 icell_max = pdl.get_field_idx<TgridVec>("cell_max");
    const u32 irho      = pdl.get_field_idx<Tscal>("rho");
    const u32 irhoetot  = pdl.get_field_idx<Tscal>("rhoetot");
    const u32 irhovel   = pdl.get_field_idx<Tvec>("rhovel");

    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

        u32 cell_count = pdat.get_obj_cnt() * AMRBlock::block_size;

        sham::DeviceBuffer<TgridVec> &buf_block_min = pdat.get_field_buf_ref<TgridVec>(0);
        sham::DeviceBuffer<TgridVec> &buf_block_max = pdat.get_field_buf_ref<TgridVec>(1);

        sham::DeviceBuffer<Tscal> &buf_rho  = pdat.get_field_buf_ref<Tscal>(irho);
        sham::DeviceBuffer<Tvec> &buf_rhov  = pdat.get_field_buf_ref<Tvec>(irhovel);
        sham::DeviceBuffer<Tscal> &buf_rhoe = pdat.get_field_buf_ref<Tscal>(irhoetot);

        sham::DeviceBuffer<Tscal> &cfl_dt_buf = cfl_dt.get_buf_check(cur_p.id_patch);

        sham::DeviceBuffer<Tscal> &block_cell_sizes
            = shambase::get_check_ref(storage.block_cell_sizes)
                  .get_refs()
                  .get(cur_p.id_patch)
                  .get()
                  .get_buf();

        sham::EventList depends_list;
        auto cfl_dt        = cfl_dt_buf.get_write_access(depends_list);
        auto acc_block_min = buf_block_min.get_read_access(depends_list);
        auto acc_block_max = buf_block_max.get_read_access(depends_list);
        auto rho           = buf_rho.get_read_access(depends_list);
        auto rhov          = buf_rhov.get_read_access(depends_list);
        auto rhoe          = buf_rhoe.get_read_access(depends_list);

        auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
            Tscal C_safe = solver_config.Csafe;
            Tscal gamma  = solver_config.eos_gamma;

            Tscal one_over_Nside = 1. / AMRBlock::Nside;

            Tscal dxfact = solver_config.grid_coord_to_pos_fact;
            shambase::parallel_for(cgh, cell_count, "compute_cfl", [=](u64 gid) {
                const u32 cell_global_id = (u32) gid;

                const u32 block_id    = cell_global_id / AMRBlock::block_size;
                const u32 cell_loc_id = cell_global_id % AMRBlock::block_size;

                TgridVec lower       = acc_block_min[block_id];
                TgridVec upper       = acc_block_max[block_id];
                Tvec lower_flt       = lower.template convert<Tscal>() * dxfact;
                Tvec upper_flt       = upper.template convert<Tscal>() * dxfact;
                Tvec block_cell_size = (upper_flt - lower_flt) * one_over_Nside;
                Tscal dx             = block_cell_size.x();

                auto conststate = shammath::ConsState<Tvec>{rho[gid], rhoe[gid], rhov[gid]};

                auto prim_state = shammath::cons_to_prim(conststate, gamma);

                constexpr Tscal div = 1. / 3.;

                Tscal cs    = sound_speed(prim_state, gamma);
                Tscal vnorm = sycl::length(prim_state.vel);
                Tscal dt    = C_safe * dx * div / (cs + vnorm);

                cfl_dt[gid] = dt;
            });
        });

        cfl_dt_buf.complete_event_state(e);
        buf_block_min.complete_event_state(e);
        buf_block_max.complete_event_state(e);
        buf_rho.complete_event_state(e);
        buf_rhov.complete_event_state(e);
        buf_rhoe.complete_event_state(e);
    });

    Tscal rank_dt = cfl_dt.compute_rank_min();

    shamlog_debug_ln("basegodunov", "rank", shamcomm::world_rank(), "found cfl dt =", rank_dt);

    Tscal next_cfl = shamalgs::collective::allreduce_min(rank_dt);

    if (shamcomm::world_rank() == 0) {
        logger::info_ln("amr::basegodunov", "cfl dt =", next_cfl);
    }

    return next_cfl;
}

template<class Tvec, class TgridVec>
auto shammodels::basegodunov::modules::ComputeCFL<Tvec, TgridVec>::compute_dust_cfl() -> Tscal {

    StackEntry stack_loc{};

    using namespace shamrock::patch;
    using namespace shamrock;
    using namespace shammath;

    SchedulerUtility utility(scheduler());
    u32 ndust = solver_config.dust_config.ndust;
    ComputeField<Tscal> dust_cfl_dt
        = utility.make_compute_field<Tscal>("dust_cfl_dt", ndust * AMRBlock::block_size);

    // load layout info
    PatchDataLayout &pdl = scheduler().pdl;

    const u32 icell_min    = pdl.get_field_idx<TgridVec>("cell_min");
    const u32 icell_max    = pdl.get_field_idx<TgridVec>("cell_max");
    const u32 irho_dust    = pdl.get_field_idx<Tscal>("rho_dust");
    const u32 irhovel_dust = pdl.get_field_idx<Tvec>("rhovel_dust");

    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchDataLayer &pdat) {
        sham::DeviceQueue &q = shamsys::instance::get_compute_scheduler().get_queue();

        u32 cell_count = pdat.get_obj_cnt() * AMRBlock::block_size;

        sham::DeviceBuffer<Tscal> &buf_rho_dust = pdat.get_field_buf_ref<Tscal>(irho_dust);
        sham::DeviceBuffer<Tvec> &buf_rhov_dust = pdat.get_field_buf_ref<Tvec>(irhovel_dust);

        sham::DeviceBuffer<Tscal> &dust_cfl_dt_buf = dust_cfl_dt.get_buf_check(cur_p.id_patch);

        sham::DeviceBuffer<Tscal> &block_cell_sizes
            = shambase::get_check_ref(storage.block_cell_sizes)
                  .get_refs()
                  .get(cur_p.id_patch)
                  .get()
                  .get_buf();

        sham::EventList depends_list;
        auto dust_cfl_dt        = dust_cfl_dt_buf.get_write_access(depends_list);
        auto rho_dust           = buf_rho_dust.get_read_access(depends_list);
        auto rhov_dust          = buf_rhov_dust.get_read_access(depends_list);
        auto acc_aabb_cell_size = block_cell_sizes.get_read_access(depends_list);

        auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
            Tscal C_safe = solver_config.Csafe;

            shambase::parallel_for(cgh, ndust * cell_count, "compute_dust_cfl", [=](u64 gid) {
                const u32 tmp_gid        = (u32) gid;
                const u32 cell_global_id = tmp_gid / ndust;
                const u32 ndust_off_loc  = tmp_gid % ndust;

                const u32 block_id    = cell_global_id / AMRBlock::block_size;
                const u32 cell_loc_id = cell_global_id % AMRBlock::block_size;

                auto conststate = shammath::DustConsState<Tvec>{
                    rho_dust[ndust * cell_global_id + ndust_off_loc],
                    rhov_dust[ndust * cell_global_id + ndust_off_loc]};
                Tscal dx = acc_aabb_cell_size[block_id];

                auto prim_state = shammath::d_cons_to_prim(conststate);

                constexpr Tscal div = 1. / 3.;

                Tscal vnorm = sycl::length(prim_state.vel);
                Tscal dt    = C_safe * dx * div / (vnorm);

                dust_cfl_dt[ndust * cell_global_id + ndust_off_loc] = dt;
            });
        });

        dust_cfl_dt_buf.complete_event_state(e);
        buf_rho_dust.complete_event_state(e);
        buf_rhov_dust.complete_event_state(e);
        block_cell_sizes.complete_event_state(e);
    });

    Tscal rank_dust_dt = dust_cfl_dt.compute_rank_min();

    shamlog_debug_ln(
        "basegodunov", "rank", shamcomm::world_rank(), "found dust cfl dt =", rank_dust_dt);

    Tscal next_dust_cfl = shamalgs::collective::allreduce_min(rank_dust_dt);

    if (shamcomm::world_rank() == 0) {
        logger::info_ln("amr::basegodunov", "dust cfl dt =", next_dust_cfl);
    }

    return next_dust_cfl;
}
template class shammodels::basegodunov::modules::ComputeCFL<f64_3, i64_3>;
