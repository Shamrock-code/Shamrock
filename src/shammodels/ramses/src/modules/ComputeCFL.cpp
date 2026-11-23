// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
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
#include "shambackends/kernel_call.hpp"
#include "shammath/riemann.hpp"
#include "shammath/riemann_dust.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"

template<class Tvec, class TgridVec>
template<shammath::FluidStateSpec FSpec>
void shammodels::basegodunov::modules::ComputeCFL<Tvec, TgridVec>::compute_cfl_gas(
    shamrock::ComputeField<Tscal> &cfl_dt, FSpec fluid) {

    StackEntry stack_loc{};

    using namespace shamrock::patch;
    using namespace shamrock;
    using namespace shammath;

    // load layout info
    PatchDataLayerLayout &pdl = scheduler().pdl_old();

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

        Tscal C_safe = solver_config.Csafe;

        Tscal one_over_Nside = 1. / AMRBlock::Nside;

        Tscal dxfact = solver_config.grid_coord_to_pos_fact;

        auto fspec = fluid;

        sham::kernel_call(
            q,
            sham::MultiRef{buf_block_min, buf_block_max, buf_rho, buf_rhov, buf_rhoe},
            sham::MultiRef{cfl_dt_buf},
            cell_count,
            [C_safe, one_over_Nside, dxfact, fspec](
                u32 gid,
                const TgridVec *__restrict acc_block_min,
                const TgridVec *__restrict acc_block_max,
                const Tscal *__restrict rho,
                const Tvec *__restrict rhov,
                const Tscal *__restrict rhoe,
                Tscal *__restrict cfl_dt) {
                const u32 cell_global_id = (u32) gid;

                const u32 block_id    = cell_global_id / AMRBlock::block_size;
                const u32 cell_loc_id = cell_global_id % AMRBlock::block_size;

                TgridVec lower = acc_block_min[block_id];
                TgridVec upper = acc_block_max[block_id];
                // size from the int coords difference, so that it does not depend on the block
                // position
                Tvec block_cell_size
                    = (upper - lower).template convert<Tscal>() * (dxfact * one_over_Nside);
                Tscal dx = block_cell_size.x();

                // rhoe is ignored by the barotropic cons_to_prim
                auto conststate = shammath::ConsState<Tvec>{rho[gid], rhoe[gid], rhov[gid]};

                auto prim_state = fspec.cons_to_prim(conststate);

                constexpr Tscal div = 1. / 3.;

                Tscal cs    = fspec.sound_speed(prim_state);
                Tscal vnorm = sycl::length(prim_state.vel);
                Tscal dt    = C_safe * dx * div / (cs + vnorm);

                cfl_dt[gid] = dt;
            });
    });
}

template<class Tvec, class TgridVec>
auto shammodels::basegodunov::modules::ComputeCFL<Tvec, TgridVec>::compute_cfl() -> Tscal {

    StackEntry stack_loc{};

    using namespace shamrock;

    SchedulerUtility utility(scheduler());
    ComputeField<Tscal> cfl_dt = utility.make_compute_field<Tscal>("cfl_dt", AMRBlock::block_size);

    // each EOSConfig alternative maps to its shammath fluid spec (FluidStateAdiabatic or
    // FluidStateBarotropic) through get_spec()
    std::visit(
        [&](const auto &eos_cfg) {
            compute_cfl_gas(cfl_dt, eos_cfg.get_spec());
        },
        solver_config.eos_config.config);

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
    PatchDataLayerLayout &pdl = scheduler().pdl_old();

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

        Tscal C_safe = solver_config.Csafe;

        sham::kernel_call(
            q,
            sham::MultiRef{buf_rho_dust, buf_rhov_dust, block_cell_sizes},
            sham::MultiRef{dust_cfl_dt_buf},
            ndust * cell_count,
            [C_safe, ndust](
                u32 gid,
                const Tscal *__restrict rho_dust,
                const Tvec *__restrict rhov_dust,
                const Tscal *__restrict acc_aabb_cell_size,
                Tscal *__restrict dust_cfl_dt) {
                const u32 tmp_gid        = (u32) gid;
                const u32 cell_global_id = tmp_gid / ndust;
                const u32 ndust_off_loc  = tmp_gid % ndust;

                const u32 block_id    = cell_global_id / AMRBlock::block_size;
                const u32 cell_loc_id = cell_global_id % AMRBlock::block_size;

                auto conststate = shammath::DustConsState<Tvec>{
                    rho_dust[ndust * cell_global_id + ndust_off_loc],
                    rhov_dust[ndust * cell_global_id + ndust_off_loc]};
                Tscal dx = acc_aabb_cell_size[block_id];

                shammath::FluidStateDust<Tvec> dust_fluid{};
                auto prim_state = dust_fluid.cons_to_prim(conststate);

                constexpr Tscal div = 1. / 3.;

                Tscal vnorm = sycl::length(prim_state.vel);
                Tscal dt    = C_safe * dx * div / (vnorm);

                dust_cfl_dt[ndust * cell_global_id + ndust_off_loc] = dt;
            });
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
