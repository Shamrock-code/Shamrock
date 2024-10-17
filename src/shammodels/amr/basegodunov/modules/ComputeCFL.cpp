// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ComputeCFL.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shammodels/amr/basegodunov/modules/ComputeCFL.hpp"
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

    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
        sycl::queue &q = shamsys::instance::get_compute_queue();

        u32 cell_count = pdat.get_obj_cnt() * AMRBlock::block_size;

        sycl::buffer<TgridVec> &buf_block_min = pdat.get_field_buf_ref<TgridVec>(0);
        sycl::buffer<TgridVec> &buf_block_max = pdat.get_field_buf_ref<TgridVec>(1);

        sycl::buffer<Tscal> &buf_rho  = pdat.get_field_buf_ref<Tscal>(irho);
        sycl::buffer<Tvec> &buf_rhov  = pdat.get_field_buf_ref<Tvec>(irhovel);
        sycl::buffer<Tscal> &buf_rhoe = pdat.get_field_buf_ref<Tscal>(irhoetot);

        sycl::buffer<Tscal> &cfl_dt_buf = cfl_dt.get_buf_check(cur_p.id_patch);

        sycl::buffer<Tscal> &block_cell_sizes
            = storage.cell_infos.get().block_cell_sizes.get_buf_check(cur_p.id_patch);

        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
            sycl::accessor cfl_dt{cfl_dt_buf, cgh, sycl::write_only, sycl::no_init};

            sycl::accessor acc_block_min{buf_block_min, cgh, sycl::read_only};
            sycl::accessor acc_block_max{buf_block_max, cgh, sycl::read_only};
            sycl::accessor rho{buf_rho, cgh, sycl::read_only};
            sycl::accessor rhov{buf_rhov, cgh, sycl::read_only};
            sycl::accessor rhoe{buf_rhoe, cgh, sycl::read_only};

            Tscal C_safe = solver_config.Csafe;
            Tscal gamma  = solver_config.eos_gamma;

            Tscal one_over_Nside = 1. / AMRBlock::Nside;

            Tscal dxfact = solver_config.grid_coord_to_pos_fact;
            shambase::parralel_for(cgh, cell_count, "compute_cfl", [=](u64 gid) {
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
    });

    Tscal rank_dt = cfl_dt.compute_rank_min();

    logger::debug_ln("basegodunov", "rank", shamcomm::world_rank(), "found cfl dt =", rank_dt);

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

    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {
        sycl::queue &q = shamsys::instance::get_compute_queue();

        u32 cell_count = pdat.get_obj_cnt() * AMRBlock::block_size;

        sycl::buffer<Tscal> &buf_rho_dust = pdat.get_field_buf_ref<Tscal>(irho_dust);
        sycl::buffer<Tvec> &buf_rhov_dust = pdat.get_field_buf_ref<Tvec>(irhovel_dust);

        sycl::buffer<Tscal> &dust_cfl_dt_buf = dust_cfl_dt.get_buf_check(cur_p.id_patch);

        sycl::buffer<Tscal> &block_cell_sizes
            = storage.cell_infos.get().block_cell_sizes.get_buf_check(cur_p.id_patch);

        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
            sycl::accessor dust_cfl_dt{dust_cfl_dt_buf, cgh, sycl::write_only, sycl::no_init};

            sycl::accessor rho_dust{buf_rho_dust, cgh, sycl::read_only};
            sycl::accessor rhov_dust{buf_rhov_dust, cgh, sycl::read_only};

            sycl::accessor acc_aabb_cell_size{block_cell_sizes, cgh, sycl::read_only};

            Tscal C_safe = solver_config.Csafe;

            shambase::parralel_for(cgh, ndust * cell_count, "compute_dust_cfl", [=](u64 gid) {
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
    });

    Tscal rank_dust_dt = dust_cfl_dt.compute_rank_min();

    logger::debug_ln(
        "basegodunov", "rank", shamcomm::world_rank(), "found dust cfl dt =", rank_dust_dt);

    Tscal next_dust_cfl = shamalgs::collective::allreduce_min(rank_dust_dt);

    if (shamcomm::world_rank() == 0) {
        logger::info_ln("amr::basegodunov", "dust cfl dt =", next_dust_cfl);
    }

    return next_dust_cfl;
}
template class shammodels::basegodunov::modules::ComputeCFL<f64_3, i64_3>;
