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
#include "shamrock/scheduler/SchedulerUtility.hpp"
#include <hipSYCL/sycl/libkernel/builtins.hpp>

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
    const u32 irhoetot     = pdl.get_field_idx<Tscal>("rhoetot");
    const u32 irhovel      = pdl.get_field_idx<Tvec>("rhovel");

    scheduler().for_each_patchdata_nonempty([&](Patch cur_p, PatchData &pdat) {

        
        sycl::queue &q                        = shamsys::instance::get_compute_queue();
        
        u32 cell_count = pdat.get_obj_cnt()*AMRBlock::block_size;

        sycl::buffer<Tscal> & buf_rho = pdat.get_field_buf_ref<Tscal>(irho);
        sycl::buffer<Tvec> & buf_rhov = pdat.get_field_buf_ref<Tvec>(irhovel);
        sycl::buffer<Tscal> & buf_rhoe = pdat.get_field_buf_ref<Tscal>(irhoetot);

        sycl::buffer<Tscal> &cfl_dt_buf = cfl_dt.get_buf_check(cur_p.id_patch);

        sycl::buffer<Tscal> &block_cell_sizes
            = storage.cell_infos.get().block_cell_sizes.get_buf_check(cur_p.id_patch);

        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {

            sycl::accessor cfl_dt{cfl_dt_buf, cgh, sycl::write_only, sycl::no_init};

            sycl::accessor rho  {buf_rho, cgh, sycl::read_only};
            sycl::accessor rhov {buf_rhov, cgh, sycl::read_only};
            sycl::accessor rhoe {buf_rhoe, cgh, sycl::read_only};
            
            sycl::accessor acc_aabb_cell_size{block_cell_sizes, cgh, sycl::read_only};

            Tscal C_safe = 0.5; // TODO move to config
            Tscal gamma = solver_config.eos_gamma;

            
            shambase::parralel_for(cgh, cell_count, "compute_cfl", [=](u64 gid) {
                
                const u32 cell_global_id = (u32) gid;

                const u32 block_id    = cell_global_id / AMRBlock::block_size;
                const u32 cell_loc_id = cell_global_id % AMRBlock::block_size;

                auto conststate = shammath::ConsState<Tvec>{rho[gid], rhoe[gid], rhov[gid]};
                Tscal dx = acc_aabb_cell_size[block_id];
                
                auto prim_state = shammath::cons_to_prim(conststate, gamma);

                Tscal cs = sound_speed(prim_state, gamma);
                Tscal vnorm = sycl::length(prim_state.vel);
                Tscal dt = C_safe * dx / ( cs+ vnorm);
                
                cfl_dt[gid] = dt;
            });
        });
    });

    Tscal rank_dt = cfl_dt.compute_rank_min();

    logger::debug_ln(
        "basegodunov", "rank", shamcomm::world_rank(), "found cfl dt =", rank_dt);

    Tscal next_cfl = shamalgs::collective::allreduce_min(rank_dt);

    if (shamcomm::world_rank() == 0) {
        logger::info_ln("amr::basegodunov", "cfl dt =", next_cfl);
    }

    return next_cfl;

}

template class shammodels::basegodunov::modules::ComputeCFL<f64_3, i64_3>;