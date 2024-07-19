// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ConsToPrim.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shammodels/amr/basegodunov/modules/ConsToPrim.hpp"
#include "shammath/riemann.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"

template<class Tvec, class TgridVec>
void shammodels::basegodunov::modules::ConsToPrim<Tvec, TgridVec>::cons_to_prim() {

    StackEntry stack_loc{};

    using MergedPDat = shamrock::MergedPatchData;

    shamrock::SchedulerUtility utility(scheduler());

    shamrock::ComputeField<Tvec> v_ghost = utility.make_compute_field<Tvec>(
        "vel", AMRBlock::block_size, [&](u64 id) {
            return storage.merged_patchdata_ghost.get().get(id).total_elements;
        });

    shamrock::ComputeField<Tscal> P_ghost = utility.make_compute_field<Tscal>(
        "P", AMRBlock::block_size, [&](u64 id) {
            return storage.merged_patchdata_ghost.get().get(id).total_elements;
        });

    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 irho_ghost                                 = ghost_layout.get_field_idx<Tscal>("rho");
    u32 irhov_ghost                                = ghost_layout.get_field_idx<Tvec>("rhovel");
    u32 irhoe_ghost                                = ghost_layout.get_field_idx<Tscal>("rhoetot");

    storage.merged_patchdata_ghost.get().for_each([&](u64 id, MergedPDat &mpdat) {

        sycl::queue &q = shamsys::instance::get_compute_queue();

        sycl::buffer<TgridVec> &buf_block_min = mpdat.pdat.get_field_buf_ref<TgridVec>(0);
        sycl::buffer<TgridVec> &buf_block_max = mpdat.pdat.get_field_buf_ref<TgridVec>(1);

        sycl::buffer<Tscal> &buf_rho  = mpdat.pdat.get_field_buf_ref<Tscal>(irho_ghost);
        sycl::buffer<Tvec> &buf_rhov  = mpdat.pdat.get_field_buf_ref<Tvec>(irhov_ghost);
        sycl::buffer<Tscal> &buf_rhoe = mpdat.pdat.get_field_buf_ref<Tscal>(irhoe_ghost);

        q.submit([&](sycl::handler &cgh) {

            sycl::accessor rho{buf_rho, cgh, sycl::read_only};
            sycl::accessor rhovel{buf_rhov, cgh, sycl::read_only};
            sycl::accessor rhoe{buf_rhoe, cgh, sycl::read_only};

            sycl::accessor vel{
                shambase::get_check_ref(v_ghost.get_buf(id)), cgh, sycl::write_only, sycl::no_init};
            sycl::accessor P{
                shambase::get_check_ref(P_ghost.get_buf(id)), cgh, sycl::write_only, sycl::no_init};

            Tscal gamma  = solver_config.eos_gamma;

            u32 cell_count = (mpdat.total_elements) * AMRBlock::block_size;

            shambase::parralel_for(cgh, cell_count, "cons_to_prim", [=](u64 gid) {

                auto conststate = shammath::ConsState<Tvec>{rho[gid], rhoe[gid], rhovel[gid]};
                
                auto prim_state = shammath::cons_to_prim(conststate, gamma);
                
                vel[gid] = prim_state.vel;
                P[gid] = prim_state.press;
            });

        });

    });

    storage.vel.set(std::move(v_ghost));
    storage.press.set(std::move(P_ghost));
}

template class shammodels::basegodunov::modules::ConsToPrim<f64_3, i64_3>;