// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file TimeIntegrator.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */
#include "shammodels/amr/basegodunov/modules/TimeIntegrator.hpp"

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
    PatchDataLayout &pdl = scheduler().pdl;

    const u32 icell_min = pdl.get_field_idx<TgridVec>("cell_min");
    const u32 icell_max = pdl.get_field_idx<TgridVec>("cell_max");
    const u32 irho      = pdl.get_field_idx<Tscal>("rho");
    const u32 irhoetot  = pdl.get_field_idx<Tscal>("rhoetot");
    const u32 irhovel   = pdl.get_field_idx<Tvec>("rhovel");

    scheduler().for_each_patchdata_nonempty(
        [&, dt](const shamrock::patch::Patch p, shamrock::patch::PatchData &pdat) {
            logger::debug_ln("[AMR Flux]", "forward euler integration patch", p.id_patch);

            sycl::queue &q = shamsys::instance::get_compute_queue();
            u32 id         = p.id_patch;

            sycl::buffer<Tscal> &dt_rho_patch  = cfield_dtrho.get_buf_check(id);
            sycl::buffer<Tvec> &dt_rhov_patch  = cfield_dtrhov.get_buf_check(id);
            sycl::buffer<Tscal> &dt_rhoe_patch = cfield_dtrhoe.get_buf_check(id);

            u32 cell_count = pdat.get_obj_cnt() * AMRBlock::block_size;

            sycl::buffer<Tscal> &buf_rho  = pdat.get_field_buf_ref<Tscal>(irho);
            sycl::buffer<Tvec> &buf_rhov  = pdat.get_field_buf_ref<Tvec>(irhovel);
            sycl::buffer<Tscal> &buf_rhoe = pdat.get_field_buf_ref<Tscal>(irhoetot);

            q.submit([&, dt](sycl::handler &cgh) {
                sycl::accessor acc_dt_rho_patch{dt_rho_patch, cgh, sycl::read_only};
                sycl::accessor acc_dt_rhov_patch{dt_rhov_patch, cgh, sycl::read_only};
                sycl::accessor acc_dt_rhoe_patch{dt_rhoe_patch, cgh, sycl::read_only};

                sycl::accessor rho{buf_rho, cgh, sycl::read_write};
                sycl::accessor rhov{buf_rhov, cgh, sycl::read_write};
                sycl::accessor rhoe{buf_rhoe, cgh, sycl::read_write};

                shambase::parralel_for(cgh, cell_count, "accumulate fluxes", [=](u32 id_a) {
                    const u32 cell_global_id = (u32) id_a;

                    rho[id_a] += dt * acc_dt_rho_patch[id_a];
                    rhov[id_a] += dt * acc_dt_rhov_patch[id_a];
                    rhoe[id_a] += dt * acc_dt_rhoe_patch[id_a];
                });
            });
        });
}

template class shammodels::basegodunov::modules::TimeIntegrator<f64_3, i64_3>;
