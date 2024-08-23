// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ComputePressure.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/stacktrace.hpp"
#include "shammodels/amr/zeus/modules/ComputePressure.hpp"
#include "shamrock/scheduler/InterfacesUtility.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"

template<class Tvec, class TgridVec>
void shammodels::zeus::modules::ComputePressure<Tvec, TgridVec>::compute_p() {

    StackEntry stack_loc{};

    using namespace shamrock::patch;
    using namespace shamrock;
    using namespace shammath;
    using MergedPDat = shamrock::MergedPatchData;

    shamrock::SchedulerUtility utility(scheduler());

    using Block = typename Config::AMRBlock;

    storage.pressure.set(
        utility.make_compute_field<Tscal>("pressure", Block::block_size, [&](u64 id) {
            return storage.merged_patchdata_ghost.get().get(id).total_elements;
        }));

    ComputeField<Tscal> &pressure_field = storage.pressure.get();

    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 irho_interf                                = ghost_layout.get_field_idx<Tscal>("rho");
    u32 ieint_interf                               = ghost_layout.get_field_idx<Tscal>("eint");

    scheduler().for_each_patchdata_nonempty([&](Patch p, PatchData &pdat) {
        MergedPDat &mpdat = storage.merged_patchdata_ghost.get().get(p.id_patch);

        PatchDataField<Tscal> &press = storage.pressure.get().get_field(p.id_patch);

        sycl::buffer<Tscal> &buf_p    = pressure_field.get_buf_check(p.id_patch);
        sycl::buffer<Tscal> &buf_rho  = mpdat.pdat.get_field_buf_ref<Tscal>(irho_interf);
        sycl::buffer<Tscal> &buf_eint = mpdat.pdat.get_field_buf_ref<Tscal>(ieint_interf);

        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
            sycl::accessor rho{buf_rho, cgh, sycl::read_only};
            sycl::accessor eint{buf_eint, cgh, sycl::read_only};
            sycl::accessor p{buf_p, cgh, sycl::write_only, sycl::no_init};

            Tscal gamma = solver_config.eos_gamma;

            shambase::parralel_for(
                cgh, mpdat.total_elements * Block::block_size, "compute pressure", [=](u64 id_a) {
                    p[id_a] = (gamma - 1) /** rho[id_a]*/ * eint[id_a];
                });
        });
    });
}

template class shammodels::zeus::modules::ComputePressure<f64_3, i64_3>;
