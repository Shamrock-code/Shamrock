// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shammodels/amr/zeus/modules/ComputePressure.hpp"
#include "shambase/stacktrace.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"
#include "shamrock/scheduler/InterfacesUtility.hpp"

template<class Tvec, class TgridVec>
using Module = shammodels::zeus::modules::ComputePressure<Tvec, TgridVec>;

template<class Tvec, class TgridVec>
void Module<Tvec, TgridVec>::compute_p(){

    StackEntry stack_loc{};

    using MergedPDat = shambase::DistributedData<shamrock::MergedPatchData>;

    shamrock::SchedulerUtility utility(scheduler());

    using Block = typename Config::AMRBlock;

    storage.pressure.set(utility.make_compute_field<Tscal>("pressure", Block::block_size, [&](u64 id) {
        return storage.merged_patchdata_ghost.get().get(id).total_elements;
    }));

    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 irho_interf      = ghost_layout.get_field_idx<Tscal>("rho");
    u32 ieint_interf     = ghost_layout.get_field_idx<Tscal>("eint");

    storage.merged_patchdata_ghost.get().for_each([&](u64 id, shamrock::MergedPatchData &mpdat) {
        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
            sycl::accessor P{
                storage.pressure.get().get_buf_check(id), cgh, sycl::write_only, sycl::no_init};
            sycl::accessor eint{
                mpdat.pdat.get_field_buf_ref<Tscal>(ieint_interf), cgh, sycl::read_only};
            sycl::accessor rho{
                mpdat.pdat.get_field_buf_ref<Tscal>(irho_interf), cgh, sycl::read_only};

            Tscal gamma = solver_config.eos_gamma;

            cgh.parallel_for(sycl::range<1>{mpdat.total_elements*Block::block_size}, [=](sycl::item<1> item) {
                P[item] = (gamma - 1) * rho[item] * eint[item];
            });
        });
    });

}

template class shammodels::zeus::modules::ComputePressure<f64_3, i64_3>;