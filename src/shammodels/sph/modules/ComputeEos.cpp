// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ComputeEos.cpp
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief 
 * 
 */
 
#include "ComputeEos.hpp"

#include "shambase/sycl_utils/sycl_utilities.hpp"
#include "shammath/sphkernels.hpp"
#include "shamsys/legacy/log.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"


template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::ComputeEos<Tvec, SPHKernel>::compute_eos(Tscal gpart_mass,Tscal eos_gamma) {

    NamedStackEntry stack_loc{"compute eos"};

    using namespace shamrock;

    shamrock::patch::PatchDataLayout &ghost_layout = storage.ghost_layout.get();
    u32 ihpart_interf                              = ghost_layout.get_field_idx<Tscal>("hpart");
    u32 iuint_interf                               = ghost_layout.get_field_idx<Tscal>("uint");

    shamrock::SchedulerUtility utility(scheduler());

    storage.pressure.set(utility.make_compute_field<Tscal>("pressure", 1, [&](u64 id) {
        return storage.merged_patchdata_ghost.get().get(id).total_elements;
    }));


    storage.merged_patchdata_ghost.get().for_each([&](u64 id, MergedPatchData &mpdat) {
        shamsys::instance::get_compute_queue().submit([&](sycl::handler &cgh) {
            sycl::accessor P{
                storage.pressure.get().get_buf_check(id), cgh, sycl::write_only, sycl::no_init};
            sycl::accessor U{
                mpdat.pdat.get_field_buf_ref<Tscal>(iuint_interf), cgh, sycl::read_only};
            sycl::accessor h{
                mpdat.pdat.get_field_buf_ref<Tscal>(ihpart_interf), cgh, sycl::read_only};

            Tscal pmass = gpart_mass;
            Tscal gamma = eos_gamma;

            cgh.parallel_for(sycl::range<1>{mpdat.total_elements}, [=](sycl::item<1> item) {
                using namespace shamrock::sph;
                P[item] = (gamma - 1) * rho_h(pmass, h[item], Kernel::hfactd) * U[item];
            });
        });
    });


}


using namespace shammath;
template class shammodels::sph::modules::ComputeEos<f64_3, M4>;
template class shammodels::sph::modules::ComputeEos<f64_3, M6>;