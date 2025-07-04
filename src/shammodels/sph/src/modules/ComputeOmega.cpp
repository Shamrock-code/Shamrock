// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ComputeOmega.cpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 *
 */

#include "shammodels/sph/modules/ComputeOmega.hpp"
#include "shammodels/sph/SPHUtilities.hpp"
#include "shamrock/scheduler/SchedulerUtility.hpp"

template<class Tvec, template<class> class SPHKernel>
auto shammodels::sph::modules::ComputeOmega<Tvec, SPHKernel>::compute_omega()
    -> shamrock::ComputeField<Tscal> {

    NamedStackEntry stack_loc{"compute omega"};

    shamrock::SchedulerUtility utility(scheduler());
    using SPHUtils = sph::SPHUtilities<Tvec, Kernel>;
    SPHUtils sph_utils(scheduler());

    using namespace shamrock;
    using namespace shamrock::patch;

    PatchDataLayout &pdl = scheduler().pdl;
    const u32 ihpart     = pdl.get_field_idx<Tscal>("hpart");

    ComputeField<Tscal> omega = utility.make_compute_field<Tscal>("omega", 1);

    scheduler().for_each_patchdata_nonempty([&](const Patch p, PatchData &pdat) {
        shamlog_debug_ln("SPHLeapfrog", "patch : n°", p.id_patch, "->", "compute omega");

        sham::DeviceBuffer<Tscal> &omega_h = omega.get_buf(p.id_patch);

        sham::DeviceBuffer<Tscal> &hnew = pdat.get_field<Tscal>(ihpart).get_buf();
        sham::DeviceBuffer<Tvec> &merged_r
            = storage.merged_xyzh.get().get(p.id_patch).field_pos.get_buf();

        sycl::range range_npart{pdat.get_obj_cnt()};

        tree::ObjectCache &neigh_cache
            = shambase::get_check_ref(storage.neigh_cache).get_cache(p.id_patch);

        sph_utils.compute_omega(
            merged_r, hnew, omega_h, range_npart, neigh_cache, solver_config.gpart_mass);
    });

    return omega;
}

using namespace shammath;
template class shammodels::sph::modules::ComputeOmega<f64_3, M4>;
template class shammodels::sph::modules::ComputeOmega<f64_3, M6>;
template class shammodels::sph::modules::ComputeOmega<f64_3, M8>;
