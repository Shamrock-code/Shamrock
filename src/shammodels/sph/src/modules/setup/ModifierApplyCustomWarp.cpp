// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ModifierApplyCustomWarp.cpp
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/constants.hpp"
#include "shamalgs/collective/indexing.hpp"
#include "shamcomm/logs.hpp"
#include "shammodels/sph/Solver.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shammodels/sph/modules/setup/ISPHSetupNode.hpp"
#include "shammodels/sph/modules/setup/ModifierApplyCustomWarp.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"

template<class Tvec, template<class> class SPHKernel>
shamrock::patch::PatchDataLayer shammodels::sph::modules::ModifierApplyCustomWarp<Tvec, SPHKernel>::
    next_n(u32 nmax) {

    ShamrockCtx &ctx                    = context;
    PatchScheduler &sched               = shambase::get_check_ref(ctx.sched);
    shamrock::patch::PatchDataLayer tmp = parent->next_n(nmax);

    ////////////////////////// load data //////////////////////////
    auto &pdl                         = sched.pdl();
    sham::DeviceBuffer<Tvec> &buf_xyz = tmp.get_field_buf_ref<Tvec>(pdl.get_field_idx<Tvec>("xyz"));
    sham::DeviceBuffer<Tvec> &buf_vxyz
        = tmp.get_field_buf_ref<Tvec>(pdl.get_field_idx<Tvec>("vxyz"));

    auto acc_xyz  = buf_xyz.copy_to_stdvec();
    auto acc_vxyz = buf_vxyz.copy_to_stdvec();

    for (i32 id_a = 0; id_a < tmp.get_obj_cnt(); ++id_a) {
        Tvec &xyz_a  = acc_xyz[id_a];
        Tvec &vxyz_a = acc_vxyz[id_a];

        Tscal r = sycl::sqrt(sycl::dot(xyz_a, xyz_a));

        Tvec k              = k_profile(r);
        Tscal psi           = psi_profile(r);
        Tscal effective_inc = inc_profile(r);

        Tvec w  = sycl::cross(k, xyz_a);
        Tvec wv = sycl::cross(k, vxyz_a);
        // Rodrigues' rotation formula
        xyz_a = xyz_a * sycl::cos(effective_inc) + w * sycl::sin(effective_inc)
                + k * sycl::dot(k, xyz_a) * (1. - sycl::cos(effective_inc));
        vxyz_a = vxyz_a * sycl::cos(effective_inc) + wv * sycl::sin(effective_inc)
                 + k * sycl::dot(k, vxyz_a) * (1. - sycl::cos(effective_inc));
    };

    buf_xyz.copy_from_stdvec(acc_xyz);
    buf_vxyz.copy_from_stdvec(acc_vxyz);

    return tmp;
}

using namespace shammath;
template class shammodels::sph::modules::ModifierApplyCustomWarp<f64_3, M4>;
template class shammodels::sph::modules::ModifierApplyCustomWarp<f64_3, M6>;
template class shammodels::sph::modules::ModifierApplyCustomWarp<f64_3, M8>;

template class shammodels::sph::modules::ModifierApplyCustomWarp<f64_3, C2>;
template class shammodels::sph::modules::ModifierApplyCustomWarp<f64_3, C4>;
template class shammodels::sph::modules::ModifierApplyCustomWarp<f64_3, C6>;
