// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ModifierOffset.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shammodels/sph/modules/setup/ModifierOffset.hpp"
#include "shambackends/kernel_call.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"

template<class Tvec>
shamrock::patch::PatchDataLayer shammodels::sph::modules::ModifierOffset<Tvec>::next_n(u32 nmax) {

    ShamrockCtx &ctx                    = context;
    PatchScheduler &sched               = shambase::get_check_ref(ctx.sched);
    shamrock::patch::PatchDataLayer tmp = parent->next_n(nmax);

    // No objects to offset
    if (tmp.get_obj_cnt() == 0) {
        return tmp;
    }

    sham::DeviceBuffer<Tvec> &buf_xyz
        = tmp.get_field_buf_ref<Tvec>(sched.pdl_old().get_field_idx<Tvec>("xyz"));
    sham::DeviceBuffer<Tvec> &buf_vxyz
        = tmp.get_field_buf_ref<Tvec>(sched.pdl_old().get_field_idx<Tvec>("vxyz"));

    auto &q = shamsys::instance::get_compute_scheduler().get_queue();

    Tvec positional_offset = this->positional_offset;
    Tvec velocity_offset   = this->velocity_offset;

    sham::kernel_call(
        q,
        sham::MultiRef{},
        sham::MultiRef{buf_xyz, buf_vxyz},
        tmp.get_obj_cnt(),
        [positional_offset,
         velocity_offset](u32 i, Tvec *__restrict__ xyz_a, Tvec *__restrict__ vxyz_a) {
            xyz_a[i] += positional_offset;
            vxyz_a[i] += velocity_offset;
        });

    return tmp;
}

using namespace shammath;
template class shammodels::sph::modules::ModifierOffset<f64_3>;
