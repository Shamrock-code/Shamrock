// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
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

#include "shambase/constants.hpp"
#include "shamalgs/collective/indexing.hpp"
#include "shammodels/sph/Solver.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shammodels/sph/modules/setup/ISPHSetupNode.hpp"
#include "shammodels/sph/modules/setup/ModifierOffset.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"

template<class Tvec, template<class> class SPHKernel>
shamrock::patch::PatchDataLayer
shammodels::sph::modules::ModifierOffset<Tvec, SPHKernel>::next_n(u32 nmax) {

    using Config = SolverConfig<Tvec, SPHKernel>;
    Config solver_config;
    ShamrockCtx &ctx                    = context;
    PatchScheduler &sched               = shambase::get_check_ref(ctx.sched);
    shamrock::patch::PatchDataLayer tmp = parent->next_n(nmax);

    ////////////////////////// load data //////////////////////////
    sham::DeviceBuffer<Tvec> &buf_xyz
        = tmp.get_field_buf_ref<Tvec>(sched.pdl.get_field_idx<Tvec>("xyz"));
    sham::DeviceBuffer<Tvec> &buf_vxyz
        = tmp.get_field_buf_ref<Tvec>(sched.pdl.get_field_idx<Tvec>("vxyz"));

    auto &q = shamsys::instance::get_compute_scheduler().get_queue();
    sham::EventList depends_list;

    auto acc_xyz  = buf_xyz.get_write_access(depends_list);
    auto acc_vxyz = buf_vxyz.get_write_access(depends_list);

    auto e = q.submit(depends_list, [&](sycl::handler &cgh) {
        Tvec positional_offset = this->positional_offset;
        Tvec velocity_offset   = this->velocity_offset;

        shambase::parallel_for(cgh, tmp.get_obj_cnt(), "Warp", [=](i32 id_a) {
            Tvec &xyz_a  = acc_xyz[id_a];
            Tvec &vxyz_a = acc_vxyz[id_a];

            xyz_a += positional_offset;
            vxyz_a += velocity_offset;
        });
    });

    buf_xyz.complete_event_state(e);
    buf_vxyz.complete_event_state(e);

    return tmp;
}

using namespace shammath;
template class shammodels::sph::modules::ModifierOffset<f64_3, M4>;
template class shammodels::sph::modules::ModifierOffset<f64_3, M6>;
template class shammodels::sph::modules::ModifierOffset<f64_3, M8>;

template class shammodels::sph::modules::ModifierOffset<f64_3, C2>;
template class shammodels::sph::modules::ModifierOffset<f64_3, C4>;
template class shammodels::sph::modules::ModifierOffset<f64_3, C6>;
