// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ModifierFilter.cpp
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shambase/constants.hpp"
#include "shamalgs/collective/indexing.hpp"
#include "shambackends/DeviceBuffer.hpp"
#include "shamcomm/logs.hpp"
#include "shammodels/sph/Solver.hpp"
#include "shammodels/sph/math/density.hpp"
#include "shammodels/sph/modules/setup/ISPHSetupNode.hpp"
#include "shammodels/sph/modules/setup/ModifierFilter.hpp"
#include "shamrock/scheduler/ShamrockCtx.hpp"
#include "shamsys/NodeInstance.hpp"
#include <vector>

template<class Tvec, template<class> class SPHKernel>
shamrock::patch::PatchData
shammodels::sph::modules::ModifierFilter<Tvec, SPHKernel>::next_n(u32 nmax) {

    using Config = SolverConfig<Tvec, SPHKernel>;
    Config solver_config;
    ShamrockCtx &ctx               = context;
    PatchScheduler &sched          = shambase::get_check_ref(ctx.sched);
    shamrock::patch::PatchData tmp = parent->next_n(nmax);

    ////////////////////////// load data //////////////////////////
    sham::DeviceBuffer<Tvec> &buf_xyz
        = tmp.get_field_buf_ref<Tvec>(sched.pdl.get_field_idx<Tvec>("xyz"));
    sham::DeviceBuffer<Tvec> &buf_vxyz
        = tmp.get_field_buf_ref<Tvec>(sched.pdl.get_field_idx<Tvec>("vxyz"));

    std::vector<Tvec> pos = buf_xyz.copy_to_stdvec();

    std::vector<u32> idx_keep = {};
    for (u32 i = 0; i < pos.size(); i++) {
        if (this->filter(pos[i])) {
            idx_keep.push_back(i);
        }
    }

    if (idx_keep.empty()) {
        tmp.resize(0);
    } else {
        sham::DeviceBuffer<u32> filter_idx_buf(
            idx_keep.size(), shamsys::instance::get_compute_scheduler_ptr());

        filter_idx_buf.copy_from_stdvec(idx_keep);

        tmp.keep_ids(filter_idx_buf, idx_keep.size());
    }

    return tmp;
}

using namespace shammath;
template class shammodels::sph::modules::ModifierFilter<f64_3, M4>;
template class shammodels::sph::modules::ModifierFilter<f64_3, M6>;
template class shammodels::sph::modules::ModifierFilter<f64_3, M8>;
