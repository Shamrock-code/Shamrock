// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file ComputeLoadBalanceValue.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shammodels/sph/modules/ComputeLoadBalanceValue.hpp"
#include "shammath/sphkernels.hpp"
#include "shamsys/legacy/log.hpp"

template<class Tvec, template<class> class SPHKernel>
void shammodels::sph::modules::ComputeLoadBalanceValue<Tvec, SPHKernel>::update_load_balancing() {
    StackEntry stack_loc{};

    shamlog_debug_ln("ComputeLoadBalanceValue", "update load balancing");
    scheduler().update_local_load_value([&](shamrock::patch::Patch p) {
        return scheduler().patch_data.owned_data.get(p.id_patch).get_obj_cnt();
    });
}

using namespace shammath;
template class shammodels::sph::modules::ComputeLoadBalanceValue<f64_3, M4>;
template class shammodels::sph::modules::ComputeLoadBalanceValue<f64_3, M6>;
template class shammodels::sph::modules::ComputeLoadBalanceValue<f64_3, M8>;
