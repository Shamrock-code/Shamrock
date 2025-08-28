// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file CLBVHDualTreeTraversal.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shamtree/details/dtt_reference.hpp"

namespace shamtree {

    template<class Tmorton, class Tvec, u32 dim>
    inline DTTResult clbvh_dual_tree_traversal(
        sham::DeviceScheduler_ptr dev_sched,
        const CompressedLeafBVH<Tmorton, Tvec, dim> &bvh,
        shambase::VecComponent<Tvec> theta_crit) {

        return details::DTTCpuReference<Tmorton, Tvec, dim>::dtt(dev_sched, bvh, theta_crit);
    }

    template DTTResult clbvh_dual_tree_traversal<u64, f64_3, 3>(
        sham::DeviceScheduler_ptr dev_sched,
        const CompressedLeafBVH<u64, f64_3, 3> &bvh,
        shambase::VecComponent<f64_3> theta_crit);

} // namespace shamtree
