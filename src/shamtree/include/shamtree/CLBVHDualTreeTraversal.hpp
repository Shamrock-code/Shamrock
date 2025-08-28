// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file CLBVHDualTreeTraversal.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "shambackends/vec.hpp"
#include "shamtree/CompressedLeafBVH.hpp"

namespace shamtree {

    struct DTTResult {
        sham::DeviceBuffer<u32_2> node_node_interactions;
        sham::DeviceBuffer<u32_2> leaf_leaf_interactions;
    };

    template<class Tmorton, class Tvec, u32 dim>
    DTTResult clbvh_dual_tree_traversal(
        sham::DeviceScheduler_ptr dev_sched,
        const CompressedLeafBVH<Tmorton, Tvec, dim> &bvh,
        shambase::VecComponent<Tvec> theta_crit);

} // namespace shamtree
