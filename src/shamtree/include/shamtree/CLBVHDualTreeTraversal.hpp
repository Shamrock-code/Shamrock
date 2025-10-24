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
 * @brief Dual tree traversal algorithm for Compressed Leaf Bounding Volume Hierarchies
 *
 * This header provides algorithms for performing dual tree traversal on Compressed Leaf
 * Bounding Volume Hierarchies (CLBVH).
 */

#include "shamalgs/impl_utils.hpp"
#include "shambackends/vec.hpp"
#include "shamtree/CompressedLeafBVH.hpp"

namespace shamtree {

    /// Result structure for dual tree traversal operations.
    struct DTTResult {
        /// Pairs of nodes that interact using M2M interactions
        sham::DeviceBuffer<u32_2> node_interactions_m2m;
        /// Pairs of nodes that interact using P2P interactions
        sham::DeviceBuffer<u32_2> node_interactions_p2p;

        struct OrderedResult {
            sham::DeviceBuffer<u32> offset_m2m;
            sham::DeviceBuffer<u32> offset_p2p;
        };

        std::optional<OrderedResult> ordered_result;

        bool is_ordered() const { return ordered_result.has_value(); }
    };

    template<class Tmorton, class Tvec, u32 dim>
    DTTResult clbvh_dual_tree_traversal(
        sham::DeviceScheduler_ptr dev_sched,
        const CompressedLeafBVH<Tmorton, Tvec, dim> &bvh,
        shambase::VecComponent<Tvec> theta_crit,
        bool ordered_result = false);

    /// namespace to control implementation behavior
    namespace impl {

        /// Get list of available dual tree traversal implementations
        std::vector<shamalgs::impl_param> get_default_impl_list_clbvh_dual_tree_traversal();

        /// Get the current implementation for dual tree traversal
        shamalgs::impl_param get_current_impl_clbvh_dual_tree_traversal_impl();

        /// Set the implementation for dual tree traversal
        void set_impl_clbvh_dual_tree_traversal(
            const std::string &impl, const std::string &param = "");

    } // namespace impl

} // namespace shamtree
