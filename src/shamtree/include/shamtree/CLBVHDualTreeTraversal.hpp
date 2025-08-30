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

#include "shambackends/vec.hpp"
#include "shamtree/CompressedLeafBVH.hpp"

namespace shamtree {

    /// Result structure for dual tree traversal operations.
    struct DTTResult {
        /// Pairs of nodes that interact using M2M interactions
        sham::DeviceBuffer<u32_2> node_interactions_m2m;
        /// Pairs of nodes that interact using P2P interactions
        sham::DeviceBuffer<u32_2> node_interactions_p2p;
    };

    template<class Tmorton, class Tvec, u32 dim>
    DTTResult clbvh_dual_tree_traversal(
        sham::DeviceScheduler_ptr dev_sched,
        const CompressedLeafBVH<Tmorton, Tvec, dim> &bvh,
        shambase::VecComponent<Tvec> theta_crit);

    namespace impl {

        /**
         * @brief Get list of available dual tree traversal implementations
         *
         * Returns a list of all available algorithm implementations that can be
         * used with the dual tree traversal function. Each implementation has
         * different performance characteristics and is suitable for different
         * use cases.
         *
         * @return Vector of implementation names as strings
         *
         * Available implementations:
         * - "reference": CPU-based reference implementation (slow but accurate)
         * - "parallel_select": GPU parallel selection algorithm
         * - "scan_multipass": GPU scan-based multipass algorithm (default)
         */
        std::vector<std::string> get_impl_list_clbvh_dual_tree_traversal();

        /**
         * @brief Set the implementation for dual tree traversal
         *
         * Selects which algorithm implementation to use for subsequent calls
         * to clbvh_dual_tree_traversal(). This setting affects global state
         * and applies to all future traversals until changed.
         *
         * @param impl Implementation name (must be from get_impl_list_clbvh_dual_tree_traversal())
         * @param param Optional implementation-specific parameter (currently unused)
         *
         * @throws std::invalid_argument if impl is not a valid implementation name
         *
         * @note The default implementation is "scan_multipass"
         */
        void
        set_impl_clbvh_dual_tree_traversal(const std::string &impl, const std::string &param = "");

    } // namespace impl

} // namespace shamtree
