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
 * @file NodeFreeAlloc.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Field variant object to instanciate a variant on the patch types
 * @date 2023-07-31
 */

#include "shamrock/solvergraph/IEdge.hpp"
#include "shamrock/solvergraph/INode.hpp"

namespace shamrock::solvergraph {

    /**
     * @brief A node that simply frees the allocation of the connected node
     *
     * This node is useful to free the memory allocated by a node
     * that is no longer needed.
     */
    class NodeFreeAlloc : public INode {

        public:
        NodeFreeAlloc() {}

        /// Utility struct to list the refs to the edges
        struct Edges {
            /**
             * @brief The node to free
             *
             * The node to free must be an instance of IFreeable
             */
            IFreeable &to_free;
        };

        /**
         * @brief Set the edges of the node
         *
         * Set the edge that will be freed by this node
         *
         * @param to_free The node to free
         */
        inline void set_edges(std::shared_ptr<IEdge> to_free) {
            __internal_set_ro_edges({});
            __internal_set_rw_edges({to_free});
        }

        /// Get the edges of the node
        inline Edges get_edges() { return Edges{get_rw_edge<IFreeable>(0)}; }

        /// Evaluate the node
        inline void _impl_evaluate_internal() { get_edges().to_free.free_alloc(); }

        /// Get the label of the node
        inline virtual std::string _impl_get_label() const { return "FreeAlloc"; };

        /// Get the TeX representation of the node
        inline virtual std::string _impl_get_tex() const {

            auto to_free = get_rw_edge_base(0).get_tex_symbol();

            std::string tex = R"tex(
                Free allocation ${to_free}$
            )tex";

            shambase::replace_all(tex, "{to_free}", to_free);

            return tex;
        }
    };

} // namespace shamrock::solvergraph
