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
 * @file NodeSetEdge.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Field variant object to instanciate a variant on the patch types
 * @date 2023-07-31
 */

#include "shamrock/solvergraph/IEdge.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include <functional>

namespace shamrock::solvergraph {

    /**
     * @brief A node that simply frees the allocation of the connected node
     *
     * This node is useful to free the memory allocated by a node
     * that is no longer needed.
     */
    template<class Tnode>
    class NodeSetEdge : public INode {

        std::function<void(Tnode &)> set_edge;

        public:
        NodeSetEdge(std::function<void(Tnode &)> set_edge) : set_edge(set_edge) {}

        /**
         * @brief Set the edges of the node
         *
         * Set the edge that will be freed by this node
         *
         * @param to_free The node to free
         */
        inline void set_edges(std::shared_ptr<IEdge> to_set) {
            __internal_set_ro_edges({});
            __internal_set_rw_edges({to_set});
        }

        /// Evaluate the node
        inline void _impl_evaluate_internal() { set_edge(get_rw_edge<Tnode>(0)); }

        /// Get the label of the node
        inline virtual std::string _impl_get_label() { return "SetEdge"; };

        /// Get the TeX representation of the node
        inline virtual std::string _impl_get_tex() {

            auto to_set = get_rw_edge_base(0).get_tex_symbol();

            std::string tex = R"tex(
                Set edge ${to_set}$
            )tex";

            shambase::replace_all(tex, "{to_set}", to_set);

            return tex;
        }
    };

} // namespace shamrock::solvergraph
