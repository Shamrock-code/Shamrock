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
 * @brief Node that applies a custom function to modify connected edges
 * @date 2023-07-31
 */

#include "shamrock/solvergraph/IEdge.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include <functional>

namespace shamrock::solvergraph {

    /**
     * @brief A node that applies a custom function to modify connected edges
     *
     * This node executes a user-defined function on a connected edge,
     * allowing for custom modifications or operations on the edge data.
     * The function is applied during node evaluation.
     *
     * @tparam Tedge The type of the edge that this node operates on
     *
     * @code{.cpp}
     * // Example: Create a node that sets values in an edge
     * auto setter_function = [](MyEdgeType& edge) {
     *     edge.set_value(42);
     * };
     * auto set_node = std::make_shared<NodeSetEdge<MyEdgeType>>(setter_function);
     * set_node->set_edges(my_edge);
     * @endcode
     */
    template<class Tedge>
    class NodeSetEdge : public INode {

        std::function<void(Tedge &)> set_edge;

        public:
        /**
         * @brief Construct a new NodeSetEdge object
         *
         * @param set_edge The function to apply to the connected edge during evaluation
         */
        NodeSetEdge(std::function<void(Tedge &)> set_edge) : set_edge(std::move(set_edge)) {}

        /**
         * @brief Set the edges of the node
         *
         * Configures the edge that will be modified by this node.
         * The edge is set as a read-write edge, allowing the custom
         * function to modify its contents during evaluation.
         *
         * @param to_set The edge to be modified by the custom function
         */
        inline void set_edges(std::shared_ptr<IEdge> to_set) {
            __internal_set_ro_edges({});
            __internal_set_rw_edges({to_set});
        }

        /**
         * @brief Evaluate the node
         *
         * Applies the custom function to the connected read-write edge.
         * This is where the actual modification of the edge takes place.
         */
        inline void _impl_evaluate_internal() { set_edge(get_rw_edge<Tedge>(0)); }

        /**
         * @brief Get the label of the node
         *
         * @return std::string The node label "SetEdge"
         */
        inline virtual std::string _impl_get_label() { return "SetEdge"; };

        /**
         * @brief Get the TeX representation of the node
         *
         * @return std::string A TeX string describing the node operation
         */
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
