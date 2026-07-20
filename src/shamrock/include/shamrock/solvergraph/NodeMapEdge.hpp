// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file NodeMapEdge.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Node that maps a read-only input edge into a read-write output edge
 */

#include "shambase/stacktrace.hpp"
#include "shamrock/solvergraph/IEdge.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include <functional>

namespace shamrock::solvergraph {

    /**
     * @brief A node that maps an input edge into an output edge
     *
     * This node executes a user-defined function that reads from a connected
     * read-only edge and writes to a connected read-write edge. The function is
     * applied during node evaluation.
     *
     * @tparam Tin  The type of the input (read-only) edge
     * @tparam Tout The type of the output (read-write) edge
     *
     * @code{.cpp}
     * // Example: Create a node that maps an input edge into an output edge
     * auto map_function = [](const MyInEdgeType &in_edge, MyOutEdgeType &out_edge) {
     *     out_edge.set_value(in_edge.get_value() + 1);
     * };
     * auto map_node = std::make_shared<NodeMapEdge<MyInEdgeType, MyOutEdgeType>>(map_function);
     * map_node->set_edges(my_in_edge, my_out_edge);
     * @endcode
     */
    template<class Tin, class Tout>
    class NodeMapEdge : public INode {

        std::function<void(const Tin &, Tout &)> map_edge;

        public:
        /**
         * @brief Construct a new NodeMapEdge object
         *
         * @param map_edge The function applied to (input, output) during evaluation
         */
        NodeMapEdge(std::function<void(const Tin &, Tout &)> map_edge)
            : map_edge(std::move(map_edge)) {}

        /**
         * @brief Set the edges of the node
         *
         * Configures the input edge as read-only and the output edge as
         * read-write. The custom function reads from @p in_edge and writes to
         * @p out_edge during evaluation.
         *
         * @param in_edge  The edge read by the custom function
         * @param out_edge The edge written by the custom function
         */
        inline void set_edges(std::shared_ptr<Tin> in_edge, std::shared_ptr<Tout> out_edge) {
            __internal_set_ro_edges({in_edge});
            __internal_set_rw_edges({out_edge});
        }

        /**
         * @brief Evaluate the node
         *
         * Applies the custom function to the connected read-only input edge and
         * read-write output edge.
         */
        inline void _impl_evaluate_internal() {
            __shamrock_stack_entry();
            map_edge(get_ro_edge<Tin>(0), get_rw_edge<Tout>(0));
        }

        /**
         * @brief Get the label of the node
         *
         * @return std::string The node label "MapEdge"
         */
        inline virtual std::string _impl_get_label() const { return "MapEdge"; };

        /**
         * @brief Get the TeX representation of the node
         *
         * @return std::string A TeX string describing the node operation
         */
        inline virtual std::string _impl_get_tex() const {

            auto in_edge  = get_ro_edge_base(0).get_tex_symbol();
            auto out_edge = get_rw_edge_base(0).get_tex_symbol();

            std::string tex = R"tex(
                Map edge ${in_edge}$ to ${out_edge}$
            )tex";

            shambase::replace_all(tex, "{in_edge}", in_edge);
            shambase::replace_all(tex, "{out_edge}", out_edge);

            return tex;
        }
    };

} // namespace shamrock::solvergraph
