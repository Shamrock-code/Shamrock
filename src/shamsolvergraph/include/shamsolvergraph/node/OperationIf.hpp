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
 * @file OperationIf.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Meta node that evaluates an optional then-node when a bool edge is true, and an optional
 * else-node when false
 *
 */

#include "shamsolvergraph/edge/IDataEdge.hpp"
#include "shamsolvergraph/node/INode.hpp"

#define NODE_EDGES(X_RO, X_RW) X_RO(shamrock::solvergraph::IDataEdge<bool>, condition)

namespace shamrock::solvergraph {

    /**
     * @brief Conditional meta node: if `condition` is true, evaluate `then_node` when one was
     * provided; otherwise evaluate `else_node` when one was provided.
     *
     * Nested nodes are graph structure and are taken in the constructor (like
     * OperationSequence). The runtime predicate is an `IDataEdge<bool>` wired with `set_edges`.
     */
    class OperationIf : public INode {
        std::shared_ptr<INode> then_node;
        std::shared_ptr<INode> else_node;
        std::string name;

        public:
        /**
         * @param name Display name for DOT / TeX
         * @param then_node Node evaluated when the condition edge is true (optional)
         * @param else_node Node evaluated when the condition edge is false (optional)
         */
        OperationIf(
            std::string name,
            std::shared_ptr<INode> then_node = {},
            std::shared_ptr<INode> else_node = {})
            : then_node(std::move(then_node)), else_node(std::move(else_node)),
              name(std::move(name)) {}

        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal();

        inline std::string _impl_get_label() const { return name; }

        std::string _impl_get_dot_graph_partial() const;

        inline virtual std::string _impl_get_dot_graph_node_start() const {
            return sham::format("n_{}", get_uuid());
        }
        inline virtual std::string _impl_get_dot_graph_node_end() const {
            return sham::format("n_{}_end", get_uuid());
        }

        std::string _impl_get_tex() const;
    };

} // namespace shamrock::solvergraph

#undef NODE_EDGES
