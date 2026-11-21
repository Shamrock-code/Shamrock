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
 * @file OperationWhile.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Meta node that repeatedly evaluates a body node while a bool edge is true
 *
 */

#include "shambase/exception.hpp"
#include "shamsolvergraph/edge/IDataEdge.hpp"
#include "shamsolvergraph/node/INode.hpp"
#include <stdexcept>

#define NODE_EDGES(X_RO, X_RW) X_RO(shamrock::solvergraph::IDataEdge<bool>, condition)

namespace shamrock::solvergraph {

    /**
     * @brief Loop meta node: while `condition` is true, evaluate `body_node`.
     *
     * The condition is checked before each iteration (so the body may run zero times). The
     * body is responsible for eventually setting `condition` to false (typically through a
     * read-write edge pointing to the same `IDataEdge<bool>`), otherwise the loop never ends.
     *
     * Nested nodes are graph structure and are taken in the constructor (like
     * OperationSequence and OperationIf). The runtime predicate is an `IDataEdge<bool>` wired
     * with `set_edges`.
     */
    class OperationWhile : public INode {
        std::shared_ptr<INode> body_node;
        std::string name;

        public:
        /**
         * @param name Display name for DOT / TeX
         * @param body_node Node evaluated at each iteration while the condition edge is true
         */
        OperationWhile(std::string name, std::shared_ptr<INode> body_node)
            : body_node(std::move(body_node)), name(std::move(name)) {
            if (!this->body_node) {
                shambase::throw_with_loc<std::invalid_argument>(
                    "OperationWhile must have a body node");
            }
        }

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
