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
 * @file OperationForUntil.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Meta node that evaluates a body node while a bool edge is true, for at most a given
 * number of iterations
 *
 */

#include "shambase/aliases_int.hpp"
#include "shamsolvergraph/edge/IDataEdge.hpp"
#include "shamsolvergraph/node/INode.hpp"

#define NODE_EDGES(X_RO, X_RW, X_RO_OPTIONAL, X_RW_OPTIONAL)                                       \
    X_RO(shamrock::solvergraph::IDataEdge<bool>, condition)                                        \
    X_RW_OPTIONAL(shamrock::solvergraph::IDataEdge<u32>, iter_count)

namespace shamrock::solvergraph {

    /**
     * @brief Variant of OperationFor that also stops as soon as `condition` becomes false,
     * equivalent to
     *
     * @code{.cpp}
     * u32 i = 0;
     * for (; i < max_iter && condition; i++) {
     *     body_node->evaluate(); // expected to update condition
     * }
     * iter_count = i; // if the optional iter_count edge is set
     * @endcode
     *
     * The condition is checked before each iteration, so the body may run zero times. Like
     * OperationFor, the loop always ends after at most `max_iter` iterations, even if the body
     * never sets `condition` to false.
     *
     * The optional `iter_count` edge holds the index of the current iteration while the body
     * runs (so the body can read it), and the number of evaluated iterations once the loop ends.
     * The body must not write it.
     *
     * Nested nodes are graph structure and are taken in the constructor (like
     * OperationSequence and OperationIf). The runtime predicate is an `IDataEdge<bool>` wired
     * with `set_edges`.
     */
    class OperationForUntil : public INode {
        std::shared_ptr<INode> body_node;
        u32 max_iter;
        std::string name;

        public:
        /**
         * @param name Display name for DOT / TeX
         * @param body_node Node evaluated at each iteration
         * @param max_iter Maximum number of iterations
         */
        OperationForUntil(std::string name, std::shared_ptr<INode> body_node, u32 max_iter);

        EXPAND_NODE_EDGES_OPTIONAL(NODE_EDGES)

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
