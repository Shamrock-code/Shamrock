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
 * @file OperationRepeat.hpp
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief Meta node that evaluates a node a number of times
 *
 */

#include "shamsolvergraph/edge/IDataEdge.hpp"
#include "shamsolvergraph/node/INode.hpp"

#define NODE_EDGES(X_RO, X_RW) X_RO(shamrock::solvergraph::IDataEdge<u32>, count)

namespace shamrock::solvergraph {

    /**
     * @brief Loop meta node: evaluate `body` `count` times.
     */
    class OperationRepeat : public INode {
        std::shared_ptr<INode> body;
        std::string name;

        public:
        /**
         * @param name Display name for DOT / TeX
         * @param body Node evaluated `count` times
         */
        OperationRepeat(std::string name, std::shared_ptr<INode> body)
            : body(std::move(body)), name(std::move(name)) {}

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
