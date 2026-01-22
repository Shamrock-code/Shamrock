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
 * @file OperationSequence.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamrock/solvergraph/INode.hpp"

namespace shamrock::solvergraph {

    class OperationSequence : public INode {
        std::vector<std::shared_ptr<INode>> nodes;
        std::string name;

        public:
        OperationSequence(std::string name, std::vector<std::shared_ptr<INode>> &&_nodes)
            : nodes(std::forward<std::vector<std::shared_ptr<INode>>>(_nodes)), name(name) {
            if (nodes.size() == 0) {
                shambase::throw_with_loc<std::invalid_argument>(
                    "OperationSequence must have at least one node");
            }
        }
        void _impl_evaluate_internal();

        inline std::string _impl_get_label() const { return name; }

        std::string _impl_get_dot_graph_partial() const;

        inline virtual std::string _impl_get_dot_graph_node_start() const {
            return nodes[0]->get_dot_graph_node_start();
        }
        inline virtual std::string _impl_get_dot_graph_node_end() const {
            return nodes[nodes.size() - 1]->get_dot_graph_node_end();
        }

        std::string _impl_get_tex() const;
    };

} // namespace shamrock::solvergraph
