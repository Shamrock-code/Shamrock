// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file OperationSequence.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamrock/solvergraph/OperationSequence.hpp"
#include <sstream>

namespace shamrock::solvergraph {

    void OperationSequence::_impl_evaluate_internal() {
        for (auto &node : nodes) {
            node->evaluate();
        }
    }

    std::string OperationSequence::_impl_get_dot_graph_partial() {

        std::stringstream ss;

        ss << "subgraph cluster_" + std::to_string(get_uuid()) + " {\n";
        for (auto &node : nodes) {
            ss << node->get_dot_graph_partial();
        }

        for (int i = 0; i < nodes.size() - 1; i++) {
            ss << nodes[i]->get_dot_graph_node_end() << " -> "
               << nodes[i + 1]->get_dot_graph_node_start() << " [weight=3];\n";
        }

        ss << shambase::format("label = \"{}\";\n", _impl_get_label());
        ss << "}\n";

        return ss.str();
    }

    std::string OperationSequence::_impl_get_tex() {
        std::stringstream ss;
        ss << "Start : " << _impl_get_label() << "\n";
        for (auto &node : nodes) {
            ss << node->get_tex_partial() << "\n";
        }
        ss << "End : " << _impl_get_label() << "\n";
        return ss.str();
    }

} // namespace shamrock::solvergraph
