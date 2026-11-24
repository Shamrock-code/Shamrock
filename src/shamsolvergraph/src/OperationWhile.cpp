// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file OperationWhile.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamsolvergraph/node/OperationWhile.hpp"
#include <sstream>

namespace shamrock::solvergraph {

    void OperationWhile::_impl_evaluate_internal() {
        // The condition is re-read at every iteration since the body is expected to update it.
        while (get_edges().condition.data) {
            body_node->evaluate();
        }
    }

    std::string OperationWhile::_impl_get_dot_graph_partial() const {

        std::stringstream ss;

        ss << "subgraph cluster_" + std::to_string(get_uuid()) + " {\n";
        ss << sham::format("n_{} [label=\"{}\", shape=diamond];\n", get_uuid(), _impl_get_label());

        ss << body_node->get_dot_graph_partial();
        ss << sham::format(
            "n_{} -> {} [label=\"true\"];\n", get_uuid(), body_node->get_dot_graph_node_start());
        ss << body_node->get_dot_graph_node_end() << " -> "
           << sham::format("n_{} [label=\"loop\", constraint=false];\n", get_uuid());

        ss << sham::format("n_{} -> n_{}_end [label=\"false\"];\n", get_uuid(), get_uuid());

        ss << sham::format("n_{}_end [label=\"\", shape=point, width=0.15];\n", get_uuid());
        ss << sham::format("label = \"{}\";\n", _impl_get_label());
        ss << "}\n";

        auto &cond = get_ro_edge_base(0);
        ss << sham::format(
            "e_{} -> n_{} [style=\"dashed\", color=green];\n", cond.get_uuid(), get_uuid());
        ss << sham::format(
            "e_{} [label=\"{}\",shape=rect, style=filled];\n", cond.get_uuid(), cond.get_label());

        return ss.str();
    }

    std::string OperationWhile::_impl_get_tex() const {
        std::stringstream ss;
        ss << "Start : " << _impl_get_label() << "\n";
        ss << "While :\n" << body_node->get_tex_partial() << "\n";
        ss << "End : " << _impl_get_label() << "\n";
        return ss.str();
    }

} // namespace shamrock::solvergraph
