// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file OperationIf.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamsolvergraph/node/OperationIf.hpp"
#include <sstream>

namespace shamrock::solvergraph {

    void OperationIf::_impl_evaluate_internal() {
        if (get_edges().condition.data) {
            if (then_node) {
                then_node->evaluate();
            }
        } else if (else_node) {
            else_node->evaluate();
        }
    }

    std::string OperationIf::_impl_get_dot_graph_partial() const {

        std::stringstream ss;

        ss << "subgraph cluster_" + std::to_string(get_uuid()) + " {\n";
        ss << shambase::format(
            "n_{} [label=\"{}\", shape=diamond];\n", get_uuid(), _impl_get_label());

        if (then_node) {
            ss << then_node->get_dot_graph_partial();
            ss << shambase::format(
                "n_{} -> {} [label=\"true\"];\n",
                get_uuid(),
                then_node->get_dot_graph_node_start());
            ss << then_node->get_dot_graph_node_end() << " -> "
               << shambase::format("n_{}_end", get_uuid()) << ";\n";
        } else {
            ss << shambase::format(
                "n_{} -> n_{}_end [label=\"true\", style=dashed];\n", get_uuid(), get_uuid());
        }

        if (else_node) {
            ss << else_node->get_dot_graph_partial();
            ss << shambase::format(
                "n_{} -> {} [label=\"false\"];\n",
                get_uuid(),
                else_node->get_dot_graph_node_start());
            ss << else_node->get_dot_graph_node_end() << " -> "
               << shambase::format("n_{}_end", get_uuid()) << ";\n";
        } else {
            ss << shambase::format(
                "n_{} -> n_{}_end [label=\"false\", style=dashed];\n", get_uuid(), get_uuid());
        }

        ss << shambase::format("n_{}_end [label=\"\", shape=point, width=0.15];\n", get_uuid());
        ss << shambase::format("label = \"{}\";\n", _impl_get_label());
        ss << "}\n";

        auto &cond = get_ro_edge_base(0);
        ss << shambase::format(
            "e_{} -> n_{} [style=\"dashed\", color=green];\n", cond.get_uuid(), get_uuid());
        ss << shambase::format(
            "e_{} [label=\"{}\",shape=rect, style=filled];\n", cond.get_uuid(), cond.get_label());

        return ss.str();
    }

    std::string OperationIf::_impl_get_tex() const {
        std::stringstream ss;
        ss << "Start : " << _impl_get_label() << "\n";
        if (then_node) {
            ss << "Then :\n" << then_node->get_tex_partial() << "\n";
        }
        if (else_node) {
            ss << "Else :\n" << else_node->get_tex_partial() << "\n";
        }
        ss << "End : " << _impl_get_label() << "\n";
        return ss.str();
    }

} // namespace shamrock::solvergraph
