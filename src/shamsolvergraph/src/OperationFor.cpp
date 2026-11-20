// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file OperationFor.cpp
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 *
 */

#include "shamsolvergraph/node/OperationFor.hpp"
#include <sstream>

namespace shamrock::solvergraph {

    void OperationFor::_impl_evaluate_internal() {
        const u32 count = get_edges().count.data;
        for (u32 i = 0; i < count; i++) {
            body->evaluate();
        }
    }

    std::string OperationFor::_impl_get_dot_graph_partial() const {

        std::stringstream ss;

        ss << "subgraph cluster_" + std::to_string(get_uuid()) + " {\n";
        ss << sham::format("n_{} [label=\"{}\", shape=invhouse];\n", get_uuid(), _impl_get_label());

        ss << body->get_dot_graph_partial();
        ss << sham::format("n_{} -> {};\n", get_uuid(), body->get_dot_graph_node_start());
        ss << body->get_dot_graph_node_end() << " -> " << sham::format("n_{}_end", get_uuid())
           << ";\n";
        // loop back edge
        ss << body->get_dot_graph_node_end() << " -> " << body->get_dot_graph_node_start()
           << " [label=\"repeat\", style=dashed, constraint=false];\n";

        ss << sham::format("n_{}_end [label=\"\", shape=point, width=0.15];\n", get_uuid());
        ss << sham::format("label = \"{}\";\n", _impl_get_label());
        ss << "}\n";

        auto &count = get_ro_edge_base(0);
        ss << sham::format(
            "e_{} -> n_{} [style=\"dashed\", color=green];\n", count.get_uuid(), get_uuid());
        ss << sham::format(
            "e_{} [label=\"{}\",shape=rect, style=filled];\n", count.get_uuid(), count.get_label());

        return ss.str();
    }

    std::string OperationFor::_impl_get_tex() const {
        std::stringstream ss;
        ss << "Start : " << _impl_get_label() << " (repeated $"
           << get_ro_edge_base(0).get_tex_symbol() << "$ times)\n";
        ss << body->get_tex_partial() << "\n";
        ss << "End : " << _impl_get_label() << "\n";
        return ss.str();
    }

} // namespace shamrock::solvergraph
