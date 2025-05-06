// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file graph2dot.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamdag/shamdag.hpp"
#include <shamdag/INode.hpp>
#include <shamdag/graph2dot.hpp>
#include <sstream>

std::string get_node_dot_graph(std::vector<std::shared_ptr<INode>> arg) {

    // this function will return the dot graph will every connected node (aka in eval chain) to the
    // one supplied in arguments
    auto endpoints = get_endpoints(arg);

    struct Node {
        std::string id;
        std::string label;
    };
    struct Edge {
        std::string child;
        std::string parent;
        std::string label;
    };

    int i = 0;
    std::unordered_map<INode *, Node> node_map;
    multi_evaluate(
        endpoints,
        [&](auto &current) {
            node_map[&current] = {std::to_string(i++), current.get_label()};
        },
        [](auto &n) {
            return true;
        });

    std::unordered_map<IDataEdge *, Edge> edge_map;

    auto register_edge = [&](IDataEdge &edge) {
        edge.on_links([&](auto &child, auto &parent) {
            edge_map[&edge] = {node_map[&child].id, node_map[&parent].id, edge.get_label()};
        });
    };

    for (auto &[ptr, n] : node_map) {
        ptr->on_edge_inputs(register_edge);
        ptr->on_edge_outputs(register_edge);
    }

    std::stringstream output;

    output << "digraph G {\n";

    for (auto &[ptr, n] : node_map) {
        output << "node_" << n.id << " [label=\"" << n.label << "\"];\n";
    }

    for (auto &[ptr, e] : edge_map) {
        output << "node_" << e.child << " -> node_" << e.parent << " [label=\"" << e.label
               << "\"];\n";
    }

    output << "}\n";

    return output.str();
}
