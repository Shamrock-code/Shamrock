// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file graph2tex.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include <shamdag/INode.hpp>
#include <shamdag/graph2tex.hpp>
#include <sstream>

std::string get_node_graph_tex(std::vector<std::shared_ptr<INode>> arg) {

    // this function will return the tex will every connected node (aka in eval chain) to the one
    // supplied in arguments
    auto endpoints = get_endpoints(arg);

    std::stringstream output;

    output << "\\documentclass{article}\n";
    output << "\\begin{document}\n";
    multi_evaluate(
        endpoints,
        [&](auto &current) {
            output << current.get_node_tex() << "\n";
        },
        [](auto &n) {
            return true;
        });
    output << "\\end{document}\n";

    return output.str();
}
