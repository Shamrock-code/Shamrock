// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file SolverGraph.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shamrock/solvergraph/IEdge.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include <unordered_map>
#include <memory>

namespace shamrock::solvergraph {

    class SolverGraph {
        public:
        std::unordered_map<std::string, std::shared_ptr<INode>> nodes;
        std::unordered_map<std::string, std::shared_ptr<IEdge>> edges;

        inline void register_node(std::string name, std::shared_ptr<INode> node) {
            // check if node already exists
            if (nodes.find(name) != nodes.end()) {
                shambase::throw_with_loc<std::invalid_argument>(
                    shambase::format("Node already exists: {}", name));
            }
            nodes.insert({name, node});
        }

        inline void register_edge(std::string name, std::shared_ptr<IEdge> edge) {
            // check if edge already exists
            if (edges.find(name) != edges.end()) {
                shambase::throw_with_loc<std::invalid_argument>(
                    shambase::format("Edge already exists: {}", name));
            }
            edges.insert({name, edge});
        }

        template<class T, class... Targs>
        inline void register_new_node(std::string name, Targs... args) {
            nodes.insert({name, std::make_shared<T>(args...)});
        }

        template<class T, class... Targs>
        inline void register_new_edge(std::string name, Targs... args) {
            edges.insert({name, std::make_shared<T>(args...)});
        }

        // generic getters

        inline INode &get_node_ref(std::string name) {
            if (nodes.find(name) == nodes.end()) {
                shambase::throw_with_loc<std::invalid_argument>(
                    shambase::format("Node does not exist: {}", name));
            }
            return shambase::get_check_ref(nodes.at(name));
        }

        inline IEdge &get_edge_ref(std::string name) {
            if (edges.find(name) == edges.end()) {
                shambase::throw_with_loc<std::invalid_argument>(
                    shambase::format("Edge does not exist: {}", name));
            }
            return shambase::get_check_ref(edges.at(name));
        }

        inline std::shared_ptr<INode> get_node_ptr(std::string name) {
            if (nodes.find(name) == nodes.end()) {
                shambase::throw_with_loc<std::invalid_argument>(
                    shambase::format("Node does not exist: {}", name));
            }
            return nodes.at(name);
        }

        inline std::shared_ptr<IEdge> get_edge_ptr(std::string name) {
            if (edges.find(name) == edges.end()) {
                shambase::throw_with_loc<std::invalid_argument>(
                    shambase::format("Edge does not exist: {}", name));
            }
            return edges.at(name);
        }

        // templated getters

        template<class T>
        inline T &get_node_ref(std::string name) {
            return shambase::get_check_ref(std::dynamic_pointer_cast<T>(nodes.at(name)));
        }

        template<class T>
        inline T &get_edge_ref(std::string name) {
            return shambase::get_check_ref(std::dynamic_pointer_cast<T>(edges.at(name)));
        }

        template<class T>
        inline std::shared_ptr<T> get_node_ptr(std::string name) {
            return std::dynamic_pointer_cast<T>(nodes.at(name));
        }

        template<class T>
        inline std::shared_ptr<T> get_edge_ptr(std::string name) {
            return std::dynamic_pointer_cast<T>(edges.at(name));
        }
    };

} // namespace shamrock::solvergraph
