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

#include "shambase/memory.hpp"
#include "shamrock/solvergraph/IEdge.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include <unordered_map>
#include <memory>

namespace shamrock::solvergraph {

    class SolverGraph {
        std::unordered_map<std::string, std::shared_ptr<INode>> nodes;
        std::unordered_map<std::string, std::shared_ptr<IEdge>> edges;

        public:
        ///////////////////////////////////////
        // base getters and setters
        ///////////////////////////////////////

        inline void register_node_ptr_base(const std::string &name, std::shared_ptr<INode> node) {
            // check if node already exists
            if (nodes.find(name) != nodes.end()) {
                shambase::throw_with_loc<std::invalid_argument>(
                    shambase::format("Node already exists: {}", name));
            }
            nodes.insert({name, node});
        }

        inline void register_edge_ptr_base(const std::string &name, std::shared_ptr<IEdge> edge) {
            // check if edge already exists
            if (edges.find(name) != edges.end()) {
                shambase::throw_with_loc<std::invalid_argument>(
                    shambase::format("Edge already exists: {}", name));
            }
            edges.insert({name, edge});
        }

        inline const std::shared_ptr<INode> &get_node_ptr_base(const std::string &name) const {
            auto it = nodes.find(name);
            if (it == nodes.end()) {
                shambase::throw_with_loc<std::invalid_argument>(
                    shambase::format("Node does not exist: {}", name));
            }
            return it->second;
        }

        inline const std::shared_ptr<IEdge> &get_edge_ptr_base(const std::string &name) const {
            auto it = edges.find(name);
            if (it == edges.end()) {
                shambase::throw_with_loc<std::invalid_argument>(
                    shambase::format("Edge does not exist: {}", name));
            }
            return it->second;
        }

        ///////////////////////////////////////
        // generic getters
        ///////////////////////////////////////

        inline INode &get_node_ref(const std::string &name) const {
            return shambase::get_check_ref(get_node_ptr_base(name));
        }

        inline IEdge &get_edge_ref(const std::string &name) const {
            return shambase::get_check_ref(get_edge_ptr_base(name));
        }

        ///////////////////////////////////////
        // templated register and getters
        ///////////////////////////////////////

        template<class T>
        inline void register_node(const std::string &name, T &&node) {
            register_node_ptr_base(name, std::make_shared<T>(std::forward<T>(node)));
        }

        template<class T>
        inline void register_edge(const std::string &name, T &&edge) {
            register_edge_ptr_base(name, std::make_shared<T>(std::forward<T>(edge)));
        }

        template<class T>
        inline std::shared_ptr<T> get_node_ptr(const std::string &name) {
            return std::dynamic_pointer_cast<T>(get_node_ptr_base(name));
        }

        template<class T>
        inline std::shared_ptr<T> get_edge_ptr(const std::string &name) {
            return std::dynamic_pointer_cast<T>(get_edge_ptr_base(name));
        }

        template<class T>
        inline T &get_node_ref(const std::string &name) {
            return shambase::get_check_ref(get_node_ptr<T>(name));
        }

        template<class T>
        inline T &get_edge_ref(const std::string &name) {
            return shambase::get_check_ref(get_edge_ptr<T>(name));
        }
    };

} // namespace shamrock::solvergraph
