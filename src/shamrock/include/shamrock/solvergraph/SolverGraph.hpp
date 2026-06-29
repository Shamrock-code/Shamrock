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
 * @file SolverGraph.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Declare a class to register and retrieve nodes and edges from a unique container.
 *
 */

#include "shambase/exception.hpp"
#include "shambase/memory.hpp"
#include "shamrock/solvergraph/IEdge.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include <unordered_map>
#include <algorithm>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace shamrock::solvergraph {

    using SolverGraphNodeCheck = std::function<bool(const std::shared_ptr<INode> &node)>;
    using SolverGraphEdgeCheck = std::function<bool(const std::shared_ptr<IEdge> &edge)>;

    struct SolverGraphConstraint {
        std::string name;
        SolverGraphNodeCheck node_check;
        SolverGraphEdgeCheck edge_check;

        inline static SolverGraphConstraint no_constraint() {
            return {.name = {}, .node_check = nullptr, .edge_check = nullptr};
        }

        inline bool check_node(const std::shared_ptr<INode> &node) const {
            if (!node_check) {
                return true;
            }
            return (node_check) (node);
        }

        inline bool check_edge(const std::shared_ptr<IEdge> &edge) const {
            if (!edge_check) {
                return true;
            }
            return (edge_check) (edge);
        }

        inline bool is_active() const { return bool(node_check) || bool(edge_check); }
    };

    /**
     * @brief A graph container for managing solver nodes and edges with type-safe access.
     *
     * The SolverGraph class provides a centralized registry for solver graph components,
     * allowing nodes and edges to be registered by name and retrieved with type safety.
     * It supports both polymorphic access through base interfaces and templated access
     * for specific derived types.
     *
     * @code{.cpp}
     * // Create a solver graph
     * SolverGraph graph;
     *
     * // Register a node with automatic type deduction
     * MyNodeType node_instance;
     * graph.register_node("my_node", std::move(node_instance));
     *
     * // Register an edge
     * MyEdgeType edge_instance;
     * graph.register_edge("my_edge", std::move(edge_instance));
     *
     * // Retrieve typed references
     * auto& node_ref = graph.get_node_ref<MyNodeType>("my_node");
     * auto& edge_ref = graph.get_edge_ref<MyEdgeType>("my_edge");
     *
     * // Or get shared pointers for polymorphic access
     * auto node_ptr = graph.get_node_ptr<MyNodeType>("my_node");
     * auto edge_ptr = graph.get_edge_ptr<MyEdgeType>("my_edge");
     *
     * // Or create a graph with registration constraints under a single name
     * auto constrained = SolverGraph::with_constraint(SolverGraphConstraint{
     *     .name = "sph_solver_graph",
     *     .node_check = [](const std::shared_ptr<INode> &n) { return true; },
     *     .edge_check = [](const std::shared_ptr<IEdge> &e) { return true; },
     * });
     * @endcode
     */
    class SolverGraph {
        /// Registry of nodes by name
        std::unordered_map<std::string, std::shared_ptr<INode>> nodes = {};

        /// Registry of edges by name
        std::unordered_map<std::string, std::shared_ptr<IEdge>> edges = {};

        SolverGraphConstraint constraint = SolverGraphConstraint::no_constraint();

        explicit SolverGraph(SolverGraphConstraint graph_constraint)
            : constraint(std::move(graph_constraint)) {}

        public:
        ///////////////////////////////////////
        // base getters and setters
        ///////////////////////////////////////

        SolverGraph() = default;

        /**
         * @brief Create a solver graph with registration constraints.
         *
         * @param graph_constraint Named constraint with optional node and edge checks
         */
        inline static SolverGraph with_constraint(SolverGraphConstraint graph_constraint) {
            return SolverGraph{std::move(graph_constraint)};
        }

        /**
         * @brief Register a node with the graph using a shared pointer.
         *
         * @param name Unique identifier for the node
         * @param node Shared pointer to the node instance
         * @throws std::invalid_argument if constraint validation fails or a node with the same name
         * already exists
         */
        inline std::shared_ptr<INode> register_node_ptr_base(
            const std::string &name, std::shared_ptr<INode> node) {

            if (!constraint.check_node(node)) {
                shambase::throw_with_loc<std::invalid_argument>(shambase::format(
                    "Solvergraph constraint '{}' rejected node '{}' (label='{}', uuid={})",
                    constraint.name,
                    name,
                    node->get_label(),
                    node->get_uuid()));
            }

            const auto [it, inserted] = nodes.try_emplace(name, std::move(node));
            if (!inserted) {
                shambase::throw_with_loc<std::invalid_argument>(
                    shambase::format("Node already exists: {}", name));
            }
            return it->second;
        }

        /**
         * @brief Register an edge with the graph using a shared pointer.
         *
         * @param name Unique identifier for the edge
         * @param edge Shared pointer to the edge instance
         * @throws std::invalid_argument if constraint validation fails or an edge with the same
         * name already exists
         */
        inline std::shared_ptr<IEdge> register_edge_ptr_base(
            const std::string &name, std::shared_ptr<IEdge> edge) {

            if (!constraint.check_edge(edge)) {
                shambase::throw_with_loc<std::invalid_argument>(shambase::format(
                    "Solvergraph constraint '{}' rejected edge '{}' (label='{}', uuid={})",
                    constraint.name,
                    name,
                    edge->get_label(),
                    edge->get_uuid()));
            }

            const auto [it, inserted] = edges.try_emplace(name, std::move(edge));
            if (!inserted) {
                shambase::throw_with_loc<std::invalid_argument>(
                    shambase::format("Edge already exists: {}", name));
            }
            return it->second;
        }

        /**
         * @brief Retrieve a node by name as a shared pointer to the base interface.
         *
         * @param name Unique identifier for the node
         * @return Reference to shared pointer to the node
         * @throws std::invalid_argument if no node with the given name exists
         */
        inline std::shared_ptr<INode> &get_node_ptr_base(const std::string &name) {
            auto it = nodes.find(name);
            if (it == nodes.end()) {
                shambase::throw_with_loc<std::invalid_argument>(
                    shambase::format("Node does not exist: {}", name));
            }
            return it->second;
        }

        /// const variant
        inline const std::shared_ptr<INode> &get_node_ptr_base(const std::string &name) const {
            auto it = nodes.find(name);
            if (it == nodes.end()) {
                shambase::throw_with_loc<std::invalid_argument>(
                    shambase::format("Node does not exist: {}", name));
            }
            return it->second;
        }

        /**
         * @brief Retrieve an edge by name as a shared pointer to the base interface.
         *
         * @param name Unique identifier for the edge
         * @return Reference to shared pointer to the edge
         * @throws std::invalid_argument if no edge with the given name exists
         */
        inline std::shared_ptr<IEdge> &get_edge_ptr_base(const std::string &name) {
            auto it = edges.find(name);
            if (it == edges.end()) {
                shambase::throw_with_loc<std::invalid_argument>(
                    shambase::format("Edge does not exist: {}", name));
            }
            return it->second;
        }

        /// const variant
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

        /**
         * @brief Get a reference to a node by name through the base interface.
         *
         * @param name Unique identifier for the node
         * @return Reference to the node through the base interface
         * @throws std::invalid_argument if no node with the given name exists
         */
        inline INode &get_node_ref_base(const std::string &name) {
            return shambase::get_check_ref(get_node_ptr_base(name));
        }

        /// const variant
        inline const INode &get_node_ref_base(const std::string &name) const {
            return shambase::get_check_ref(get_node_ptr_base(name));
        }

        /**
         * @brief Get a reference to an edge by name through the base interface.
         *
         * @param name Unique identifier for the edge
         * @return Reference to the edge through the base interface
         * @throws std::invalid_argument if no edge with the given name exists
         */
        inline IEdge &get_edge_ref_base(const std::string &name) {
            return shambase::get_check_ref(get_edge_ptr_base(name));
        }

        /// const variant
        inline const IEdge &get_edge_ref_base(const std::string &name) const {
            return shambase::get_check_ref(get_edge_ptr_base(name));
        }

        ///////////////////////////////////////
        // templated register and getters
        ///////////////////////////////////////

        /**
         * @brief Register a node with automatic type deduction and shared pointer creation.
         *
         * This method creates a shared pointer from the provided node instance and
         * registers it with the graph. The node type is automatically deduced from
         * the template parameter.
         *
         * @tparam T Type of the node (must derive from INode)
         * @param name Unique identifier for the node
         * @param node Node instance to register (will be moved)
         * @throws std::invalid_argument if constraint validation fails or a node with the same name
         * already exists
         */
        template<class T>
        inline std::shared_ptr<T> register_node(const std::string &name, T &&node) {
            static_assert(std::is_base_of<INode, T>::value, "T must derive from INode");
            register_node_ptr_base(name, std::make_shared<T>(std::forward<T>(node)));
            return get_node_ptr<T>(name);
        }

        /**
         * @brief Register an edge with automatic type deduction and shared pointer creation.
         *
         * This method creates a shared pointer from the provided edge instance and
         * registers it with the graph. The edge type is automatically deduced from
         * the template parameter.
         *
         * @tparam T Type of the edge (must derive from IEdge)
         * @param name Unique identifier for the edge
         * @param edge Edge instance to register (will be moved)
         * @throws std::invalid_argument if constraint validation fails or an edge with the same
         * name already exists
         */
        template<class T>
        inline std::shared_ptr<T> register_edge(const std::string &name, T &&edge) {
            static_assert(std::is_base_of<IEdge, T>::value, "T must derive from IEdge");
            register_edge_ptr_base(name, std::make_shared<T>(std::forward<T>(edge)));
            return get_edge_ptr<T>(name);
        }

        /**
         * @brief Get a typed shared pointer to a node by name.
         *
         * This method performs a dynamic cast to the requested type and returns
         * a shared pointer. Throw an exception if the cast fails.
         *
         * @tparam T Type of the node to retrieve
         * @param name Unique identifier for the node
         * @return Shared pointer to the typed node, or throw an exception if cast fails
         * @throws std::invalid_argument if no node with the given name exists or cast fails
         */
        template<class T>
        inline std::shared_ptr<T> get_node_ptr(const std::string &name) {
            auto tmp = std::dynamic_pointer_cast<T>(get_node_ptr_base(name));
            if (!bool(tmp)) {
                shambase::throw_with_loc<std::invalid_argument>(
                    shambase::format("Node exists but is not from the requested type: {}", name));
            }
            return tmp;
        }

        /// const variant
        template<class T>
        inline std::shared_ptr<T> get_node_ptr(const std::string &name) const {
            auto tmp = std::dynamic_pointer_cast<T>(get_node_ptr_base(name));
            if (!bool(tmp)) {
                shambase::throw_with_loc<std::invalid_argument>(
                    shambase::format("Node exists but is not from the requested type: {}", name));
            }
            return tmp;
        }

        /**
         * @brief Get a typed shared pointer to an edge by name.
         *
         * This method performs a dynamic cast to the requested type and returns
         * a shared pointer. Throw an exception if the cast fails.
         *
         * @tparam T Type of the edge to retrieve
         * @param name Unique identifier for the edge
         * @return Shared pointer to the typed edge, or throw an exception if cast fails
         * @throws std::invalid_argument if no edge with the given name exists or cast fails
         */
        template<class T>
        inline std::shared_ptr<T> get_edge_ptr(const std::string &name) {
            auto tmp = std::dynamic_pointer_cast<T>(get_edge_ptr_base(name));
            if (!bool(tmp)) {
                shambase::throw_with_loc<std::invalid_argument>(
                    shambase::format("Edge exists but is not from the requested type: {}", name));
            }
            return tmp;
        }

        /// const variant
        template<class T>
        inline std::shared_ptr<T> get_edge_ptr(const std::string &name) const {
            auto tmp = std::dynamic_pointer_cast<T>(get_edge_ptr_base(name));
            if (!bool(tmp)) {
                shambase::throw_with_loc<std::invalid_argument>(
                    shambase::format("Edge exists but is not from the requested type: {}", name));
            }
            return tmp;
        }

        /**
         * @brief Get a typed reference to a node by name.
         *
         * This method performs a dynamic cast and returns a reference to the
         * typed node. Throws an exception if the cast fails.
         *
         * @tparam T Type of the node to retrieve
         * @param name Unique identifier for the node
         * @return Reference to the typed node
         * @throws std::invalid_argument if no node with the given name exists or cast fails
         */
        template<class T>
        inline T &get_node_ref(const std::string &name) {
            return shambase::get_check_ref(get_node_ptr<T>(name));
        }

        /// const variant
        template<class T>
        inline const T &get_node_ref(const std::string &name) const {
            return shambase::get_check_ref(get_node_ptr<T>(name));
        }

        /**
         * @brief Get a typed reference to an edge by name.
         *
         * This method performs a dynamic cast and returns a reference to the
         * typed edge. Throws an exception if the cast fails.
         *
         * @tparam T Type of the edge to retrieve
         * @param name Unique identifier for the edge
         * @return Reference to the typed edge
         * @throws std::invalid_argument if no edge with the given name exists or cast fails
         */
        template<class T>
        inline T &get_edge_ref(const std::string &name) {
            return shambase::get_check_ref(get_edge_ptr<T>(name));
        }

        /// const variant
        template<class T>
        inline const T &get_edge_ref(const std::string &name) const {
            return shambase::get_check_ref(get_edge_ptr<T>(name));
        }

        /// Returns edge registration keys in lexicographic order (deterministic).
        inline std::vector<std::string> get_edge_names() const {
            std::vector<std::string> ret{};
            ret.reserve(edges.size());

            for (const auto &entry : edges) {
                ret.push_back(entry.first);
            }
            std::sort(ret.begin(), ret.end());
            return ret;
        }

        /// Returns node registration keys in lexicographic order (deterministic).
        inline std::vector<std::string> get_node_names() const {
            std::vector<std::string> ret{};
            ret.reserve(nodes.size());

            for (const auto &entry : nodes) {
                ret.push_back(entry.first);
            }
            std::sort(ret.begin(), ret.end());
            return ret;
        }
    };

} // namespace shamrock::solvergraph
