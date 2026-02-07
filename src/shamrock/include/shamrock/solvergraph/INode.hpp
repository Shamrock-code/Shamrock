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
 * @file INode.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/WithUUID.hpp"
#include "shambase/memory.hpp"
#include "shamrock/solvergraph/IEdge.hpp"
#include "shamrock/solvergraph/INullOptEdge.hpp"
#include <memory>
#include <vector>

namespace shamrock::solvergraph {

    /// Inode is node between data edges, takes multiple inputs, multiple outputs
    class INode : public std::enable_shared_from_this<INode>,
                  public shambase::WithUUID<INode, u64> {

        /// Read only edges
        std::vector<std::shared_ptr<IEdge>> ro_edges;
        /// Read write edges
        std::vector<std::shared_ptr<IEdge>> rw_edges;

        public:
        INode() = default;

        INode(const INode &)            = delete; /// would violate shared_from_this() & unique UUID
        INode &operator=(const INode &) = delete; /// would violate shared_from_this() & unique UUID

        /// Move constructor - automatically delegates to base classes and members
        INode(INode &&) noexcept = default;

        /// Move assignment - automatically delegates to base classes and members
        INode &operator=(INode &&) noexcept = default;

        /// Get a shared pointer to this node
        inline std::shared_ptr<INode> getptr_shared() { return shared_from_this(); }
        /// Get a weak pointer to this node
        inline std::weak_ptr<INode> getptr_weak() { return weak_from_this(); }

        /// Get the read only edges
        inline std::vector<std::shared_ptr<IEdge>> &get_ro_edges() { return ro_edges; }
        /// Get the read write edges
        inline std::vector<std::shared_ptr<IEdge>> &get_rw_edges() { return rw_edges; }

        /// Set the read only edges
        inline void __internal_set_ro_edges(std::vector<std::shared_ptr<IEdge>> new_ro_edges);
        /// Set the read write edges
        inline void __internal_set_rw_edges(std::vector<std::shared_ptr<IEdge>> new_rw_edges);

        /// Apply a function to the read only edges
        template<class Func>
        void on_edge_ro_edges(Func &&f);

        /// Apply a function to the read write edges
        template<class Func>
        void on_edge_rw_edges(Func &&f);

        /// Destructor (virtual) & reset the edges
        virtual ~INode() {
            __internal_set_ro_edges({});
            __internal_set_rw_edges({});
        }

        /// Get a read only edge and cast it to the type T
        template<class T>
        inline const T &get_ro_edge(int slot) {
            return shambase::get_check_ref(std::dynamic_pointer_cast<T>(ro_edges.at(slot)));
        }

        /// Get a read write edge and cast it to the type T
        template<class T>
        inline T &get_rw_edge(int slot) {
            return shambase::get_check_ref(std::dynamic_pointer_cast<T>(rw_edges.at(slot)));
        }

        /// Get a read only edge and cast it to the type T, return an optional
        template<class T>
        inline std::optional<const T &> get_ro_edge_optional(int slot) {
            auto &tmp = ro_edges.at(slot);
            if (T *ptr = std::dynamic_pointer_cast<T>(tmp)) {
                return *ptr;
            } else if (is_null_opt_edge(tmp)) {
                return std::nullopt;
            } else {
                shambase::throw_with_loc<std::invalid_argument>(
                    shambase::format("Edge is not from the requested type: {}", slot));
            }
        }

        /// Get a read write edge and cast it to the type T, return an optional
        template<class T>
        inline std::optional<T &> get_rw_edge_optional(int slot) {
            auto &tmp = rw_edges.at(slot);
            if (T *ptr = std::dynamic_pointer_cast<T>(tmp)) {
                return *ptr;
            } else if (is_null_opt_edge(tmp)) {
                return std::nullopt;
            } else {
                shambase::throw_with_loc<std::invalid_argument>(
                    shambase::format("Edge is not from the requested type: {}", slot));
            }
        }

        /// Get a reference to a read only edge
        inline const IEdge &get_ro_edge_base(int slot) {
            return shambase::get_check_ref(ro_edges.at(slot));
        }

        inline const IEdge &get_ro_edge_base(int slot) const {
            return shambase::get_check_ref(ro_edges.at(slot));
        }

        /// Get a reference to a read write edge and cast it to the type IEdge
        inline IEdge &get_rw_edge_base(int slot) {
            return shambase::get_check_ref(rw_edges.at(slot));
        }

        inline const IEdge &get_rw_edge_base(int slot) const {
            return shambase::get_check_ref(rw_edges.at(slot));
        }

        /// Evaluate the node
        inline void evaluate() { _impl_evaluate_internal(); }

        /// Get the dot graph of the node (Currently only an alias to get_dot_graph_partial)
        inline std::string get_dot_graph() { return get_dot_graph_partial(); };

        /// Get the dot graph of the subgraph corresponding to the node
        inline std::string get_dot_graph_partial() { return _impl_get_dot_graph_partial(); };

        /// Get the id of the node start in the dot graph
        inline std::string get_dot_graph_node_start() { return _impl_get_dot_graph_node_start(); };
        /// Get the id of the node end in the dot graph
        inline std::string get_dot_graph_node_end() { return _impl_get_dot_graph_node_end(); };

        /// Get the TeX of the node
        inline std::string get_tex() { return _impl_get_tex(); };
        /// Get the TeX of the node partial
        inline std::string get_tex_partial() { return _impl_get_tex(); };

        /// print the node info
        inline virtual std::string print_node_info() const {
            std::string node_info = shambase::format("Node info :\n");
            node_info += shambase::format(" - Node type : {}\n", typeid(*this).name());
            node_info += shambase::format(" - Node UUID : {}\n", get_uuid());
            node_info += shambase::format(" - Node label : {}\n", _impl_get_label());

            auto append_edges_info = [&](const char *title, const auto &edges) {
                node_info += shambase::format(" - {}: {}\n", title, edges.size());
                for (const auto &edge : edges) {
                    const auto &e = *edge; // necessary to avoid -Wpotentially-evaluated-expression
                    node_info += shambase::format(
                        "     - Edge ptr = {}, uuid = {}, label = {},\n          type = {} \n",
                        static_cast<void *>(edge.get()),
                        edge->get_uuid(),
                        edge->get_label(),
                        typeid(e).name());
                }
            };

            append_edges_info("Node Read Only edges", ro_edges);
            append_edges_info("Node Read Write edges", rw_edges);

            return node_info;
        };

        protected:
        /// evaluate the node
        virtual void _impl_evaluate_internal() = 0;

        /// get the label of the node
        virtual std::string _impl_get_label() const = 0;

        /// get the dot graph of the node partial
        virtual std::string _impl_get_dot_graph_partial() const;
        /// get the dot graph of the node start
        virtual std::string _impl_get_dot_graph_node_start() const;
        /// get the dot graph of the node end
        virtual std::string _impl_get_dot_graph_node_end() const;

        /// get the tex of the node
        virtual std::string _impl_get_tex() const = 0;
    };

    inline void INode::__internal_set_ro_edges(std::vector<std::shared_ptr<IEdge>> new_ro_edges) {
        for (auto e : ro_edges) {
            // shambase::get_check_ref(e).parent = {};
        }
        this->ro_edges = new_ro_edges;
        for (auto e : ro_edges) {
            // shambase::get_check_ref(e).parent = getptr_weak();
        }
    }

    inline void INode::__internal_set_rw_edges(std::vector<std::shared_ptr<IEdge>> new_rw_edges) {
        for (auto e : rw_edges) {
            // shambase::get_check_ref(e).child = {};
        }
        this->rw_edges = new_rw_edges;
        for (auto e : rw_edges) {
            // shambase::get_check_ref(e).child = getptr_weak();
        }
    }

    template<class Func>
    inline void INode::on_edge_ro_edges(Func &&f) {
        for (auto &in : ro_edges) {
            f(shambase::get_check_ref(in));
        }
    }

    template<class Func>
    inline void INode::on_edge_rw_edges(Func &&f) {
        for (auto &out : rw_edges) {
            f(shambase::get_check_ref(out));
        }
    }

    inline std::string INode::_impl_get_dot_graph_partial() const {
        std::string node_str
            = shambase::format("n_{} [label=\"{}\"];\n", this->get_uuid(), _impl_get_label());

        std::string edge_str = "";
        for (auto &in : ro_edges) {
            edge_str += shambase::format(
                "e_{} -> n_{} [style=\"dashed\", color=green];\n",
                in->get_uuid(),
                this->get_uuid());
            edge_str += shambase::format(
                "e_{} [label=\"{}\",shape=rect, style=filled];\n", in->get_uuid(), in->get_label());
        }
        for (auto &out : rw_edges) {
            edge_str += shambase::format(
                "n_{} -> e_{} [style=\"dashed\", color=red];\n", this->get_uuid(), out->get_uuid());
            edge_str += shambase::format(
                "e_{} [label=\"{}\",shape=rect, style=filled];\n",
                out->get_uuid(),
                out->get_label());
        }

        return shambase::format("{}{}", node_str, edge_str);
    };

    inline std::string INode::_impl_get_dot_graph_node_start() const {
        return shambase::format("n_{}", this->get_uuid());
    }
    inline std::string INode::_impl_get_dot_graph_node_end() const {
        return shambase::format("n_{}", this->get_uuid());
    }

} // namespace shamrock::solvergraph
