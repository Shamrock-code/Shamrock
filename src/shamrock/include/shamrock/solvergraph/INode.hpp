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
 * @file INode.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include "shambase/WithUUID.hpp"
#include "shambase/memory.hpp"
#include "shamrock/solvergraph/IEdge.hpp"
#include <memory>
#include <vector>

namespace shamrock::solvergraph {

    // Inode is node between data edges, takes multiple inputs, multiple outputs
    class INode : public std::enable_shared_from_this<INode>,
                  public shambase::WithUUID<INode, u64> {

        std::vector<std::shared_ptr<IEdge>> ro_edges;
        std::vector<std::shared_ptr<IEdge>> rw_edges;

        public:
        inline std::shared_ptr<INode> getptr_shared() { return shared_from_this(); }
        inline std::weak_ptr<INode> getptr_weak() { return weak_from_this(); }

        inline std::vector<std::shared_ptr<IEdge>> &get_ro_edges() { return ro_edges; }
        inline std::vector<std::shared_ptr<IEdge>> &get_rw_edges() { return rw_edges; }

        inline void __internal_set_ro_edges(std::vector<std::shared_ptr<IEdge>> new_ro_edges);
        inline void __internal_set_rw_edges(std::vector<std::shared_ptr<IEdge>> new_rw_edges);

        template<class Func>
        void on_edge_ro_edges(Func &&f);

        template<class Func>
        void on_edge_rw_edges(Func &&f);

        virtual ~INode() {
            __internal_set_ro_edges({});
            __internal_set_rw_edges({});
        }

        template<class T>
        inline const T &get_ro_edge(int slot) {
            return shambase::get_check_ref(std::dynamic_pointer_cast<T>(ro_edges.at(slot)));
        }

        template<class T>
        inline T &get_rw_edge(int slot) {
            return shambase::get_check_ref(std::dynamic_pointer_cast<T>(rw_edges.at(slot)));
        }

        inline const IEdge &get_ro_edge_base(int slot) {
            return shambase::get_check_ref(ro_edges.at(slot));
        }

        inline IEdge &get_rw_edge_base(int slot) {
            return shambase::get_check_ref(rw_edges.at(slot));
        }

        inline void evaluate() { _impl_evaluate_internal(); }

        inline std::string get_dot_graph() { return get_dot_graph_partial(); };
        inline std::string get_dot_graph_partial() { return _impl_get_dot_graph_partial(); };
        inline std::string get_dot_graph_node_start() { return _impl_get_dot_graph_node_start(); };
        inline std::string get_dot_graph_node_end() { return _impl_get_dot_graph_node_end(); };

        inline std::string get_tex() { return _impl_get_tex(); };
        inline std::string get_tex_partial() { return _impl_get_tex(); };

        protected:
        virtual void _impl_evaluate_internal() = 0;

        virtual std::string _impl_get_label() = 0;

        virtual std::string _impl_get_dot_graph_partial();
        virtual std::string _impl_get_dot_graph_node_start();
        virtual std::string _impl_get_dot_graph_node_end();

        virtual std::string _impl_get_tex() = 0;
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

    inline std::string INode::_impl_get_dot_graph_partial() {
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

    inline std::string INode::_impl_get_dot_graph_node_start() {
        return shambase::format("n_{}", this->get_uuid());
    }
    inline std::string INode::_impl_get_dot_graph_node_end() {
        return shambase::format("n_{}", this->get_uuid());
    }

} // namespace shamrock::solvergraph
