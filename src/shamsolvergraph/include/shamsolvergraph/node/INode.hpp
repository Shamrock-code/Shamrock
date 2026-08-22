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
#include "shambase/stacktrace.hpp"
#include "shamsolvergraph/edge/IEdge.hpp"
#include "shamsolvergraph/edge/INullOptEdge.hpp"
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
        inline std::optional<std::reference_wrapper<const T>> get_ro_edge_optional(int slot) {
            auto &edge = ro_edges.at(slot);

            auto ptr = std::dynamic_pointer_cast<T>(edge);
            if (ptr) {
                return std::cref(*ptr);
            }

            if (is_null_opt_edge(edge)) {
                return std::nullopt;
            }

            throw shambase::make_except_with_loc<std::invalid_argument>(
                sham::format("Edge is not from the requested type: {}", slot));
        }

        /// Get a read write edge and cast it to the type T, return an optional
        template<class T>
        inline std::optional<std::reference_wrapper<T>> get_rw_edge_optional(int slot) {
            auto &edge = rw_edges.at(slot);

            auto ptr = std::dynamic_pointer_cast<T>(edge);
            if (ptr) {
                return std::cref(*ptr);
            }

            if (is_null_opt_edge(edge)) {
                return std::nullopt;
            }

            throw shambase::make_except_with_loc<std::invalid_argument>(
                sham::format("Edge is not from the requested type: {}", slot));
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
        /// Get the label of the node
        inline std::string get_label() const { return _impl_get_label(); };

        /// print the node info
        inline virtual std::string print_node_info() const {
            std::string node_info = sham::format("Node info :\n");
            node_info += sham::format(" - Node type : {}\n", typeid(*this).name());
            node_info += sham::format(" - Node UUID : {}\n", get_uuid());
            node_info += sham::format(" - Node label : {}\n", _impl_get_label());

            auto append_edges_info = [&](const char *title, const auto &edges) {
                node_info += sham::format(" - {}: {}\n", title, edges.size());
                for (const auto &edge : edges) {
                    const auto &e = *edge; // necessary to avoid -Wpotentially-evaluated-expression
                    node_info += sham::format(
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
            = sham::format("n_{} [label=\"{}\"];\n", this->get_uuid(), _impl_get_label());

        std::string edge_str = "";
        for (auto &in : ro_edges) {
            edge_str += sham::format(
                "e_{} -> n_{} [style=\"dashed\", color=green];\n",
                in->get_uuid(),
                this->get_uuid());
            edge_str += sham::format(
                "e_{} [label=\"{}\",shape=rect, style=filled];\n", in->get_uuid(), in->get_label());
        }
        for (auto &out : rw_edges) {
            edge_str += sham::format(
                "n_{} -> e_{} [style=\"dashed\", color=red];\n", this->get_uuid(), out->get_uuid());
            edge_str += sham::format(
                "e_{} [label=\"{}\",shape=rect, style=filled];\n",
                out->get_uuid(),
                out->get_label());
        }

        return sham::format("{}{}", node_str, edge_str);
    };

    inline std::string INode::_impl_get_dot_graph_node_start() const {
        return sham::format("n_{}", this->get_uuid());
    }
    inline std::string INode::_impl_get_dot_graph_node_end() const {
        return sham::format("n_{}", this->get_uuid());
    }

    /// Cast a single generic edge to the concrete type expected, throwing a message naming the
    /// edge and the expected/actual type on any mismatch. Used to implement the untyped
    /// per-slot INode::set_edges() overload's cast step, which the generic vector-based overload
    /// forwards to.
    template<class T>
    inline std::shared_ptr<T> __node_edge_cast_checked(
        const std::shared_ptr<IEdge> &edge, const char *edge_name, const char *type_name) {
        if (!edge) {
            throw shambase::make_except_with_loc<std::invalid_argument>(sham::format(
                "set_edges: edge \"{}\" is null: expected type {}", edge_name, type_name));
        }

        auto casted = std::dynamic_pointer_cast<T>(edge);
        if (!casted) {
            const auto &e = *edge; // necessary to avoid -Wpotentially-evaluated-expression
            throw shambase::make_except_with_loc<std::invalid_argument>(sham::format(
                "set_edges: edge \"{}\" has the wrong type: expected {}, got {}",
                edge_name,
                type_name,
                typeid(e).name()));
        }

        return casted;
    }

    /// Same as above, but for optional edge slots: a null edge or a null-opt edge sentinel is
    /// interpreted as "no value" instead of raising an error.
    template<class T>
    inline std::optional<std::shared_ptr<T>> __node_edge_cast_checked_optional(
        const std::shared_ptr<IEdge> &edge, const char *edge_name, const char *type_name) {
        if (!edge || is_null_opt_edge(edge)) {
            return std::nullopt;
        }
        return __node_edge_cast_checked<T>(edge, edge_name, type_name);
    }

    /// Ensure that all edges passed to the generic vector-based INode::set_edges() were consumed,
    /// i.e. that neither too few nor too many edges were provided for a given side of the node.
    inline void __node_edge_check_count(size_t provided, size_t expected, const char *side_name) {
        if (provided != expected) {
            throw shambase::make_except_with_loc<std::invalid_argument>(sham::format(
                "set_edges: wrong number of {} edges provided: expected {}, got {}",
                side_name,
                expected,
                provided));
        }
    }

} // namespace shamrock::solvergraph

#define INODE_DECL_RO(type, name) const type &name;
#define INODE_DECL_RW(type, name) type & name;
#define INODE_PARAM_RO(type, name) const std::shared_ptr<type> &name,
#define INODE_PARAM_RW(type, name) const std::shared_ptr<type> &name,
#define INODE_PUSH_RO1(type, name) name,
#define INODE_PUSH_RW1(type, name)
#define INODE_PUSH_RO2(type, name)
#define INODE_PUSH_RW2(type, name) name,
#define INODE_GET_RO(type, name) get_ro_edge<type>(ro++),
#define INODE_GET_RW(type, name) get_rw_edge<type>(rw++),

/// Param type used by the untyped, per-slot set_edges_untyped() overload: same slot
/// count/order as the typed setter, but every slot is a plain IEdge, cast to its concrete type
/// before being forwarded to the typed setter. (Named differently from set_edges() itself: for a
/// node whose edge type already is IEdge, e.g. NodeFreeAlloc, the two would otherwise be the same
/// overload.)
#define INODE_PARAM_EDGE(type, name) const std::shared_ptr<shamrock::solvergraph::IEdge> &name,

#define INODE_CHECK_RO_ARG(type, name)                                                             \
    shamrock::solvergraph::__node_edge_cast_checked<type>(name, #name, #type),
#define INODE_CHECK_RW_ARG(type, name)                                                             \
    shamrock::solvergraph::__node_edge_cast_checked<type>(name, #name, #type),

/// Count/declare/forward macros used by the generic vector-based set_edges() overload to forward
/// each vector slot, in order, to the untyped per-slot setter above. The slots are bound to named
/// local variables first (rather than picked directly as call arguments) because argument
/// evaluation order is unspecified in C++, and here it must match the vector's slot order.
#define INODE_COUNT_RO(type, name) ro_count++;
#define INODE_COUNT_RW(type, name) rw_count++;
#define INODE_DECLARE_PICK_RO(type, name) auto &&name = ro_edges_in[ro_idx++];
#define INODE_DECLARE_PICK_RW(type, name) auto &&name = rw_edges_in[rw_idx++];
#define INODE_FORWARD_ARG(type, name) name,

#define INODE_DECL_RO_OPTIONAL(type, name)                                                         \
    const std::optional<std::reference_wrapper<const type>> name;
#define INODE_DECL_RW_OPTIONAL(type, name) const std::optional<std::reference_wrapper<type>> name;
#define INODE_PARAM_RO_OPTIONAL(type, name) const std::optional<std::shared_ptr<type>> &name,
#define INODE_PARAM_RW_OPTIONAL(type, name) const std::optional<std::shared_ptr<type>> &name,
#define INODE_PUSH_RO1_OPTIONAL(type, name) shamrock::solvergraph::obsfucate_null_opt_edge(name),
#define INODE_PUSH_RW1_OPTIONAL(type, name)
#define INODE_PUSH_RO2_OPTIONAL(type, name)
#define INODE_PUSH_RW2_OPTIONAL(type, name) shamrock::solvergraph::obsfucate_null_opt_edge(name),
#define INODE_GET_RO_OPTIONAL(type, name) get_ro_edge_optional<type>(ro++),
#define INODE_GET_RW_OPTIONAL(type, name) get_rw_edge_optional<type>(rw++),

#define INODE_CHECK_RO_ARG_OPTIONAL(type, name)                                                    \
    shamrock::solvergraph::__node_edge_cast_checked_optional<type>(name, #name, #type),
#define INODE_CHECK_RW_ARG_OPTIONAL(type, name)                                                    \
    shamrock::solvergraph::__node_edge_cast_checked_optional<type>(name, #name, #type),

#define EXPAND_NODE_EDGES(EDGES)                                                                   \
                                                                                                   \
    struct Edges {                                                                                 \
        EDGES(INODE_DECL_RO, INODE_DECL_RW)                                                        \
    };                                                                                             \
                                                                                                   \
    inline void set_edges(                                                                         \
        EDGES(INODE_PARAM_RO, INODE_PARAM_RW) SourceLocation loc = SourceLocation{}) {             \
        __shamrock_log_callsite(loc);                                                              \
                                                                                                   \
        __internal_set_ro_edges({EDGES(INODE_PUSH_RO1, INODE_PUSH_RW1)});                          \
        __internal_set_rw_edges({EDGES(INODE_PUSH_RO2, INODE_PUSH_RW2)});                          \
    }                                                                                              \
                                                                                                   \
    inline void set_edges_untyped(                                                                 \
        EDGES(INODE_PARAM_EDGE, INODE_PARAM_EDGE) SourceLocation loc = SourceLocation{}) {         \
        __shamrock_log_callsite(loc);                                                              \
                                                                                                   \
        set_edges(EDGES(INODE_CHECK_RO_ARG, INODE_CHECK_RW_ARG) loc);                              \
    }                                                                                              \
                                                                                                   \
    inline void set_edges(                                                                         \
        std::vector<std::shared_ptr<shamrock::solvergraph::IEdge>> ro_edges_in,                    \
        std::vector<std::shared_ptr<shamrock::solvergraph::IEdge>> rw_edges_in,                    \
        SourceLocation loc = SourceLocation{}) {                                                   \
        __shamrock_log_callsite(loc);                                                              \
                                                                                                   \
        size_t ro_count = 0;                                                                       \
        size_t rw_count = 0;                                                                       \
        EDGES(INODE_COUNT_RO, INODE_COUNT_RW)                                                      \
                                                                                                   \
        shamrock::solvergraph::__node_edge_check_count(ro_edges_in.size(), ro_count, "read-only"); \
        shamrock::solvergraph::__node_edge_check_count(                                            \
            rw_edges_in.size(), rw_count, "read-write");                                           \
                                                                                                   \
        size_t ro_idx = 0;                                                                         \
        size_t rw_idx = 0;                                                                         \
        EDGES(INODE_DECLARE_PICK_RO, INODE_DECLARE_PICK_RW)                                        \
        set_edges_untyped(EDGES(INODE_FORWARD_ARG, INODE_FORWARD_ARG) loc);                        \
    }                                                                                              \
                                                                                                   \
    inline Edges get_edges() {                                                                     \
        int ro = 0;                                                                                \
        int rw = 0;                                                                                \
        return Edges{EDGES(INODE_GET_RO, INODE_GET_RW)};                                           \
    }

#define EXPAND_NODE_EDGES_OPTIONAL(EDGES)                                                          \
                                                                                                   \
    struct Edges {                                                                                 \
        EDGES(INODE_DECL_RO, INODE_DECL_RW, INODE_DECL_RO_OPTIONAL, INODE_DECL_RW_OPTIONAL)        \
    };                                                                                             \
                                                                                                   \
    inline void set_edges(                                                                         \
        EDGES(INODE_PARAM_RO, INODE_PARAM_RW, INODE_PARAM_RO_OPTIONAL, INODE_PARAM_RW_OPTIONAL)    \
            SourceLocation loc = SourceLocation{}) {                                               \
        __shamrock_log_callsite(loc);                                                              \
                                                                                                   \
        __internal_set_ro_edges({EDGES(                                                            \
            INODE_PUSH_RO1, INODE_PUSH_RW1, INODE_PUSH_RO1_OPTIONAL, INODE_PUSH_RW1_OPTIONAL)});   \
        __internal_set_rw_edges({EDGES(                                                            \
            INODE_PUSH_RO2, INODE_PUSH_RW2, INODE_PUSH_RO2_OPTIONAL, INODE_PUSH_RW2_OPTIONAL)});   \
    }                                                                                              \
                                                                                                   \
    inline void set_edges_untyped(                                                                 \
        EDGES(INODE_PARAM_EDGE, INODE_PARAM_EDGE, INODE_PARAM_EDGE, INODE_PARAM_EDGE)              \
            SourceLocation loc = SourceLocation{}) {                                               \
        __shamrock_log_callsite(loc);                                                              \
                                                                                                   \
        set_edges(EDGES(                                                                           \
            INODE_CHECK_RO_ARG,                                                                    \
            INODE_CHECK_RW_ARG,                                                                    \
            INODE_CHECK_RO_ARG_OPTIONAL,                                                           \
            INODE_CHECK_RW_ARG_OPTIONAL) loc);                                                     \
    }                                                                                              \
                                                                                                   \
    inline void set_edges(                                                                         \
        std::vector<std::shared_ptr<shamrock::solvergraph::IEdge>> ro_edges_in,                    \
        std::vector<std::shared_ptr<shamrock::solvergraph::IEdge>> rw_edges_in,                    \
        SourceLocation loc = SourceLocation{}) {                                                   \
        __shamrock_log_callsite(loc);                                                              \
                                                                                                   \
        size_t ro_count = 0;                                                                       \
        size_t rw_count = 0;                                                                       \
        EDGES(INODE_COUNT_RO, INODE_COUNT_RW, INODE_COUNT_RO, INODE_COUNT_RW)                      \
                                                                                                   \
        shamrock::solvergraph::__node_edge_check_count(ro_edges_in.size(), ro_count, "read-only"); \
        shamrock::solvergraph::__node_edge_check_count(                                            \
            rw_edges_in.size(), rw_count, "read-write");                                           \
                                                                                                   \
        size_t ro_idx = 0;                                                                         \
        size_t rw_idx = 0;                                                                         \
        EDGES(                                                                                     \
            INODE_DECLARE_PICK_RO,                                                                 \
            INODE_DECLARE_PICK_RW,                                                                 \
            INODE_DECLARE_PICK_RO,                                                                 \
            INODE_DECLARE_PICK_RW)                                                                 \
        set_edges_untyped(EDGES(                                                                   \
            INODE_FORWARD_ARG, INODE_FORWARD_ARG, INODE_FORWARD_ARG, INODE_FORWARD_ARG) loc);      \
    }                                                                                              \
                                                                                                   \
    inline Edges get_edges() {                                                                     \
        int ro = 0;                                                                                \
        int rw = 0;                                                                                \
        return Edges{                                                                              \
            EDGES(INODE_GET_RO, INODE_GET_RW, INODE_GET_RO_OPTIONAL, INODE_GET_RW_OPTIONAL)};      \
    }
