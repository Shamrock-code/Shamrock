// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file OperationForUntil.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Implementation of the OperationForUntil meta node
 *
 */

#include "shambase/exception.hpp"
#include "shamsolvergraph/edge/INullOptEdge.hpp"
#include "shamsolvergraph/node/OperationForUntil.hpp"
#include <sstream>
#include <stdexcept>

namespace shamrock::solvergraph {

    OperationForUntil::OperationForUntil(
        std::string name, std::shared_ptr<INode> body_node, u32 max_iter)
        : body_node(std::move(body_node)), max_iter(max_iter), name(std::move(name)) {
        if (!this->body_node) {
            shambase::throw_with_loc<std::invalid_argument>(
                "OperationForUntil must have a body node");
        }
    }

    void OperationForUntil::_impl_evaluate_internal() {
        auto edges = get_edges();

        // Resolve the optional edge once, outside of the loop, and use its data as the loop
        // counter. When it is absent the counter is a local instead, so the loop needs no
        // per-iteration check.
        u32 unused_iter_count = 0;
        u32 &i = edges.iter_count.has_value() ? edges.iter_count->get().data : unused_iter_count;

        // The edge may hold a stale value (e.g. from a previous evaluation): always restart at 0.
        i = 0;

        // The condition is re-read at every iteration since the body is expected to update it.
        for (; i < max_iter && edges.condition.data; i++) {
            body_node->evaluate();
        }
    }

    std::string OperationForUntil::_impl_get_dot_graph_partial() const {

        std::stringstream ss;

        ss << "subgraph cluster_" + std::to_string(get_uuid()) + " {\n";
        ss << sham::format(
            "n_{} [label=\"{}\\ni < {}\", shape=diamond];\n",
            get_uuid(),
            _impl_get_label(),
            max_iter);

        ss << body_node->get_dot_graph_partial();
        ss << sham::format(
            "n_{} -> {} [label=\"true\"];\n", get_uuid(), body_node->get_dot_graph_node_start());
        ss << body_node->get_dot_graph_node_end() << " -> "
           << sham::format("n_{} [label=\"i++\", constraint=false];\n", get_uuid());

        ss << sham::format("n_{} -> n_{}_end [label=\"false\"];\n", get_uuid(), get_uuid());

        ss << sham::format("n_{}_end [label=\"\", shape=point, width=0.15];\n", get_uuid());
        ss << sham::format("label = \"{}\";\n", _impl_get_label());
        ss << "}\n";

        auto &cond = get_ro_edge_base(0);
        ss << sham::format(
            "e_{} -> n_{} [style=\"dashed\", color=green];\n", cond.get_uuid(), get_uuid());
        ss << sham::format(
            "e_{} [label=\"{}\",shape=rect, style=filled];\n", cond.get_uuid(), cond.get_label());

        auto &iter_count = get_rw_edge_base(0);
        if (dynamic_cast<const INullOptEdge *>(&iter_count) == nullptr) {
            ss << sham::format(
                "n_{} -> e_{} [style=\"dashed\", color=red];\n", get_uuid(), iter_count.get_uuid());
            ss << sham::format(
                "e_{} [label=\"{}\",shape=rect, style=filled];\n",
                iter_count.get_uuid(),
                iter_count.get_label());
        }

        return ss.str();
    }

    std::string OperationForUntil::_impl_get_tex() const {
        std::stringstream ss;
        ss << "Start : " << _impl_get_label() << "\n";
        ss << "For (at most " << max_iter << " iterations) :\n"
           << body_node->get_tex_partial() << "\n";
        ss << "End : " << _impl_get_label() << "\n";
        return ss.str();
    }

} // namespace shamrock::solvergraph
