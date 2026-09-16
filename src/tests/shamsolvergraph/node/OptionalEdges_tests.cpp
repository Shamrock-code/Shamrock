// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamsolvergraph/edge/IDataEdge.hpp"
#include "shamsolvergraph/node/INode.hpp"
#include "shamtest/shamtest.hpp"
#include <memory>
#include <optional>

namespace {

#define NODE_EDGES(X_RO, X_RW, X_RO_OPTIONAL, X_RW_OPTIONAL)                                       \
    X_RO_OPTIONAL(shamrock::solvergraph::IDataEdge<u32>, opt_a)                                    \
    X_RO_OPTIONAL(shamrock::solvergraph::IDataEdge<u32>, opt_b)                                    \
    X_RW(shamrock::solvergraph::IDataEdge<u32>, out)

    class OptionalEdgeProbeNode : public shamrock::solvergraph::INode {
        public:
        EXPAND_NODE_EDGES_OPTIONAL(NODE_EDGES)

        void _impl_evaluate_internal() override {
            auto edges     = get_edges();
            bool has_a     = edges.opt_a.has_value();
            bool has_b     = edges.opt_b.has_value();
            edges.out.data = (has_a ? 1u : 0u) | (has_b ? 2u : 0u);
        }

        std::string _impl_get_label() const override { return "OptionalEdgeProbe"; }
        std::string _impl_get_tex() const override { return ""; }
    };

#undef NODE_EDGES

} // namespace

NEW_TEST(Unittest, "shamsolvergraph/node/OptionalEdges", 1) {
    using namespace shamrock::solvergraph;

    auto make_opt_edge = []() {
        return IDataEdge<u32>::make_shared("opt", "o");
    };

    auto make_out_edge = []() {
        return IDataEdge<u32>::make_shared("out", "out");
    };

    {
        auto out = make_out_edge();
        OptionalEdgeProbeNode node;
        node.set_edges(std::nullopt, std::nullopt, out);
        node.evaluate();
        REQUIRE_EQUAL(out->data, u32{0});
    }

    {
        auto a   = make_opt_edge();
        auto out = make_out_edge();
        OptionalEdgeProbeNode node;
        node.set_edges(a, std::nullopt, out);
        node.evaluate();
        REQUIRE_EQUAL(out->data, u32{1});
    }

    {
        auto b   = make_opt_edge();
        auto out = make_out_edge();
        OptionalEdgeProbeNode node;
        node.set_edges(std::nullopt, b, out);
        node.evaluate();
        REQUIRE_EQUAL(out->data, u32{2});
    }

    {
        auto a   = make_opt_edge();
        auto b   = make_opt_edge();
        auto out = make_out_edge();
        OptionalEdgeProbeNode node;
        node.set_edges(a, b, out);
        node.evaluate();
        REQUIRE_EQUAL(out->data, u32{3});
    }
}
