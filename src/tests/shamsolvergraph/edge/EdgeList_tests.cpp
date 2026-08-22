// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamsolvergraph/edge/EdgeList.hpp"
#include "shamsolvergraph/edge/IDataEdge.hpp"
#include "shamsolvergraph/node/INode.hpp"
#include "shamtest/shamtest.hpp"
#include <memory>
#include <vector>

namespace {

#define NODE_EDGES(X_RO, X_RW)                                                                     \
    X_RO(shamrock::solvergraph::IDataEdge<u32>, offset)                                            \
    X_RO(shamrock::solvergraph::EdgeList<shamrock::solvergraph::IDataEdge<u32>>, ins)              \
    X_RW(shamrock::solvergraph::IDataEdge<u32>, out)

    /// Sums a runtime sized list of inputs, shifted by a fixed size input
    class EdgeListProbeNode : public shamrock::solvergraph::INode {
        public:
        EXPAND_NODE_EDGES(NODE_EDGES)

        void _impl_evaluate_internal() override {
            auto edges = get_edges();

            u32 acc = edges.offset.data;
            edges.ins.for_each([&](u32 i, const shamrock::solvergraph::IDataEdge<u32> &in) {
                acc += in.data;
            });

            edges.out.data = acc;
        }

        std::string _impl_get_label() const override { return "EdgeListProbe"; }
        std::string _impl_get_tex() const override { return ""; }
    };

#undef NODE_EDGES

    std::shared_ptr<shamrock::solvergraph::IDataEdge<u32>> make_data_edge(u32 value) {
        auto edge  = shamrock::solvergraph::IDataEdge<u32>::make_shared("in", "in");
        edge->data = value;
        return edge;
    }

} // namespace

NEW_TEST(Unittest, "shamsolvergraph/edge/EdgeList", 1) {
    using namespace shamrock::solvergraph;

    using ListT = EdgeList<IDataEdge<u32>>;

    auto make_out_edge = []() {
        return IDataEdge<u32>::make_shared("out", "out");
    };

    { // empty list
        auto offset = make_data_edge(7);
        auto ins    = ListT::make_shared("ins", "ins");
        auto out    = make_out_edge();

        REQUIRE_EQUAL(ins->size(), u32{0});
        REQUIRE(ins->empty());

        EdgeListProbeNode node;
        node.set_edges(offset, ins, out);
        node.evaluate();
        REQUIRE_EQUAL(out->data, u32{7});
    }

    { // single entry
        auto offset = make_data_edge(7);
        auto ins    = ListT::make_shared("ins", "ins");
        auto out    = make_out_edge();

        ins->set_entries({make_data_edge(3)});

        REQUIRE_EQUAL(ins->size(), u32{1});

        EdgeListProbeNode node;
        node.set_edges(offset, ins, out);
        node.evaluate();
        REQUIRE_EQUAL(out->data, u32{10});
    }

    { // three entries, and the list can be refilled between evaluations
        auto offset = make_data_edge(7);
        auto ins    = ListT::make_shared("ins", "ins");
        auto out    = make_out_edge();

        ins->set_entries({make_data_edge(1), make_data_edge(2), make_data_edge(4)});

        REQUIRE_EQUAL(ins->size(), u32{3});
        REQUIRE_EQUAL(ins->get(0).data, u32{1});
        REQUIRE_EQUAL(ins->get(2).data, u32{4});

        EdgeListProbeNode node;
        node.set_edges(offset, ins, out);
        node.evaluate();
        REQUIRE_EQUAL(out->data, u32{14});

        // same node, different arity
        ins->set_entries({make_data_edge(10), make_data_edge(20)});
        node.evaluate();
        REQUIRE_EQUAL(out->data, u32{37});
    }

    { // the listed edges are visible to the graph tooling
        auto ins = ListT::make_shared("ins", "ins");
        ins->set_entries({make_data_edge(1), make_data_edge(2)});

        REQUIRE_EQUAL(ins->get_sub_edges().size(), std::size_t{2});
        REQUIRE_EQUAL(ins->get_sub_edges()[0]->get_uuid(), ins->get(0).get_uuid());
    }

    { // free_alloc drops the list without destroying the listed edges
        auto ins   = ListT::make_shared("ins", "ins");
        auto entry = make_data_edge(5);
        ins->set_entries({entry});

        REQUIRE_EQUAL(entry.use_count(), long{2});

        ins->free_alloc();

        REQUIRE_EQUAL(ins->size(), u32{0});
        REQUIRE_EQUAL(entry.use_count(), long{1});
        REQUIRE_EQUAL(entry->data, u32{5});
    }

    { // a null entry is rejected
        auto ins = ListT::make_shared("ins", "ins");
        REQUIRE_EXCEPTION_THROW(
            ins->set_entries({make_data_edge(1), nullptr}), std::invalid_argument);
    }
}
