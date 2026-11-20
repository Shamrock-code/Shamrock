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
#include "shamsolvergraph/node/OperationFor.hpp"
#include "shamtest/shamtest.hpp"
#include <memory>
#include <string>

namespace {

    /// Adds `increment` to `counter` and, if set, overwrites `count` (to check it is read once)
    class ProbeNode : public shamrock::solvergraph::INode {
        public:
        std::shared_ptr<shamrock::solvergraph::IDataEdge<u32>> counter;
        std::shared_ptr<shamrock::solvergraph::IDataEdge<u32>> count;
        u32 increment;

        ProbeNode(
            std::shared_ptr<shamrock::solvergraph::IDataEdge<u32>> counter,
            u32 increment,
            std::shared_ptr<shamrock::solvergraph::IDataEdge<u32>> count = {})
            : counter(std::move(counter)), count(std::move(count)), increment(increment) {}

        void _impl_evaluate_internal() override {
            counter->data += increment;
            if (count) {
                count->data = 1000;
            }
        }

        std::string _impl_get_label() const override { return "Probe"; }
        std::string _impl_get_tex() const override { return ""; }
    };

} // namespace

NEW_TEST(Unittest, "shamsolvergraph/node/OperationFor", 1) {
    using namespace shamrock::solvergraph;

    auto make_edge = [](u32 value) {
        auto e  = IDataEdge<u32>::make_shared("e", "e");
        e->data = value;
        return e;
    };

    // evaluated count times
    for (u32 n : {0u, 1u, 7u}) {
        auto counter = make_edge(0);
        auto count   = make_edge(n);

        OperationFor node("for", std::make_shared<ProbeNode>(counter, u32{2}));
        node.set_edges(count);
        node.evaluate();

        REQUIRE_EQUAL(counter->data, 2 * n);
    }

    // the count is read once before the first iteration
    {
        auto counter = make_edge(0);
        auto count   = make_edge(3);

        OperationFor node("for", std::make_shared<ProbeNode>(counter, u32{1}, count));
        node.set_edges(count);
        node.evaluate();

        REQUIRE_EQUAL(counter->data, u32{3});
    }

    // dot graph & tex do not throw
    {
        auto counter = make_edge(0);
        auto count   = make_edge(1);

        OperationFor node("for", std::make_shared<ProbeNode>(counter, u32{1}));
        node.set_edges(count);

        REQUIRE(node.get_dot_graph().size() > 0);
        REQUIRE(node.get_tex().size() > 0);
    }

    REQUIRE_EXCEPTION_THROW(OperationFor("for", nullptr), std::invalid_argument);
}
