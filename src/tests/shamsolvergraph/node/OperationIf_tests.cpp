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
#include "shamsolvergraph/node/OperationIf.hpp"
#include "shamtest/shamtest.hpp"
#include <memory>
#include <string>

namespace {

    class ProbeNode : public shamrock::solvergraph::INode {
        public:
        std::shared_ptr<shamrock::solvergraph::IDataEdge<u32>> counter;
        u32 increment;

        ProbeNode(std::shared_ptr<shamrock::solvergraph::IDataEdge<u32>> counter, u32 increment)
            : counter(std::move(counter)), increment(increment) {}

        void _impl_evaluate_internal() override { counter->data += increment; }

        std::string _impl_get_label() const override { return "Probe"; }
        std::string _impl_get_tex() const override { return ""; }
    };

} // namespace

NEW_TEST(Unittest, "shamsolvergraph/node/OperationIf", 1) {
    using namespace shamrock::solvergraph;

    auto make_counter = []() {
        return IDataEdge<u32>::make_shared("counter", "c");
    };
    auto make_condition = [](bool value) {
        auto cond  = IDataEdge<bool>::make_shared("cond", "cond");
        cond->data = value;
        return cond;
    };

    {
        auto then_count = make_counter();
        auto cond       = make_condition(true);
        auto then_node  = std::make_shared<ProbeNode>(then_count, u32{1});

        OperationIf node("if", then_node);
        node.set_edges(cond);
        node.evaluate();

        REQUIRE_EQUAL(then_count->data, u32{1});
    }

    {
        auto then_count = make_counter();
        auto cond       = make_condition(false);
        auto then_node  = std::make_shared<ProbeNode>(then_count, u32{1});

        OperationIf node("if", then_node);
        node.set_edges(cond);
        node.evaluate();

        REQUIRE_EQUAL(then_count->data, u32{0});
    }

    {
        auto then_count = make_counter();
        auto else_count = make_counter();
        auto cond       = make_condition(true);
        auto then_node  = std::make_shared<ProbeNode>(then_count, u32{1});
        auto else_node  = std::make_shared<ProbeNode>(else_count, u32{2});

        OperationIf node("if", then_node, else_node);
        node.set_edges(cond);
        node.evaluate();

        REQUIRE_EQUAL(then_count->data, u32{1});
        REQUIRE_EQUAL(else_count->data, u32{0});
    }

    {
        auto then_count = make_counter();
        auto else_count = make_counter();
        auto cond       = make_condition(false);
        auto then_node  = std::make_shared<ProbeNode>(then_count, u32{1});
        auto else_node  = std::make_shared<ProbeNode>(else_count, u32{2});

        OperationIf node("if", then_node, else_node);
        node.set_edges(cond);
        node.evaluate();

        REQUIRE_EQUAL(then_count->data, u32{0});
        REQUIRE_EQUAL(else_count->data, u32{2});
    }

    {
        auto else_count = make_counter();
        auto cond       = make_condition(true);
        auto else_node  = std::make_shared<ProbeNode>(else_count, u32{2});

        OperationIf node("if", {}, else_node);
        node.set_edges(cond);
        node.evaluate();

        REQUIRE_EQUAL(else_count->data, u32{0});
    }

    {
        auto else_count = make_counter();
        auto cond       = make_condition(false);
        auto else_node  = std::make_shared<ProbeNode>(else_count, u32{2});

        OperationIf node("if", {}, else_node);
        node.set_edges(cond);
        node.evaluate();

        REQUIRE_EQUAL(else_count->data, u32{2});
    }

    {
        auto cond = make_condition(true);
        OperationIf node("if");
        node.set_edges(cond);
        node.evaluate();
        REQUIRE(node.get_dot_graph().find("true") != std::string::npos);
        REQUIRE(node.get_tex().find("Then") == std::string::npos);
        REQUIRE(node.get_tex().find("Else") == std::string::npos);
    }

    {
        auto then_count = make_counter();
        auto else_count = make_counter();
        auto cond       = make_condition(true);
        auto then_node  = std::make_shared<ProbeNode>(then_count, u32{1});
        auto else_node  = std::make_shared<ProbeNode>(else_count, u32{2});

        OperationIf node("if", then_node, else_node);
        node.set_edges(cond);

        std::string dot = node.get_dot_graph();
        REQUIRE(dot.find("true") != std::string::npos);
        REQUIRE(dot.find("false") != std::string::npos);
        REQUIRE(node.get_tex().find("Then") != std::string::npos);
        REQUIRE(node.get_tex().find("Else") != std::string::npos);
        REQUIRE_EQUAL(node.get_label(), std::string{"if"});
    }
}
