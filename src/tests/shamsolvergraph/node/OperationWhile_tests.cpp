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
#include "shamsolvergraph/node/OperationWhile.hpp"
#include "shamtest/shamtest.hpp"
#include <memory>
#include <stdexcept>
#include <string>

namespace {

    /// Increments `counter` and sets `condition` to false once `counter` reaches `limit`.
    class CountdownNode : public shamrock::solvergraph::INode {
        public:
        std::shared_ptr<shamrock::solvergraph::IDataEdge<u32>> counter;
        std::shared_ptr<shamrock::solvergraph::IDataEdge<bool>> condition;
        u32 limit;

        CountdownNode(
            std::shared_ptr<shamrock::solvergraph::IDataEdge<u32>> counter,
            std::shared_ptr<shamrock::solvergraph::IDataEdge<bool>> condition,
            u32 limit)
            : counter(std::move(counter)), condition(std::move(condition)), limit(limit) {}

        void _impl_evaluate_internal() override {
            counter->data += 1;
            condition->data = counter->data < limit;
        }

        std::string _impl_get_label() const override { return "Countdown"; }
        std::string _impl_get_tex() const override { return ""; }
    };

} // namespace

NEW_TEST(Unittest, "shamsolvergraph/node/OperationWhile", 1) {
    using namespace shamrock::solvergraph;

    auto make_counter = []() {
        return IDataEdge<u32>::make_shared("counter", "c");
    };
    auto make_condition = [](bool value) {
        auto cond  = IDataEdge<bool>::make_shared("cond", "cond");
        cond->data = value;
        return cond;
    };

    // condition initially true: body runs until it sets the condition to false
    {
        auto count = make_counter();
        auto cond  = make_condition(true);
        auto body  = std::make_shared<CountdownNode>(count, cond, u32{5});

        OperationWhile node("while", body);
        node.set_edges(cond);
        node.evaluate();

        REQUIRE_EQUAL(count->data, u32{5});
        REQUIRE_EQUAL(cond->data, false);
    }

    // condition initially false: body never runs
    {
        auto count = make_counter();
        auto cond  = make_condition(false);
        auto body  = std::make_shared<CountdownNode>(count, cond, u32{5});

        OperationWhile node("while", body);
        node.set_edges(cond);
        node.evaluate();

        REQUIRE_EQUAL(count->data, u32{0});
    }

    // re-evaluation after resetting the condition runs the loop again
    {
        auto count = make_counter();
        auto cond  = make_condition(true);
        auto body  = std::make_shared<CountdownNode>(count, cond, u32{3});

        OperationWhile node("while", body);
        node.set_edges(cond);
        node.evaluate();
        REQUIRE_EQUAL(count->data, u32{3});

        count->data = 0;
        cond->data  = true;
        node.evaluate();
        REQUIRE_EQUAL(count->data, u32{3});
    }

    // a body is mandatory
    {
        REQUIRE_EXCEPTION_THROW(OperationWhile("while", {}), std::invalid_argument);
    }

    // dot graph & tex
    {
        auto count = make_counter();
        auto cond  = make_condition(true);
        auto body  = std::make_shared<CountdownNode>(count, cond, u32{1});

        OperationWhile node("while", body);
        node.set_edges(cond);

        std::string dot = node.get_dot_graph();
        REQUIRE(dot.find("true") != std::string::npos);
        REQUIRE(dot.find("false") != std::string::npos);
        REQUIRE(dot.find("loop") != std::string::npos);
        REQUIRE(node.get_tex().find("While") != std::string::npos);
        REQUIRE_EQUAL(node.get_label(), std::string{"while"});
    }
}
