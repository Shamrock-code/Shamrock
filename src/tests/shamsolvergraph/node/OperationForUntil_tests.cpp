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
#include "shamsolvergraph/node/OperationForUntil.hpp"
#include "shamtest/shamtest.hpp"
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

    /// Records the iteration index it sees, and sets `condition` to false after `limit` calls.
    class CountdownNode : public shamrock::solvergraph::INode {
        public:
        std::shared_ptr<shamrock::solvergraph::IDataEdge<u32>> iter;
        std::shared_ptr<shamrock::solvergraph::IDataEdge<bool>> condition;
        u32 limit;
        std::vector<u32> seen_iters;

        CountdownNode(
            std::shared_ptr<shamrock::solvergraph::IDataEdge<u32>> iter,
            std::shared_ptr<shamrock::solvergraph::IDataEdge<bool>> condition,
            u32 limit)
            : iter(std::move(iter)), condition(std::move(condition)), limit(limit) {}

        void _impl_evaluate_internal() override {
            seen_iters.push_back(iter->data);
            condition->data = seen_iters.size() < limit;
        }

        std::string _impl_get_label() const override { return "Countdown"; }
        std::string _impl_get_tex() const override { return ""; }
    };

} // namespace

NEW_TEST(Unittest, "shamsolvergraph/node/OperationForUntil", 1) {
    using namespace shamrock::solvergraph;

    auto make_iter = []() {
        auto iter  = IDataEdge<u32>::make_shared("iter", "i");
        iter->data = 42; // must be reset by the node
        return iter;
    };
    auto make_condition = [](bool value) {
        auto cond  = IDataEdge<bool>::make_shared("cond", "cond");
        cond->data = value;
        return cond;
    };

    // condition becomes false before max_iter: stops on the condition
    {
        auto iter = make_iter();
        auto cond = make_condition(true);
        auto body = std::make_shared<CountdownNode>(iter, cond, u32{3});

        OperationForUntil node("for", body, 10);
        node.set_edges(cond, iter);
        node.evaluate();

        REQUIRE_EQUAL(body->seen_iters, (std::vector<u32>{0, 1, 2}));
        REQUIRE_EQUAL(iter->data, u32{3});
        REQUIRE_EQUAL(cond->data, false);
    }

    // condition never becomes false: stops after max_iter
    {
        auto iter = make_iter();
        auto cond = make_condition(true);
        auto body = std::make_shared<CountdownNode>(iter, cond, u32{100});

        OperationForUntil node("for", body, 4);
        node.set_edges(cond, iter);
        node.evaluate();

        REQUIRE_EQUAL(body->seen_iters, (std::vector<u32>{0, 1, 2, 3}));
        REQUIRE_EQUAL(iter->data, u32{4});
        REQUIRE_EQUAL(cond->data, true);
    }

    // condition becomes false exactly at max_iter
    {
        auto iter = make_iter();
        auto cond = make_condition(true);
        auto body = std::make_shared<CountdownNode>(iter, cond, u32{4});

        OperationForUntil node("for", body, 4);
        node.set_edges(cond, iter);
        node.evaluate();

        REQUIRE_EQUAL(iter->data, u32{4});
        REQUIRE_EQUAL(cond->data, false);
    }

    // condition initially false: body never runs
    {
        auto iter = make_iter();
        auto cond = make_condition(false);
        auto body = std::make_shared<CountdownNode>(iter, cond, u32{3});

        OperationForUntil node("for", body, 10);
        node.set_edges(cond, iter);
        node.evaluate();

        REQUIRE(body->seen_iters.empty());
        REQUIRE_EQUAL(iter->data, u32{0});
    }

    // max_iter = 0: body never runs
    {
        auto iter = make_iter();
        auto cond = make_condition(true);
        auto body = std::make_shared<CountdownNode>(iter, cond, u32{3});

        OperationForUntil node("for", body, 0);
        node.set_edges(cond, iter);
        node.evaluate();

        REQUIRE(body->seen_iters.empty());
        REQUIRE_EQUAL(iter->data, u32{0});
    }

    // a body is mandatory
    {
        REQUIRE_EXCEPTION_THROW(OperationForUntil("for", {}, 10), std::invalid_argument);
    }

    // dot graph & tex
    {
        auto iter = make_iter();
        auto cond = make_condition(true);
        auto body = std::make_shared<CountdownNode>(iter, cond, u32{1});

        OperationForUntil node("for", body, 7);
        node.set_edges(cond, iter);

        std::string dot = node.get_dot_graph();
        REQUIRE(dot.find("true") != std::string::npos);
        REQUIRE(dot.find("false") != std::string::npos);
        REQUIRE(dot.find("i++") != std::string::npos);
        REQUIRE(dot.find("i < 7") != std::string::npos);
        REQUIRE(node.get_tex().find("at most 7 iterations") != std::string::npos);
        REQUIRE_EQUAL(node.get_label(), std::string{"for"});
    }
}
