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
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

    /// Records the iteration index it sees (if an `iter_count` edge is given), and sets
    /// `condition` to false after `limit` calls.
    class CountdownNode : public shamrock::solvergraph::INode {
        public:
        std::shared_ptr<shamrock::solvergraph::IDataEdge<u32>> iter_count;
        std::shared_ptr<shamrock::solvergraph::IDataEdge<bool>> condition;
        u32 limit;
        u32 calls = 0;
        std::vector<u32> seen_iters;

        CountdownNode(
            std::shared_ptr<shamrock::solvergraph::IDataEdge<u32>> iter_count,
            std::shared_ptr<shamrock::solvergraph::IDataEdge<bool>> condition,
            u32 limit)
            : iter_count(std::move(iter_count)), condition(std::move(condition)), limit(limit) {}

        void _impl_evaluate_internal() override {
            calls++;
            if (iter_count) {
                seen_iters.push_back(iter_count->data);
            }
            condition->data = calls < limit;
        }

        std::string _impl_get_label() const override { return "Countdown"; }
        std::string _impl_get_tex() const override { return ""; }
    };

} // namespace

NEW_TEST(Unittest, "shamsolvergraph/node/OperationForUntil", 1) {
    using namespace shamrock::solvergraph;

    auto make_iter_count = []() {
        auto iter_count  = IDataEdge<u32>::make_shared("iter_count", "n_iter");
        iter_count->data = 42; // must be reset by the node
        return iter_count;
    };
    auto make_condition = [](bool value) {
        auto cond  = IDataEdge<bool>::make_shared("cond", "cond");
        cond->data = value;
        return cond;
    };

    // condition becomes false before max_iter: stops on the condition
    {
        auto iter_count = make_iter_count();
        auto cond       = make_condition(true);
        auto body       = std::make_shared<CountdownNode>(iter_count, cond, u32{3});

        OperationForUntil node("for", body, 10);
        node.set_edges(cond, iter_count);
        node.evaluate();

        REQUIRE_EQUAL(body->seen_iters, (std::vector<u32>{0, 1, 2}));
        REQUIRE_EQUAL(iter_count->data, u32{3});
        REQUIRE_EQUAL(cond->data, false);
    }

    // condition never becomes false: stops after max_iter
    {
        auto iter_count = make_iter_count();
        auto cond       = make_condition(true);
        auto body       = std::make_shared<CountdownNode>(iter_count, cond, u32{100});

        OperationForUntil node("for", body, 4);
        node.set_edges(cond, iter_count);
        node.evaluate();

        REQUIRE_EQUAL(body->seen_iters, (std::vector<u32>{0, 1, 2, 3}));
        REQUIRE_EQUAL(iter_count->data, u32{4});
        REQUIRE_EQUAL(cond->data, true);
    }

    // condition becomes false exactly at max_iter
    {
        auto iter_count = make_iter_count();
        auto cond       = make_condition(true);
        auto body       = std::make_shared<CountdownNode>(iter_count, cond, u32{4});

        OperationForUntil node("for", body, 4);
        node.set_edges(cond, iter_count);
        node.evaluate();

        REQUIRE_EQUAL(iter_count->data, u32{4});
        REQUIRE_EQUAL(cond->data, false);
    }

    // condition initially false: body never runs
    {
        auto iter_count = make_iter_count();
        auto cond       = make_condition(false);
        auto body       = std::make_shared<CountdownNode>(iter_count, cond, u32{3});

        OperationForUntil node("for", body, 10);
        node.set_edges(cond, iter_count);
        node.evaluate();

        REQUIRE(body->seen_iters.empty());
        REQUIRE_EQUAL(iter_count->data, u32{0});
    }

    // max_iter = 0: body never runs
    {
        auto iter_count = make_iter_count();
        auto cond       = make_condition(true);
        auto body       = std::make_shared<CountdownNode>(iter_count, cond, u32{3});

        OperationForUntil node("for", body, 0);
        node.set_edges(cond, iter_count);
        node.evaluate();

        REQUIRE(body->seen_iters.empty());
        REQUIRE_EQUAL(iter_count->data, u32{0});
    }

    // iter_count edge absent: the loop still stops on the condition or on max_iter
    {
        auto cond = make_condition(true);
        auto body = std::make_shared<CountdownNode>(nullptr, cond, u32{3});

        OperationForUntil node("for", body, 10);
        node.set_edges(cond, std::nullopt);
        node.evaluate();

        REQUIRE_EQUAL(body->calls, u32{3});
        REQUIRE_EQUAL(cond->data, false);
    }
    {
        auto cond = make_condition(true);
        auto body = std::make_shared<CountdownNode>(nullptr, cond, u32{100});

        OperationForUntil node("for", body, 4);
        node.set_edges(cond, std::nullopt);
        node.evaluate();

        REQUIRE_EQUAL(body->calls, u32{4});
        REQUIRE_EQUAL(cond->data, true);
    }

    // a body is mandatory
    {
        REQUIRE_EXCEPTION_THROW(OperationForUntil("for", {}, 10), std::invalid_argument);
    }

    // dot graph & tex
    {
        auto iter_count = make_iter_count();
        auto cond       = make_condition(true);
        auto body       = std::make_shared<CountdownNode>(iter_count, cond, u32{1});

        OperationForUntil node("for", body, 7);
        node.set_edges(cond, iter_count);

        std::string dot = node.get_dot_graph();
        REQUIRE(dot.find("true") != std::string::npos);
        REQUIRE(dot.find("false") != std::string::npos);
        REQUIRE(dot.find("i++") != std::string::npos);
        REQUIRE(dot.find("i < 7") != std::string::npos);
        REQUIRE(node.get_tex().find("at most 7 iterations") != std::string::npos);
        REQUIRE_EQUAL(node.get_label(), std::string{"for"});
        REQUIRE(dot.find(iter_count->get_label()) != std::string::npos);
    }

    // dot graph without the iter_count edge does not show the null placeholder
    {
        auto cond = make_condition(true);
        auto body = std::make_shared<CountdownNode>(nullptr, cond, u32{1});

        OperationForUntil node("for", body, 7);
        node.set_edges(cond, std::nullopt);

        std::string dot = node.get_dot_graph();
        REQUIRE(dot.find("i < 7") != std::string::npos);
        REQUIRE(dot.find("null_opt_edge") == std::string::npos);
    }
}
