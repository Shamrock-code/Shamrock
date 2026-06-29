// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamrock/solvergraph/IEdge.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/SolverGraph.hpp"
#include "shamtest/shamtest.hpp"

class TestEdge : public shamrock::solvergraph::IEdge {
    public:
    std::string _impl_get_dot_label() const override { return "dot"; }
    std::string _impl_get_tex_symbol() const override { return "tex"; }
    void free_alloc() override {}
};

class TestNode : public shamrock::solvergraph::INode {
    public:
    void _impl_evaluate_internal() override {}
    std::string _impl_get_label() const override { return "node_label"; }
    std::string _impl_get_tex() const override { return "node_tex"; }
};

NEW_TEST(Unittest, "shamrock/solvergraph/SolverGraph", 1) {
    using namespace shamrock::solvergraph;

    {
        SolverGraph graph{};

        graph.register_node("my_node", TestNode{});
        graph.register_edge("my_edge", TestEdge{});

        REQUIRE_EQUAL(graph.get_node_names().size(), 1);
        REQUIRE_EQUAL(graph.get_edge_names().size(), 1);
        REQUIRE_EQUAL(graph.get_node_ptr<TestNode>("my_node")->_impl_get_label(), "node_label");
        REQUIRE_EQUAL(graph.get_edge_ptr<TestEdge>("my_edge")->get_label(), "dot");
    }

    {
        SolverGraph graph{};

        graph.register_node("b", TestNode{});
        graph.register_node("a", TestNode{});
        graph.register_edge("z", TestEdge{});
        graph.register_edge("y", TestEdge{});

        REQUIRE_EQUAL_NAMED(
            "sorted node names", graph.get_node_names(), (std::vector<std::string>{"a", "b"}));
        REQUIRE_EQUAL_NAMED(
            "sorted edge names", graph.get_edge_names(), (std::vector<std::string>{"y", "z"}));
    }

    {
        auto graph = SolverGraph::with_constraint(
            SolverGraphConstraint{
                .name = "test_constraint",
                .node_check =
                    [](const std::shared_ptr<INode> &) {
                        return false;
                    },
            });

        REQUIRE_EXCEPTION_THROW(
            graph.register_node("rejected_node", TestNode{}), std::invalid_argument);
        REQUIRE_EQUAL(graph.get_node_names().size(), 0);
    }

    {
        auto graph = SolverGraph::with_constraint(
            SolverGraphConstraint{
                .name = "test_constraint",
                .edge_check =
                    [](const std::shared_ptr<IEdge> &) {
                        return false;
                    },
            });

        REQUIRE_EXCEPTION_THROW(
            graph.register_edge("rejected_edge", TestEdge{}), std::invalid_argument);
        REQUIRE_EQUAL(graph.get_edge_names().size(), 0);
    }

    {
        auto graph = SolverGraph::with_constraint(
            SolverGraphConstraint{
                .name = "both_checks",
                .node_check =
                    [](const std::shared_ptr<INode> &) {
                        return true;
                    },
                .edge_check =
                    [](const std::shared_ptr<IEdge> &) {
                        return false;
                    },
            });

        graph.register_node("ok_node", TestNode{});
        REQUIRE_EQUAL(graph.get_node_names().size(), 1);

        REQUIRE_EXCEPTION_THROW(graph.register_edge("bad_edge", TestEdge{}), std::invalid_argument);
        REQUIRE_EQUAL(graph.get_edge_names().size(), 0);
    }

    {
        SolverGraph graph{};

        graph.register_node("dup", TestNode{});

        REQUIRE_EXCEPTION_THROW(graph.register_node("dup", TestNode{}), std::invalid_argument);
    }

    {
        SolverGraph graph{};

        graph.register_edge("dup", TestEdge{});

        REQUIRE_EXCEPTION_THROW(graph.register_edge("dup", TestEdge{}), std::invalid_argument);
    }

    {
        SolverGraph graph{};

        REQUIRE_EXCEPTION_THROW(
            graph.register_node_ptr_base("null_node", std::shared_ptr<INode>{}),
            std::invalid_argument);
        REQUIRE_EQUAL(graph.get_node_names().size(), 0);

        REQUIRE_EXCEPTION_THROW(
            graph.register_edge_ptr_base("null_edge", std::shared_ptr<IEdge>{}),
            std::invalid_argument);
        REQUIRE_EQUAL(graph.get_edge_names().size(), 0);
    }
}
