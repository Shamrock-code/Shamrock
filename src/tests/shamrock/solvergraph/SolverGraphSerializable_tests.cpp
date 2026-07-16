// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamcomm/logs.hpp"
#include "shamrock/solvergraph/IEdge.hpp"
#include "shamrock/solvergraph/INode.hpp"
#include "shamrock/solvergraph/JsonSerializable.hpp"
#include "shamrock/solvergraph/SolverGraphSerializable.hpp"
#include "shamtest/shamtest.hpp"
#include <string>

class TestNode : public shamrock::solvergraph::INode {
    public:
    void _impl_evaluate_internal() override {}
    std::string _impl_get_label() const override { return "node_label"; }
    std::string _impl_get_tex() const override { return "node_tex"; }
};

class TestEdge : public shamrock::solvergraph::IEdge {
    public:
    std::string _impl_get_dot_label() const override { return "dot"; }
    std::string _impl_get_tex_symbol() const override { return "tex"; }
    void free_alloc() override {}
};

class TestSerializableEdge : public shamrock::solvergraph::IEdge,
                             public shamrock::solvergraph::JsonSerializable {
    public:
    int value = 0;

    std::string _impl_get_dot_label() const override { return "serializable_dot"; }
    std::string _impl_get_tex_symbol() const override { return "serializable_tex"; }
    void free_alloc() override {}

    void to_json(nlohmann::json &j) const override { j["value"] = value; }
    std::string type_name() const override { return "TestSerializableEdge"; }

    static TestSerializableEdge from_json(const nlohmann::json &j) {
        TestSerializableEdge edge{};
        edge.value = j.at("value").get<int>();
        return edge;
    }
};

NEW_TEST(Unittest, "shamrock/solvergraph/SolverGraphSerializable", 1) {
    using namespace shamrock::solvergraph;

    JsonSerializable_registry::instance().register_type<TestSerializableEdge>(
        "TestSerializableEdge");

    {
        SolverGraphSerializable graph{};

        REQUIRE_EXCEPTION_THROW(
            graph.register_node("rejected_node", TestNode{}), std::invalid_argument);
        REQUIRE_EQUAL(graph.get_node_names().size(), 0);

        REQUIRE_EXCEPTION_THROW(
            graph.register_edge("rejected_edge", TestEdge{}), std::invalid_argument);
        REQUIRE_EQUAL(graph.get_edge_names().size(), 0);

        TestSerializableEdge edge{};
        edge.value = 7;

        graph.register_edge("ok_edge", std::move(edge));

        REQUIRE_EQUAL(graph.get_edge_names().size(), 1);
        REQUIRE_EQUAL(graph.get_edge_ptr<TestSerializableEdge>("ok_edge")->value, 7);
        REQUIRE_EQUAL(
            graph.get_edge_ptr<TestSerializableEdge>("ok_edge")->type_name(),
            "TestSerializableEdge");
    }

    {
        SolverGraphSerializable graph{};

        TestSerializableEdge edge_a{};
        edge_a.value = 11;
        TestSerializableEdge edge_b{};
        edge_b.value = 22;

        graph.register_edge("edge_a", std::move(edge_a));
        graph.register_edge("edge_b", std::move(edge_b));

        nlohmann::json j = graph;

        // shamcomm::logs::raw_ln("j =", j.dump(4));

        SolverGraphSerializable restored{};
        from_json(j, restored);

        REQUIRE_EQUAL_NAMED(
            "restored edge names",
            restored.get_edge_names(),
            (std::vector<std::string>{"edge_a", "edge_b"}));
        REQUIRE_EQUAL(restored.get_edge_ptr<TestSerializableEdge>("edge_a")->value, 11);
        REQUIRE_EQUAL(restored.get_edge_ptr<TestSerializableEdge>("edge_b")->value, 22);
    }
}
