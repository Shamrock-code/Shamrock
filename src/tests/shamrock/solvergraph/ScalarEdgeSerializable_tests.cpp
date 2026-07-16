// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamrock/solvergraph/JsonSerializable.hpp"
#include "shamrock/solvergraph/ScalarEdgeSerializable.hpp"
#include "shamtest/shamtest.hpp"
#include <memory>
#include <string>

NEW_TEST(Unittest, "shamrock/solvergraph/ScalarEdgeSerializable", 1) {
    using Edge = shamrock::solvergraph::ScalarEdgeSerializable<f64>;

    Edge original("my_label", "my_tex");
    original.value = 3.14159;

    nlohmann::json j;
    original.to_json(j);

    {
        Edge restored = Edge::from_json(j);

        REQUIRE_EQUAL(restored.value, original.value);
        REQUIRE_EQUAL(restored.get_label(), "my_label");
        REQUIRE_EQUAL(restored.get_raw_tex_symbol(), "my_tex");
        REQUIRE_EQUAL(restored.type_name(), "ScalarEdgeSerializable<f64>");
    }

    {
        auto ptr     = shamrock::solvergraph::JsonSerializable::from_json(j);
        auto *as_edge = dynamic_cast<Edge *>(ptr.get());

        REQUIRE(as_edge != nullptr);
        REQUIRE_EQUAL(as_edge->value, original.value);
        REQUIRE_EQUAL(as_edge->get_label(), "my_label");
        REQUIRE_EQUAL(as_edge->get_raw_tex_symbol(), "my_tex");
        REQUIRE_EQUAL(as_edge->type_name(), "ScalarEdgeSerializable<f64>");
    }
}
