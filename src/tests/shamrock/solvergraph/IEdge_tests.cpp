// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamrock/solvergraph/IEdge.hpp"
#include "shamtest/shamtest.hpp"

class TestEdge : public shamrock::solvergraph::IEdge {
    public:
    std::string _impl_get_dot_label() const override { return "dot"; }
    std::string _impl_get_tex_symbol() const override { return "tex"; }
    void free_alloc() override {}
};

TestStart(Unittest, "shamrock::solvergraph", iedge_tests, 1) {
    TestEdge edge{};

    REQUIRE_EQUAL(edge.get_label(), "dot");
    REQUIRE_EQUAL(edge.get_tex_symbol(), "tex");
}
