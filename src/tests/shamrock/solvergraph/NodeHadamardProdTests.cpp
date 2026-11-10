// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambackends/math.hpp"
#include "shamrock/solvergraph/NodeHadamardProd.hpp"
#include "shamtest/shamtest.hpp"
#include "tests/shamrock/solvergraph/linalg_node_test_utils.hpp"
#include <memory>
#include <vector>

namespace {

    /// Check NodeHadamardProd (out <- in1 * in2) against a host-side reference
    template<class T>
    void test_hadamard_prod(u32 nvar) {
        using namespace shamrock::solvergraph::test_utils;

        auto sizes = make_sizes();
        auto in1   = make_dyadic_field<T>(*sizes, nvar, 0x111);
        auto in2   = make_dyadic_field<T>(*sizes, nvar, 0x222);
        // empty output: the node must allocate it
        auto out = std::make_shared<shamrock::solvergraph::Field<T>>(nvar, "out", "out");

        shamrock::solvergraph::NodeHadamardProd<T> node(nvar);
        node.set_edges(sizes, in1, in2, out);
        node.evaluate();

        for (auto [id_patch, obj_cnt] : patch_obj_cnts) {
            std::vector<T> in1h = to_host(*in1, id_patch);
            std::vector<T> in2h = to_host(*in2, id_patch);
            std::vector<T> expected(in1h.size());
            for (u32 i = 0; i < in1h.size(); i++) {
                expected[i] = in1h[i] * in2h[i];
            }
            REQUIRE_EQUAL_CUSTOM_COMP(to_host(*out, id_patch), expected, sham::equals);
        }
    }

} // namespace

NEW_TEST(Unittest, "shamrock/solvergraph/NodeHadamardProd", 1) {
    for (u32 nvar : shamrock::solvergraph::test_utils::nvars) {
        test_hadamard_prod<f64>(nvar);
    }
}
