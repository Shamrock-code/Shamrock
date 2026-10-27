// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambackends/math.hpp"
#include "shamrock/solvergraph/NodeAYPX.hpp"
#include "shamtest/shamtest.hpp"
#include "tests/shamrock/solvergraph/linalg_node_test_utils.hpp"
#include <vector>

namespace {

    /// Check NodeAYPX (y <- alpha * y + x) against a host-side reference
    template<class T>
    void test_aypx(u32 nvar) {
        using Tscal = shambase::VecComponent<T>;
        using namespace shamrock::solvergraph::test_utils;

        auto sizes = make_sizes();
        auto x     = make_dyadic_field<T>(*sizes, nvar, 0x111);
        auto y     = make_dyadic_field<T>(*sizes, nvar, 0x222);
        auto alpha = make_scalar<Tscal>(-0.75);

        std::vector<std::vector<T>> expected;
        for (auto [id_patch, obj_cnt] : patch_obj_cnts) {
            std::vector<T> xh = to_host(*x, id_patch);
            std::vector<T> yh = to_host(*y, id_patch);
            for (u32 i = 0; i < yh.size(); i++) {
                yh[i] = alpha->value * yh[i] + xh[i];
            }
            expected.push_back(yh);
        }

        shamrock::solvergraph::NodeAYPX<T> node(nvar);
        node.set_edges(sizes, x, alpha, y);
        node.evaluate();

        for (u32 i = 0; i < patch_obj_cnts.size(); i++) {
            REQUIRE_EQUAL_CUSTOM_COMP(
                to_host(*y, patch_obj_cnts[i].first), expected[i], sham::equals);
        }
    }

} // namespace

NEW_TEST(Unittest, "shamrock/solvergraph/NodeAYPX", 1) {
    for (u32 nvar : shamrock::solvergraph::test_utils::nvars) {
        test_aypx<f64>(nvar);
        test_aypx<f64_3>(nvar);
    }
}
