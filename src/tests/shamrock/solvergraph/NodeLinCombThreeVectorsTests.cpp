// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambackends/math.hpp"
#include "shamrock/solvergraph/NodeLinCombThreeVectors.hpp"
#include "shamtest/shamtest.hpp"
#include "tests/shamrock/solvergraph/linalg_node_test_utils.hpp"
#include <limits>
#include <vector>

namespace {

    /// Check NodeLinCombThreeVectors (z <- alpha_0 * z + alpha_1 * x + alpha_2 * y) against a
    /// host-side reference
    template<class T>
    void test_lin_comb_three_vectors(u32 nvar, shambase::VecComponent<T> alpha_0_val) {
        using Tscal = shambase::VecComponent<T>;
        using namespace shamrock::solvergraph::test_utils;

        auto sizes   = make_sizes();
        auto x       = make_dyadic_field<T>(*sizes, nvar, 0x111);
        auto y       = make_dyadic_field<T>(*sizes, nvar, 0x222);
        auto z       = make_dyadic_field<T>(*sizes, nvar, 0x333);
        auto alpha_0 = make_scalar<Tscal>(alpha_0_val);
        auto alpha_1 = make_scalar<Tscal>(-0.75);
        auto alpha_2 = make_scalar<Tscal>(2.5);

        std::vector<std::vector<T>> expected;
        for (auto [id_patch, obj_cnt] : patch_obj_cnts) {
            std::vector<T> xh = to_host(*x, id_patch);
            std::vector<T> yh = to_host(*y, id_patch);
            std::vector<T> zh = to_host(*z, id_patch);
            for (u32 i = 0; i < zh.size(); i++) {
                zh[i] = (alpha_0_val != 0)
                            ? alpha_0->data * zh[i] + alpha_1->data * xh[i] + alpha_2->data * yh[i]
                            : alpha_1->data * xh[i] + alpha_2->data * yh[i];
            }
            expected.push_back(zh);
        }

        if (alpha_0_val == 0) {
            // z must not be read when alpha_0 == 0: poison it to check that
            for (auto [id_patch, obj_cnt] : patch_obj_cnts) {
                std::vector<T> nans(obj_cnt * nvar, T(std::numeric_limits<Tscal>::quiet_NaN()));
                z->get_buf(id_patch).copy_from_stdvec(nans);
            }
        }

        shamrock::solvergraph::NodeLinCombThreeVectors<T> node(nvar);
        node.set_edges(sizes, x, y, alpha_0, alpha_1, alpha_2, z);
        node.evaluate();

        for (u32 i = 0; i < patch_obj_cnts.size(); i++) {
            REQUIRE_EQUAL_CUSTOM_COMP(
                to_host(*z, patch_obj_cnts[i].first), expected[i], sham::equals);
        }
    }

    /// Check that the output field is allocated by the node when it is empty (alpha_0 == 0)
    void test_lin_comb_three_vectors_alloc(u32 nvar) {
        using T = f64;
        using namespace shamrock::solvergraph::test_utils;

        auto sizes   = make_sizes();
        auto x       = make_dyadic_field<T>(*sizes, nvar, 0x111);
        auto y       = make_dyadic_field<T>(*sizes, nvar, 0x222);
        auto z       = std::make_shared<shamrock::solvergraph::Field<T>>(nvar, "z", "z");
        auto alpha_0 = make_scalar<T>(0);
        auto alpha_1 = make_scalar<T>(0.5);
        auto alpha_2 = make_scalar<T>(-4);

        shamrock::solvergraph::NodeLinCombThreeVectors<T> node(nvar);
        node.set_edges(sizes, x, y, alpha_0, alpha_1, alpha_2, z);
        node.evaluate();

        for (auto [id_patch, obj_cnt] : patch_obj_cnts) {
            std::vector<T> xh = to_host(*x, id_patch);
            std::vector<T> yh = to_host(*y, id_patch);
            std::vector<T> expected(xh.size());
            for (u32 i = 0; i < xh.size(); i++) {
                expected[i] = alpha_1->data * xh[i] + alpha_2->data * yh[i];
            }
            REQUIRE_EQUAL_CUSTOM_COMP(to_host(*z, id_patch), expected, sham::equals);
        }
    }

} // namespace

NEW_TEST(Unittest, "shamrock/solvergraph/NodeLinCombThreeVectors", 1) {
    for (u32 nvar : shamrock::solvergraph::test_utils::nvars) {
        test_lin_comb_three_vectors<f64>(nvar, 1.5);
        test_lin_comb_three_vectors<f64>(nvar, 0);
        test_lin_comb_three_vectors_alloc(nvar);
    }
}
