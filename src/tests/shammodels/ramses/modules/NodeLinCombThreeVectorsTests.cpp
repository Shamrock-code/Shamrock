// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambackends/math.hpp"
#include "shammodels/ramses/modules/NodeLinCombThreeVectors.hpp"
#include "shamtest/shamtest.hpp"
#include "tests/shammodels/ramses/modules/linalg_node_test_utils.hpp"
#include <limits>
#include <vector>

namespace {

    /// Check NodeLinCombThreeVectors (z <- alpha_0 * z + alpha_1 * x + alpha_2 * y) against a
    /// host-side reference
    template<class T>
    void test_lin_comb_three_vectors(u32 block_size, shambase::VecComponent<T> alpha_0_val) {
        using Tscal = shambase::VecComponent<T>;
        using namespace shammodels::ramses::test_utils;

        auto sizes   = make_sizes();
        auto x       = make_dyadic_field<T>(*sizes, block_size, 0x111);
        auto y       = make_dyadic_field<T>(*sizes, block_size, 0x222);
        auto z       = make_dyadic_field<T>(*sizes, block_size, 0x333);
        auto alpha_0 = make_scalar<Tscal>(alpha_0_val);
        auto alpha_1 = make_scalar<Tscal>(-0.75);
        auto alpha_2 = make_scalar<Tscal>(2.5);

        std::vector<std::vector<T>> expected;
        for (auto [id_patch, block_count] : patch_block_counts) {
            std::vector<T> xh = to_host(*x, id_patch);
            std::vector<T> yh = to_host(*y, id_patch);
            std::vector<T> zh = to_host(*z, id_patch);
            for (u32 i = 0; i < zh.size(); i++) {
                T xy_part = alpha_1->value * xh[i] + alpha_2->value * yh[i];
                zh[i]     = (alpha_0_val != 0) ? alpha_0->value * zh[i] + xy_part : xy_part;
            }
            expected.push_back(zh);
        }

        if (alpha_0_val == 0) {
            // z must not be read when alpha_0 == 0: poison it to check that
            for (auto [id_patch, block_count] : patch_block_counts) {
                std::vector<T> nans(
                    block_count * block_size, T(std::numeric_limits<Tscal>::quiet_NaN()));
                z->get_buf(id_patch).copy_from_stdvec(nans);
            }
        }

        shammodels::basegodunov::modules::NodeLinCombThreeVectors<T> node(block_size);
        node.set_edges(sizes, x, y, alpha_0, alpha_1, alpha_2, z);
        node.evaluate();

        for (u32 i = 0; i < patch_block_counts.size(); i++) {
            REQUIRE_EQUAL_CUSTOM_COMP(
                to_host(*z, patch_block_counts[i].first), expected[i], sham::equals);
        }
    }

    /// Check that the output field is allocated by the node when it is empty (alpha_0 == 0)
    void test_lin_comb_three_vectors_alloc(u32 block_size) {
        using T = f64;
        using namespace shammodels::ramses::test_utils;

        auto sizes   = make_sizes();
        auto x       = make_dyadic_field<T>(*sizes, block_size, 0x111);
        auto y       = make_dyadic_field<T>(*sizes, block_size, 0x222);
        auto z       = std::make_shared<shamrock::solvergraph::Field<T>>(block_size, "z", "z");
        auto alpha_0 = make_scalar<T>(0);
        auto alpha_1 = make_scalar<T>(0.5);
        auto alpha_2 = make_scalar<T>(-4);

        shammodels::basegodunov::modules::NodeLinCombThreeVectors<T> node(block_size);
        node.set_edges(sizes, x, y, alpha_0, alpha_1, alpha_2, z);
        node.evaluate();

        for (auto [id_patch, block_count] : patch_block_counts) {
            std::vector<T> xh = to_host(*x, id_patch);
            std::vector<T> yh = to_host(*y, id_patch);
            std::vector<T> expected(xh.size());
            for (u32 i = 0; i < xh.size(); i++) {
                expected[i] = alpha_1->value * xh[i] + alpha_2->value * yh[i];
            }
            REQUIRE_EQUAL_CUSTOM_COMP(to_host(*z, id_patch), expected, sham::equals);
        }
    }

} // namespace

NEW_TEST(Unittest, "shammodels/ramses/modules/NodeLinCombThreeVectors", 1) {
    for (u32 block_size : shammodels::ramses::test_utils::block_sizes) {
        test_lin_comb_three_vectors<f64>(block_size, 1.5);
        test_lin_comb_three_vectors<f64>(block_size, 0);
        test_lin_comb_three_vectors_alloc(block_size);
    }
}
