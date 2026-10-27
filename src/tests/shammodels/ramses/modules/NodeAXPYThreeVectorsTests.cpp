// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambackends/math.hpp"
#include "shammodels/ramses/modules/NodeAXPYThreeVectors.hpp"
#include "shamtest/shamtest.hpp"
#include "tests/shammodels/ramses/modules/linalg_node_test_utils.hpp"
#include <vector>

namespace {

    /// Check NodeAXPYThreeVectors (z <- z + alpha * x + beta * y) against a host-side reference
    template<class T>
    void test_axpy_three_vectors(u32 block_size) {
        using Tscal = shambase::VecComponent<T>;
        using namespace shammodels::ramses::test_utils;

        auto sizes = make_sizes();
        auto x     = make_dyadic_field<T>(*sizes, block_size, 0x111);
        auto y     = make_dyadic_field<T>(*sizes, block_size, 0x222);
        auto z     = make_dyadic_field<T>(*sizes, block_size, 0x333);
        auto alpha = make_scalar<Tscal>(-0.75);
        auto beta  = make_scalar<Tscal>(2.5);

        std::vector<std::vector<T>> expected;
        for (auto [id_patch, block_count] : patch_block_counts) {
            std::vector<T> xh = to_host(*x, id_patch);
            std::vector<T> yh = to_host(*y, id_patch);
            std::vector<T> zh = to_host(*z, id_patch);
            for (u32 i = 0; i < zh.size(); i++) {
                zh[i] = zh[i] + alpha->value * xh[i] + beta->value * yh[i];
            }
            expected.push_back(zh);
        }

        shammodels::basegodunov::modules::NodeAXPYThreeVectors<T> node(block_size);
        node.set_edges(sizes, x, y, alpha, beta, z);
        node.evaluate();

        for (u32 i = 0; i < patch_block_counts.size(); i++) {
            REQUIRE_EQUAL_CUSTOM_COMP(
                to_host(*z, patch_block_counts[i].first), expected[i], sham::equals);
        }
    }

} // namespace

NEW_TEST(Unittest, "shammodels/ramses/modules/NodeAXPYThreeVectors", 1) {
    for (u32 block_size : shammodels::ramses::test_utils::block_sizes) {
        test_axpy_three_vectors<f64>(block_size);
    }
}
