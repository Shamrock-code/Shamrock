// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambackends/math.hpp"
#include "shammodels/ramses/modules/NodeAXPY.hpp"
#include "shamtest/shamtest.hpp"
#include "tests/shammodels/ramses/modules/linalg_node_test_utils.hpp"
#include <vector>

namespace {

    /// Check NodeAXPY (y <- y + alpha * x) against a host-side reference
    template<class T>
    void test_axpy(u32 block_size) {
        using Tscal = shambase::VecComponent<T>;
        using namespace shammodels::ramses::test_utils;

        auto sizes = make_sizes();
        auto x     = make_dyadic_field<T>(*sizes, block_size, 0x111);
        auto y     = make_dyadic_field<T>(*sizes, block_size, 0x222);
        auto alpha = make_scalar<Tscal>(-0.75);

        std::vector<std::vector<T>> expected;
        for (auto [id_patch, block_count] : patch_block_counts) {
            std::vector<T> xh = to_host(*x, id_patch);
            std::vector<T> yh = to_host(*y, id_patch);
            for (u32 i = 0; i < yh.size(); i++) {
                yh[i] = yh[i] + alpha->value * xh[i];
            }
            expected.push_back(yh);
        }

        shammodels::basegodunov::modules::NodeAXPY<T> node(block_size);
        node.set_edges(sizes, x, alpha, y);
        node.evaluate();

        for (u32 i = 0; i < patch_block_counts.size(); i++) {
            REQUIRE_EQUAL_CUSTOM_COMP(
                to_host(*y, patch_block_counts[i].first), expected[i], sham::equals);
        }
    }

} // namespace

NEW_TEST(Unittest, "shammodels/ramses/modules/NodeAXPY", 1) {
    for (u32 block_size : shammodels::ramses::test_utils::block_sizes) {
        test_axpy<f64>(block_size);
        test_axpy<f64_3>(block_size);
    }
}
