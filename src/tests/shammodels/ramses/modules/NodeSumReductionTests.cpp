// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamalgs/primitives/mock_vector.hpp"
#include "shammodels/ramses/modules/NodeSumReduction.hpp"
#include "shamtest/shamtest.hpp"
#include "tests/shammodels/ramses/modules/linalg_node_test_utils.hpp"
#include <bit>
#include <cmath>
#include <utility>
#include <vector>

namespace {

    /// Check NodeSumReduction against a host-side reference, and that it is run-to-run identical
    void test_sum_reduction(u32 block_size) {
        using T = f64;
        using namespace shammodels::ramses::test_utils;

        // larger patches than the elementwise tests, so that the device reduction does not fit in
        // a single work-group
        std::vector<std::pair<u64, u32>> block_counts = {{0, 37}, {3, 129}, {9, 2}};

        auto sizes = make_sizes(block_counts);
        auto field = make_field<T>(*sizes, block_size, 0x444, [](u64 seed, u32 len) {
            return shamalgs::primitives::mock_vector<T>(seed, len, -1., 1.);
        });

        long double expected = 0;
        long double abs_sum  = 0;
        for (auto [id_patch, block_count] : block_counts) {
            for (T v : to_host(*field, id_patch)) {
                expected += v;
                abs_sum += std::abs(v);
            }
        }

        auto result = std::make_shared<shamrock::solvergraph::ScalarEdge<T>>("sum", "S");

        shammodels::basegodunov::modules::NodeSumReduction<T> node(block_size);
        node.set_edges(sizes, field, result);

        node.evaluate();
        T first = result->value;

        REQUIRE_FLOAT_EQUAL(first, T(expected), 1e-14 * T(abs_sum));

        // determinism: a second evaluation must give the bitwise identical result
        result->value = 0;
        node.evaluate();
        T second = result->value;

        REQUIRE_EQUAL(std::bit_cast<u64>(first), std::bit_cast<u64>(second));
    }

} // namespace

NEW_TEST(Unittest, "shammodels/ramses/modules/NodeSumReduction", 1) {
    for (u32 block_size : shammodels::ramses::test_utils::block_sizes) {
        test_sum_reduction(block_size);
    }
}
