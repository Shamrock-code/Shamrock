// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shamalgs/primitives/mock_vector.hpp"
#include "shamrock/solvergraph/NodeSumReduction.hpp"
#include "shamtest/shamtest.hpp"
#include "tests/shamrock/solvergraph/linalg_node_test_utils.hpp"
#include <bit>
#include <cmath>
#include <utility>
#include <vector>

namespace {

    /// Check NodeSumReduction against a host-side reference, and that it is run-to-run identical
    void test_sum_reduction(u32 nvar) {
        using T = f64;
        using namespace shamrock::solvergraph::test_utils;

        // larger patches than the elementwise tests, so that the device reduction does not fit in
        // a single work-group
        std::vector<std::pair<u64, u32>> obj_cnts = {{0, 37}, {3, 129}, {9, 2}};

        auto sizes = make_sizes(obj_cnts);
        auto field = make_field<T>(*sizes, nvar, 0x444, [](u64 seed, u32 len) {
            return shamalgs::primitives::mock_vector<T>(seed, len, -1., 1.);
        });

        long double expected = 0;
        long double abs_sum  = 0;
        for (auto [id_patch, obj_cnt] : obj_cnts) {
            for (T v : to_host(*field, id_patch)) {
                expected += v;
                abs_sum += std::abs(v);
            }
        }

        auto result = std::make_shared<shamrock::solvergraph::IDataEdge<T>>("sum", "S");

        shamrock::solvergraph::NodeSumReduction<T> node(nvar);
        node.set_edges(sizes, field, result);

        node.evaluate();
        T first = result->data;

        REQUIRE_FLOAT_EQUAL(first, T(expected), 1e-14 * T(abs_sum));

        // determinism: a second evaluation must give the bitwise identical result
        result->data = 0;
        node.evaluate();
        T second = result->data;

        REQUIRE_EQUAL(std::bit_cast<u64>(first), std::bit_cast<u64>(second));
    }

    /// Check that only the first `sizes * nvar` values are summed when the field holds extra
    /// objects at the end of each patch (e.g. ghost blocks, as with sizes without ghost zones)
    void test_sum_reduction_partial(u32 nvar) {
        using T = f64;
        using namespace shamrock::solvergraph::test_utils;

        std::vector<std::pair<u64, u32>> obj_cnts_all  = {{0, 37}, {3, 129}};
        std::vector<std::pair<u64, u32>> obj_cnts_summ = {{0, 30}, {3, 100}};

        auto sizes_all  = make_sizes(obj_cnts_all);
        auto sizes_summ = make_sizes(obj_cnts_summ);
        auto field      = make_field<T>(*sizes_all, nvar, 0x555, [](u64 seed, u32 len) {
            return shamalgs::primitives::mock_vector<T>(seed, len, -1., 1.);
        });

        long double expected = 0;
        long double abs_sum  = 0;
        for (auto [id_patch, obj_cnt] : obj_cnts_summ) {
            std::vector<T> vals = to_host(*field, id_patch);
            for (u32 i = 0; i < obj_cnt * nvar; i++) {
                expected += vals[i];
                abs_sum += std::abs(vals[i]);
            }
        }

        auto result = std::make_shared<shamrock::solvergraph::IDataEdge<T>>("sum", "S");

        shamrock::solvergraph::NodeSumReduction<T> node(nvar);
        node.set_edges(sizes_summ, field, result);
        node.evaluate();

        REQUIRE_FLOAT_EQUAL(result->data, T(expected), 1e-14 * T(abs_sum));
    }

} // namespace

NEW_TEST(Unittest, "shamrock/solvergraph/NodeSumReduction", 1) {
    for (u32 nvar : shamrock::solvergraph::test_utils::nvars) {
        test_sum_reduction(nvar);
        test_sum_reduction_partial(nvar);
    }
}
