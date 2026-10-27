// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambackends/math.hpp"
#include "shammodels/ramses/modules/NodeOverwrite.hpp"
#include "shamtest/shamtest.hpp"
#include "tests/shammodels/ramses/modules/linalg_node_test_utils.hpp"
#include <memory>
#include <vector>

namespace {

    /// Check NodeOverwrite (y <- x), on an already allocated and on an empty output field
    template<class T>
    void test_overwrite(u32 block_size) {
        using namespace shammodels::ramses::test_utils;

        auto sizes       = make_sizes();
        auto x           = make_dyadic_field<T>(*sizes, block_size, 0x111);
        auto y_allocated = make_dyadic_field<T>(*sizes, block_size, 0x222);
        auto y_empty     = std::make_shared<shamrock::solvergraph::Field<T>>(block_size, "y", "y");

        for (auto &y : {y_allocated, y_empty}) {
            shammodels::basegodunov::modules::NodeOverwrite<T> node(block_size);
            node.set_edges(sizes, x, y);
            node.evaluate();

            for (auto [id_patch, block_count] : patch_block_counts) {
                REQUIRE_EQUAL_CUSTOM_COMP(
                    to_host(*y, id_patch), to_host(*x, id_patch), sham::equals);
            }
        }
    }

} // namespace

NEW_TEST(Unittest, "shammodels/ramses/modules/NodeOverwrite", 1) {
    for (u32 block_size : shammodels::ramses::test_utils::block_sizes) {
        test_overwrite<f64>(block_size);
    }
}
