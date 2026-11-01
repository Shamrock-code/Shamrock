// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/DistributedData.hpp"
#include "shambase/logs/loglevels.hpp"
#include "shamalgs/details/random/random.hpp"
#include "shamcomm/worldInfo.hpp"
#include "shammodels/ramses/modules/ResidualDot.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamsolvergraph/edge/IDataEdge.hpp"
#include "shamtest/shamtest.hpp"
#include <memory>
#include <vector>

using Tvec  = f64_3;
using Tscal = f64;

NEW_TEST(Unittest, "shammodels/ramses/modules/ResidualDot", 2) {

    std::vector<Tvec> ref_vals0 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
    std::vector<Tvec> ref_vals1 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
    std::vector<Tvec> ref_vals2 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
    std::vector<Tvec> ref_vals3 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};

    Tscal expect_result = 0;

    for (auto &val : ref_vals0) {
        expect_result += val[0] * val[0] + val[1] * val[1] + val[2] * val[2];
    }
    for (auto &val : ref_vals1) {
        expect_result += val[0] * val[0] + val[1] * val[1] + val[2] * val[2];
    }
    for (auto &val : ref_vals2) {
        expect_result += val[0] * val[0] + val[1] * val[1] + val[2] * val[2];
    }
    for (auto &val : ref_vals3) {
        expect_result += val[0] * val[0] + val[1] * val[1] + val[2] * val[2];
    }

    std::shared_ptr<shamrock::solvergraph::Field<Tvec>> field_test
        = std::make_shared<shamrock::solvergraph::Field<Tvec>>(1, "", "");

    auto sizes = std::make_shared<shamrock::solvergraph::Indexes<u32>>("", "");

    if (shamcomm::world_rank() == 0) {
        shambase::DistributedData<u32> &size_rank0 = sizes->indexes;
        size_rank0.add_obj(0, ref_vals0.size());
        size_rank0.add_obj(1, ref_vals1.size());
        field_test->ensure_sizes(size_rank0);
        field_test->get_buf(0).copy_from_stdvec(ref_vals0);
        field_test->get_buf(1).copy_from_stdvec(ref_vals1);
    }

    if (shamcomm::world_rank() == 1) {
        shambase::DistributedData<u32> &size_rank1 = sizes->indexes;
        size_rank1.add_obj(2, ref_vals2.size());
        size_rank1.add_obj(3, ref_vals3.size());
        field_test->ensure_sizes(size_rank1);
        field_test->get_buf(2).copy_from_stdvec(ref_vals2);
        field_test->get_buf(3).copy_from_stdvec(ref_vals3);
    }

    auto result = shamrock::solvergraph::IDataEdge<Tscal>::make_shared("", "");

    shammodels::basegodunov::modules::ResidualDot<Tvec> node(1);
    node.set_edges(sizes, field_test, result);
    node.evaluate();

    REQUIRE_EQUAL(expect_result, result->data);
}

NEW_TEST(Unittest, "shammodels/ramses/modules/ResidualDot:exclude_ghosts", 1) {

    // 3 objects of 2 values per patch, only the first 2 objects (the patch blocks) are summed, the
    // last one (a ghost block) is excluded
    std::vector<Tscal> vals0 = {1, 2, 3, 4, 100, 100};
    std::vector<Tscal> vals1 = {5, 6, 7, 8, 100, 100};

    Tscal expect_result = 0;
    for (u32 i = 0; i < 4; i++) {
        expect_result += vals0[i] * vals0[i] + vals1[i] * vals1[i];
    }

    auto field_test = std::make_shared<shamrock::solvergraph::Field<Tscal>>(2, "", "");

    shambase::DistributedData<u32> field_sizes;
    field_sizes.add_obj(0, 3);
    field_sizes.add_obj(1, 3);
    field_test->ensure_sizes(field_sizes);
    field_test->get_buf(0).copy_from_stdvec(vals0);
    field_test->get_buf(1).copy_from_stdvec(vals1);

    auto sizes = std::make_shared<shamrock::solvergraph::Indexes<u32>>("", "");
    sizes->indexes.add_obj(0, 2);
    sizes->indexes.add_obj(1, 2);

    auto result = shamrock::solvergraph::IDataEdge<Tscal>::make_shared("", "");

    shammodels::basegodunov::modules::ResidualDot<Tscal> node(2);
    node.set_edges(sizes, field_test, result);
    node.evaluate();

    REQUIRE_EQUAL(expect_result, result->data);
}
