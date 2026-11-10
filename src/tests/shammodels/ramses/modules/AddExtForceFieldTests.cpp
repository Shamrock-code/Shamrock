// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/DistributedData.hpp"
#include "shammodels/ramses/modules/AddExtForceField.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamtest/shamtest.hpp"
#include <memory>
#include <vector>

using Tvec = f64_3;

NEW_TEST(Unittest, "shammodels/ramses/modules/AddExtForceField", 1) {

    // 2 values per block, 2 patches. The acceleration field has a third (ghost) block that must be
    // ignored.
    std::vector<Tvec> a_ext0 = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}};
    std::vector<Tvec> a_ext1 = {{-1, 0, 1}, {2, -2, 0}};
    std::vector<Tvec> a_field0
        = {{0.5, 0, 0}, {0, 0.5, 0}, {0, 0, 0.5}, {1, 1, 1}, {9, 9, 9}, {9, 9, 9}};
    std::vector<Tvec> a_field1 = {{1, 2, 3}, {4, 5, 6}, {9, 9, 9}, {9, 9, 9}};

    auto sizes = std::make_shared<shamrock::solvergraph::Indexes<u32>>("", "");
    sizes->indexes.add_obj(0, 2);
    sizes->indexes.add_obj(1, 1);

    shambase::DistributedData<u32> sizes_with_ghost;
    sizes_with_ghost.add_obj(0, 3);
    sizes_with_ghost.add_obj(1, 2);

    auto a_ext = std::make_shared<shamrock::solvergraph::Field<Tvec>>(2, "", "");
    a_ext->ensure_sizes(sizes->indexes);
    a_ext->get_buf(0).copy_from_stdvec(a_ext0);
    a_ext->get_buf(1).copy_from_stdvec(a_ext1);

    auto a_field = std::make_shared<shamrock::solvergraph::Field<Tvec>>(2, "", "");
    a_field->ensure_sizes(sizes_with_ghost);
    a_field->get_buf(0).copy_from_stdvec(a_field0);
    a_field->get_buf(1).copy_from_stdvec(a_field1);

    shammodels::basegodunov::modules::AddExtForceField<Tvec> node(2);
    node.set_edges(sizes, a_field, a_ext);
    node.evaluate();

    auto check = [&](u64 id, std::vector<Tvec> ext, const std::vector<Tvec> &field) {
        std::vector<Tvec> res = a_ext->get_buf(id).copy_to_stdvec();
        REQUIRE_EQUAL(res.size(), ext.size());
        for (u32 i = 0; i < ext.size(); i++) {
            Tvec expect = ext[i] + field[i];
            REQUIRE_EQUAL(res[i][0], expect[0]);
            REQUIRE_EQUAL(res[i][1], expect[1]);
            REQUIRE_EQUAL(res[i][2], expect[2]);
        }
    };

    check(0, a_ext0, a_field0);
    check(1, a_ext1, a_field1);

    // the acceleration field is not modified
    std::vector<Tvec> field0_after = a_field->get_buf(0).copy_to_stdvec();
    REQUIRE_EQUAL(field0_after.size(), a_field0.size());
    REQUIRE_EQUAL(field0_after[4][0], 9.);
}
