// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambackends/math.hpp"
#include "shammodels/sph/modules/SetWhenMask.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamtest/shamtest.hpp"
#include <memory>
#include <vector>

namespace {

    /// Run SetWhenMask on two patches and check the result against a host reference
    template<class T>
    void test_set_when_mask(u32 nvar, T val_to_set, std::vector<T> f0, std::vector<T> f1) {

        std::vector<u32> mask0 = {0, 1, 0, 2};
        std::vector<u32> mask1 = {1, 0, 0};

        REQUIRE_EQUAL(f0.size(), mask0.size() * nvar);
        REQUIRE_EQUAL(f1.size(), mask1.size() * nvar);

        auto sizes = std::make_shared<shamrock::solvergraph::Indexes<u32>>("", "");
        sizes->indexes.add_obj(0, u32(mask0.size()));
        sizes->indexes.add_obj(1, u32(mask1.size()));

        auto mask = std::make_shared<shamrock::solvergraph::Field<u32>>(1, "", "");
        mask->ensure_sizes(sizes->indexes);
        mask->get_buf(0).copy_from_stdvec(mask0);
        mask->get_buf(1).copy_from_stdvec(mask1);

        auto field = std::make_shared<shamrock::solvergraph::Field<T>>(nvar, "", "");
        field->ensure_sizes(sizes->indexes);
        field->get_buf(0).copy_from_stdvec(f0);
        field->get_buf(1).copy_from_stdvec(f1);

        shammodels::sph::modules::SetWhenMask<T> node(val_to_set, nvar);
        node.set_edges(sizes, mask, field);
        node.evaluate();

        auto check = [&](u64 id, const std::vector<u32> &m, const std::vector<T> &f) {
            std::vector<T> res = field->get_buf(id).copy_to_stdvec();
            REQUIRE_EQUAL(res.size(), f.size());
            for (u32 i = 0; i < f.size(); i++) {
                T expect = (m[i / nvar] != 0) ? val_to_set : f[i];
                REQUIRE(sham::equals(res[i], expect));
            }
        };

        check(0, mask0, f0);
        check(1, mask1, f1);

        // the mask is not modified
        REQUIRE_EQUAL(mask->get_buf(0).copy_to_stdvec(), mask0);
        REQUIRE_EQUAL(mask->get_buf(1).copy_to_stdvec(), mask1);
    }

} // namespace

NEW_TEST(Unittest, "shammodels/sph/modules/SetWhenMask", 1) {

    // scalar field, one variable per object
    test_set_when_mask<f64>(1, 0., {1, 2, 3, 4}, {5, 6, 7});

    // scalar field, non-zero value to set
    test_set_when_mask<f64>(1, -1.5, {1, 2, 3, 4}, {5, 6, 7});

    // vector field (e.g. the acceleration)
    test_set_when_mask<f64_3>(
        1,
        f64_3{0, 0, 0},
        {f64_3{1, 2, 3}, f64_3{4, 5, 6}, f64_3{7, 8, 9}, f64_3{10, 11, 12}},
        {f64_3{-1, -2, -3}, f64_3{-4, -5, -6}, f64_3{-7, -8, -9}});

    // multi-variable field (e.g. one value per dust species): all the variables of a selected
    // object are set
    test_set_when_mask<f64>(
        3, 0., {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {13, 14, 15, 16, 17, 18, 19, 20, 21});
}
