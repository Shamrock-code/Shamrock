// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambase/DistributedData.hpp"
#include "shammodels/ramses/modules/ComputeNextRho.hpp"
#include "shamrock/patch/PatchDataField.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamrock/solvergraph/ScalarEdge.hpp"
#include "shamtest/shamtest.hpp"
#include <memory>
#include <vector>

using Tvec  = f64_3;
using Tscal = f64;

NEW_TEST(Unittest, "shammodels/ramses/modules/ComputeNextRho", 1) {

    // 2 values per block, 2 patches: patch 0 has 2 blocks + 1 ghost block, patch 1 has 1 block +
    // 2 ghost blocks
    std::vector<Tscal> rho0    = {1, 2, 3, 4, 5, 6};
    std::vector<Tscal> rho1    = {0.5, 0.25, 7, 8, 9, 10};
    std::vector<Tscal> dt_rho0 = {0.5, -1, 2, 0.125};
    std::vector<Tscal> dt_rho1 = {-0.5, 4};

    Tscal dt_val = 0.375;

    auto sizes = std::make_shared<shamrock::solvergraph::Indexes<u32>>("", "");
    sizes->indexes.add_obj(0, 2);
    sizes->indexes.add_obj(1, 1);

    auto sizes_with_ghost = std::make_shared<shamrock::solvergraph::Indexes<u32>>("", "");
    sizes_with_ghost->indexes.add_obj(0, 3);
    sizes_with_ghost->indexes.add_obj(1, 3);

    auto make_field = [&](const shambase::DistributedData<u32> &field_sizes,
                          const std::vector<Tscal> &v0,
                          const std::vector<Tscal> &v1) {
        auto f = std::make_shared<shamrock::solvergraph::Field<Tscal>>(2, "", "");
        f->ensure_sizes(field_sizes);
        f->get_buf(0).copy_from_stdvec(v0);
        f->get_buf(1).copy_from_stdvec(v1);
        return f;
    };

    auto rho    = make_field(sizes_with_ghost->indexes, rho0, rho1);
    auto dt_rho = make_field(sizes->indexes, dt_rho0, dt_rho1);

    auto dt   = std::make_shared<shamrock::solvergraph::ScalarEdge<Tscal>>("", "");
    dt->value = dt_val;

    auto rho_next = std::make_shared<shamrock::solvergraph::Field<Tscal>>(2, "", "");

    shammodels::basegodunov::modules::ComputeNextRho<Tvec> node(2);
    node.set_edges(dt, sizes, sizes_with_ghost, rho, dt_rho, rho_next);
    node.evaluate();

    auto check = [&](u64 id, const std::vector<Tscal> &r, const std::vector<Tscal> &dr) {
        std::vector<Tscal> res = rho_next->get_buf(id).copy_to_stdvec();
        REQUIRE_EQUAL(res.size(), r.size());
        for (u32 i = 0; i < r.size(); i++) {
            // patch cells are advanced, ghost cells keep rho^n
            Tscal expect = (i < dr.size()) ? r[i] + dt_val * dr[i] : r[i];
            REQUIRE_EQUAL(res[i], expect);
        }
    };

    check(0, rho0, dt_rho0);
    check(1, rho1, dt_rho1);

    // rho^n is not modified
    std::vector<Tscal> rho0_after = rho->get_buf(0).copy_to_stdvec();
    for (u32 i = 0; i < rho0.size(); i++) {
        REQUIRE_EQUAL(rho0_after[i], rho0[i]);
    }
}
