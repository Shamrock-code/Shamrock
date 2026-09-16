// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shammodels/sph/modules/BallabioTsLimiter.hpp"
#include "shamrock/solvergraph/Field.hpp"
#include "shamrock/solvergraph/Indexes.hpp"
#include "shamtest/shamtest.hpp"
#include <memory>
#include <vector>

NEW_TEST(Unittest, "shammodels/sph/modules/BallabioTsLimiter", 1) {
    using Tvec  = f64_3;
    using Tscal = f64;
    using namespace shamrock;
    using namespace shammodels::sph::modules;

    u32 ndust = 2;
    u32 N     = 3;

    auto part_counts = std::make_shared<solvergraph::Indexes<u32>>("", "");
    part_counts->indexes.add_obj(0, u32{N});

    auto hpart = std::make_shared<solvergraph::Field<Tscal>>(1, "hpart", "h");
    auto cs    = std::make_shared<solvergraph::Field<Tscal>>(1, "cs", "c_s");
    auto t_j   = std::make_shared<solvergraph::Field<Tscal>>(ndust, "Ts_j", "Ts_j");

    hpart->ensure_sizes(part_counts->indexes);
    cs->ensure_sizes(part_counts->indexes);
    t_j->ensure_sizes(part_counts->indexes);

    hpart->get_buf(0).copy_from_stdvec({2.0, 4.0, 6.0});
    cs->get_buf(0).copy_from_stdvec({1.0, 2.0, 3.0});
    t_j->get_buf(0).copy_from_stdvec({5.0, 1.0, 3.0, 1.5, 1.0, 4.0});

    BallabioTsLimiter<Tvec> node(ndust);
    node.set_edges(part_counts, hpart, cs, t_j);
    node.evaluate();

    std::vector<Tscal> expected = {2.0, 1.0, 2.0, 1.5, 1.0, 2.0};
    REQUIRE_EQUAL(t_j->get_buf(0).copy_to_stdvec(), expected);
}
