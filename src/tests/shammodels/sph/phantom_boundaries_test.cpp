// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#include "shambackends/vec.hpp"
#include "shammodels/sph/config/BCConfig.hpp"
#include "shammodels/sph/io/Phantom2Shamrock.hpp"
#include "shammodels/sph/io/PhantomDump.hpp"
#include "shamtest/shamtest.hpp"
#include <tuple>
#include <variant>

NEW_TEST(Unittest, "shammodels/sph/phantom_boundaries_periodic_non_cubic", 1) {
    using namespace shammodels::sph;

    // non-cubic and off-centre so that every bound differs from the others
    f64_3 bmin = {-1., -2., -3.};
    f64_3 bmax = {4., 5., 6.};

    BCConfig<f64_3> cfg;
    cfg.set_periodic();

    PhantomDump dump{};
    write_shamrock_boundaries_in_phantom_dump<f64_3>(
        cfg, std::tuple<f64_3, f64_3>{bmin, bmax}, dump, false);

    REQUIRE_EQUAL(dump.read_header_float<f64>("xmin"), bmin.x());
    REQUIRE_EQUAL(dump.read_header_float<f64>("xmax"), bmax.x());
    REQUIRE_EQUAL(dump.read_header_float<f64>("ymin"), bmin.y());
    REQUIRE_EQUAL(dump.read_header_float<f64>("ymax"), bmax.y());
    REQUIRE_EQUAL(dump.read_header_float<f64>("zmin"), bmin.z());
    REQUIRE_EQUAL(dump.read_header_float<f64>("zmax"), bmax.z());

    // reading the dump back must yield periodic boundaries
    BCConfig<f64_3> cfg_read = get_shamrock_boundary_config<f64_3>(dump);
    REQUIRE(std::holds_alternative<BCConfig<f64_3>::Periodic>(cfg_read.config));
}

NEW_TEST(Unittest, "shammodels/sph/phantom_boundaries_free", 1) {
    using namespace shammodels::sph;

    BCConfig<f64_3> cfg;
    cfg.set_free();

    PhantomDump dump{};
    write_shamrock_boundaries_in_phantom_dump<f64_3>(
        cfg, std::tuple<f64_3, f64_3>{{-1., -2., -3.}, {4., 5., 6.}}, dump, false);

    // phantom only stores the box in the header for periodic boundaries
    REQUIRE(!dump.has_header_entry("xmin"));

    BCConfig<f64_3> cfg_read = get_shamrock_boundary_config<f64_3>(dump);
    REQUIRE(std::holds_alternative<BCConfig<f64_3>::Free>(cfg_read.config));
}
