// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file constexpr_Tests.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Compile-time checks that UnitSystem and Constants can be built in constant expressions
 *
 */

#include "shamtest/shamtest.hpp"
#include "shamunits/Constants.hpp"
#include "shamunits/UnitSystem.hpp"
#include "shamunits/unit_table.hpp"

namespace {
    using namespace shamunits;

    constexpr UnitSystem<double> si{};
    static_assert(si.s == 1.0 && si.m_inv == 1.0);

    constexpr UnitSystem<double> astro{3.15576e13, 1.495978707e11, 1.98847e30};
    static_assert(astro.s_inv == 3.15576e13);

    constexpr Constants<double> astro_cte{astro};
    static_assert(astro_cte.units.kg_inv == 1.98847e30);

    // the Unit-value getter is const + constexpr, so it works at compile time
    static_assert(si.get<units::astronomical_unit>() == 149597870700.0);
} // namespace

NEW_TEST(Unittest, "shamunits/constexpr", 1) {
    // the asserts above are the test; this keeps it visible in the test list
    REQUIRE_EQUAL(si.s, 1.0);
}
