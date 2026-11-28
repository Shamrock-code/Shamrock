// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file unit_tables_Tests.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Tests of the unit and prefix tables (shamunits/unit_table.hpp, prefix_table.hpp)
 *
 */

#include "shamtest/shamtest.hpp"
#include "shamunits/prefix_table.hpp"
#include "shamunits/unit_table.hpp"
#include <stdexcept>

namespace {
    using namespace shamunits;
    using details::find_named;

    static_assert(units::registry.size() == 39);
    static_assert(prefix::registry.size() == 13);

    // Long and short names reach the same entry
    static_assert(find_named(units::registry, "metre")->unit == units::metre);
    static_assert(find_named(units::registry, "m") == find_named(units::registry, "metre"));
    static_assert(find_named(units::registry, "yr")->unit == units::year);
    static_assert(find_named(units::registry, "Becquerel")->unit == units::becquerel);
    static_assert(find_named(prefix::registry, "None")->unit == prefix::none);
    static_assert(find_named(prefix::registry, "_")->unit == prefix::none);

    // Same short name, different registries: metre and milli, tesla and tera
    static_assert(find_named(units::registry, "m")->unit == units::metre);
    static_assert(find_named(prefix::registry, "m")->unit == prefix::milli);
    static_assert(find_named(units::registry, "T")->unit == units::tesla);
    static_assert(find_named(prefix::registry, "T")->unit == prefix::tera);

    // A prefix multiplies the unit once (the former nested-prefix bug cannot exist here)
    static_assert(
        prefix::mega * units::year == Unit<>{.dim = {.second = 1}, .si_factor = 1e6 * 31557600.});
    static_assert((prefix::kilo * units::joule).si_factor == 1e3);
    static_assert(upow(prefix::kilo *units::metre, 2).dim == Dimension{.metre = 2});

    // Every prefix is dimensionless
    static_assert([] {
        for (const NamedUnit &p : prefix::registry) {
            if (!(p.unit.dim == Dimension{})) {
                return false;
            }
        }
        return true;
    }());

} // namespace

NEW_TEST(Unittest, "shamunits/unit_tables", 1) {
    REQUIRE(units::from_name("au") == units::astronomical_unit);
    REQUIRE(units::from_name("astronomical_unit") == units::astronomical_unit);
    REQUIRE(prefix::from_name("kilo") == prefix::kilo);
    REQUIRE(prefix::from_name("None") == prefix::none);
    REQUIRE_EXCEPTION_THROW(units::from_name("nonsense"), std::invalid_argument);
    REQUIRE_EXCEPTION_THROW(prefix::from_name("nonsense"), std::invalid_argument);
}
