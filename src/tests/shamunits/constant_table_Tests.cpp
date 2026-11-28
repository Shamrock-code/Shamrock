// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file constant_table_Tests.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Tests of the physical constants table (shamunits/constant_table.hpp)
 *
 */

#include "shambase/aliases_float.hpp"
#include "shamtest/shamtest.hpp"
#include "shamunits/Constants.hpp"
#include "shamunits/UnitSystem.hpp"
#include "shamunits/constant_table.hpp"
#include "shamunits/prefix_table.hpp"
#include "shamunits/unit_table.hpp"
#include <stdexcept>

namespace {
    using namespace shamunits;
    using details::find_named;

    template<class T>
    constexpr bool close(T a, T b, T rel) {
        T d  = a - b;
        T ab = (b < 0) ? -b : b;
        return ((d < 0) ? -d : d) <= rel * ab;
    }

    static_assert(constants::registry.size() == 35);
    static_assert(find_named(constants::registry, "G")->unit == constants::G);
    static_assert(
        find_named(constants::registry, "astronomical_unit")->unit == units::astronomical_unit);
    static_assert(constants::sol_mass == units::solar_mass);
    static_assert(constants::year == units::year);

    // A constant is used like a unit, at compile time
    constexpr UnitSystem<f64> si{};
    static_assert(si.get<constants::c>() == 299792458.);
    static_assert(si.get<constants::G>() == 6.6743015e-11);

    constexpr UnitSystem<f64> astro{
        si.get<prefix::mega * units::year>(),
        si.get<units::astronomical_unit>(),
        si.get<constants::sol_mass>(),
    };
    // G in (Myr, au, Msol) = 4 pi^2 (au^3/Msol/yr^2) * 1e12 (yr -> Myr)
    static_assert(close(astro.get<constants::G>(), 3.947813e13, 1e-6));
    static_assert(
        close(Constants<f64>{astro}.get<constants::G>(), astro.get<constants::G>(), 1e-15));

    // Derived constants keep their SI definitions
    static_assert(close(si.get<constants::epsilon_0>(), 8.8541878128e-12, 1e-9));
    static_assert(close(si.get<constants::hbar>(), 1.054571817e-34, 1e-9));

} // namespace

NEW_TEST(Unittest, "shamunits/constant_table", 1) {
    UnitSystem<f64> si{};
    Constants<f64> cte{si};
    REQUIRE(constants::from_name("G") == constants::G);
    REQUIRE_EQUAL(cte.get(constants::from_name("c")), 299792458.);
    REQUIRE_FLOAT_EQUAL(cte.get(constants::G, 2) / (6.6743015e-11 * 6.6743015e-11), 1., 1e-15);
    REQUIRE_EXCEPTION_THROW(constants::from_name("nonsense"), std::invalid_argument);
}
