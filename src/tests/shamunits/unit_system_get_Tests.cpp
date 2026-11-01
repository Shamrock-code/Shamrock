// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file unit_system_get_Tests.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Tests of the Unit-value interface of UnitSystem: get<u>(), get(u), get(u, p), to
 *
 */

#include "shambase/aliases_float.hpp"
#include "shamtest/shamtest.hpp"
#include "shamunits/UnitSystem.hpp"
#include "shamunits/prefix_table.hpp"
#include "shamunits/unit_table.hpp"
#include <cmath>
#include <string>
#include <utility>

namespace {
    using namespace shamunits;

    /// Relative comparison usable in constant expressions
    template<class T>
    constexpr bool close(T a, T b, T rel) {
        T d  = a - b;
        T ab = (b < 0) ? -b : b;
        return ((d < 0) ? -d : d) <= rel * ab;
    }

    // Compile-time: the whole README example is a constant expression now
    constexpr UnitSystem<f64> si{};
    static_assert(si.get<units::astronomical_unit>() == 149597870700.);
    static_assert(si.get<prefix::mega * units::year>() == 1e6 * 31557600.);
    static_assert(si.get<upow(units::astronomical_unit, 2)>() == 149597870700. * 149597870700.);
    static_assert(si.get<units::metre / upow(units::second, 2)>() == 1.);

    constexpr UnitSystem<f64> astro{
        si.get<prefix::mega * units::year>(),
        si.get<units::astronomical_unit>(),
        si.get<units::solar_mass>(),
    };
    static_assert(close(astro.get<upow(units::astronomical_unit, 2)>(), 1., 1e-12));
    // to<u>() is the inverse of get<u>(): one code length is 1 au = 1.496e11 m
    static_assert(close(
        astro.to<units::astronomical_unit>() * astro.get<units::astronomical_unit>(), 1., 1e-15));
    static_assert(close(astro.to<units::metre>(), 149597870700., 1e-15));
    static_assert(close(astro.get(units::astronomical_unit, 2), 1., 1e-12));
    // G in (Myr, au, Msol): 3.947813e13
    static_assert(close(
        astro.get<
            6.6743015e-11 * units::newton * upow(units::metre, 2) / upow(units::kilogram, 2)>(),
        3.947813e13,
        1e-6));

    // The result type is chosen at the call site; the factor is converted once
    static_assert(si.get<units::astronomical_unit, f32>() == f32(149597870700.));

    /// For registry entry i and power p: get<upow(u, p)>() == get(upow(u, p)) == get(u, p), and
    /// get * to == 1, in system `s`
    template<class T, std::size_t i, int p>
    bool entry_power_ok(const UnitSystem<T> &s, T tol) {
        constexpr Unit<> u  = units::registry[i].unit;
        constexpr Unit<> up = upow(u, p);
        T ct                = s.template get<up>();
        if (!std::isnormal(ct)) {
            return true; // the result itself does not fit in T
        }
        T rt     = s.get(up);
        T rp     = s.get(u, p);
        T rt_inv = s.to(u, p);
        bool ok  = close(rt, ct, tol) && close(rp, ct, tol);
        return ok
               && (!std::isnormal(rt) || !std::isnormal(rt_inv) || close(rt * rt_inv, T(1), tol));
    }
    template<class T, std::size_t i, int... ps>
    int entry_failures(const UnitSystem<T> &s, T tol, std::integer_sequence<int, ps...>) {
        return (int(!entry_power_ok<T, i, ps - 3>(s, tol)) + ...);
    }
    template<class T, std::size_t... is>
    int all_failures(const UnitSystem<T> &s, T tol, std::index_sequence<is...>) {
        return (entry_failures<T, is>(s, tol, std::make_integer_sequence<int, 7>{}) + ...);
    }
    template<class T>
    int registry_failures(const UnitSystem<T> &s, T tol) {
        return all_failures<T>(s, tol, std::make_index_sequence<units::registry.size()>{});
    }

} // namespace

NEW_TEST(Unittest, "shamunits/unit_system_get", 1) {
    // Three entry points agree for every unit and p in [-3, 3], f64 and f32, several systems
    UnitSystem<f64> systems64[] = {
        UnitSystem<f64>{},
        UnitSystem<f64>{3.15576e13, 1.495978707e11, 1.98847e30},
        UnitSystem<f64>{1, 1e-2, 1e-3},
        UnitSystem<f64>{2, 3, 5, 7, 11, 13, 17},
    };
    UnitSystem<f32> systems32[] = {
        UnitSystem<f32>{},
        UnitSystem<f32>{3.15576e13f, 1.495978707e11f, 1.98847e30f},
        UnitSystem<f32>{1, 1e-2f, 1e-3f},
        UnitSystem<f32>{2, 3, 5, 7, 11, 13, 17},
    };
    for (const auto &s : systems64) {
        REQUIRE_EQUAL(registry_failures<f64>(s, 1e-13), 0);
    }
    for (const auto &s : systems32) {
        REQUIRE_EQUAL(registry_failures<f32>(s, 1e-5f), 0);
    }

    // Runtime lookup by name, with a prefix, as the Python bindings do
    UnitSystem<f64> &astro_rt = systems64[1];
    f64 myr                   = astro_rt.get(prefix::from_name("M") * units::from_name("yr"));
    REQUIRE_FLOAT_EQUAL(myr, 1., 1e-12);
    REQUIRE_FLOAT_EQUAL(astro_rt.to(units::from_name("m")) / 149597870700., 1., 1e-15);

    // f32 in (Myr, au, Msol): G stays finite, since each base unit is multiplied in once and
    // in double precision (an f32 intermediate used to overflow to NaN)
    UnitSystem<f32> astro32{3.15576e13f, 1.495978707e11f, 1.98847e30f};
    constexpr Unit<> grav
        = 6.6743015e-11 * units::newton * upow(units::metre, 2) / upow(units::kilogram, 2);
    f32 g32 = astro32.get<grav>();
    REQUIRE(std::isfinite(g32));
    REQUIRE_FLOAT_EQUAL(g32 / 3.947813e13f, 1.f, 1e-5f);
}
