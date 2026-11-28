// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file registry_Tests.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Tests of the shamunits name registry and of the runtime integer power ipow
 *
 */

#include "shambase/aliases_float.hpp"
#include "shamtest/shamtest.hpp"
#include "shamunits/Unit.hpp"
#include "shamunits/details/registry.hpp"
#include "shamunits/details/utils.hpp"
#include <array>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
    using namespace shamunits;
    using shamunits::details::find_named;
    using shamunits::details::ipow;
    using shamunits::details::names_are_valid;
    using shamunits::details::pow_constexpr_fast_inv;
    using shamunits::details::unit_from_name;

    // Local stand-in registry, independent of the real unit tables
    constexpr Unit<> metre  = base_unit({.metre = 1});
    constexpr Unit<> second = base_unit({.second = 1});
    constexpr Unit<> au     = 149597870700. * metre;

    constexpr auto test_registry = std::to_array<NamedUnit>({
        {"metre", "m", metre},
        {"second", "s", second},
        {"astronomical_unit", "au", au},
    });

    // Lookup by long or short name, at compile time
    static_assert(find_named(test_registry, "metre")->unit == metre);
    static_assert(find_named(test_registry, "au")->unit == au);
    static_assert(
        find_named(test_registry, "astronomical_unit") == find_named(test_registry, "au"));
    static_assert(find_named(test_registry, "nonsense") == nullptr);
    static_assert(find_named(test_registry, "") == nullptr);

    // Name validation, which the real tables will static_assert
    static_assert(names_are_valid(test_registry));
    static_assert(!names_are_valid(
        std::to_array<NamedUnit>({
            {"metre", "m", metre},
            {"minute", "m", second}, // short name clash
        })));
    static_assert(!names_are_valid(
        std::to_array<NamedUnit>({
            {"metre", "m", metre},
            {"m", "mtr", metre}, // long name equal to another short name
        })));
    static_assert(!names_are_valid(
        std::to_array<NamedUnit>({
            {"metre", "", metre}, // empty name
        })));

    // ipow is bit-identical to pow_constexpr_fast_inv<n>, for every n in [-8, 8]
    template<class T, int n>
    constexpr bool ipow_matches(T a) {
        return ipow(a, n, T{1} / a) == pow_constexpr_fast_inv<n>(a, T{1} / a);
    }
    template<class T, int... ns>
    constexpr bool ipow_matches_all(T a, std::integer_sequence<int, ns...>) {
        return (ipow_matches<T, ns - 8>(a) && ...);
    }
    template<class T>
    constexpr bool ipow_matches_range(T a) {
        return ipow_matches_all<T>(a, std::make_integer_sequence<int, 17>{});
    }
    static_assert(ipow_matches_range<f64>(149597870700.));
    static_assert(ipow_matches_range<f64>(1. / 3.));
    static_assert(ipow_matches_range<f32>(1.5f));
    static_assert(ipow_matches_range<f32>(0.1f));
    static_assert(ipow(2., 0, .5) == 1.);
    static_assert(ipow(2., -3, .5) == .125); // uses a_inv, no division

} // namespace

NEW_TEST(Unittest, "shamunits/registry", 1) {
    REQUIRE(unit_from_name(test_registry, "au") == au);
    REQUIRE(unit_from_name(test_registry, "second") == second);
    REQUIRE_EXCEPTION_THROW(unit_from_name(test_registry, "nonsense"), std::invalid_argument);

    // ipow at run time (exponent not a constant): same values as the compile-time expansion
    volatile int n_runtime = -3;
    f64 a                  = 1.495978707e11;
    REQUIRE_EQUAL(ipow(a, n_runtime, 1. / a), (pow_constexpr_fast_inv<-3>(a, 1. / a)));
}
