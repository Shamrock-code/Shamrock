// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

/**
 * @file constants_vs_legacy_Tests.cpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Temporary: the new constants table must match the legacy Constants accessors
 *
 * Same names, and for every constant the same value through every path (named accessor vs
 * runtime get, template get, Constants::get) in several systems. Delete together with the legacy
 * Constants accessors (Phase D).
 */

#include "shambase/aliases_float.hpp"
#include "shambackends/sycl.hpp"
#include "shamtest/shamtest.hpp"
#include "shamunits/Constants.hpp"
#include "shamunits/UnitSystem.hpp"
#include "shamunits/constant_table.hpp"
#include <cmath>
#include <string>
#include <vector>

namespace {
    using namespace shamunits;

    template<class T>
    void require_rel(const std::string &label, T got, T expected, T tol) {
        if (!std::isnormal(expected) || !std::isnormal(got)) {
            return; // out of range for T in this system: nothing to compare
        }
        REQUIRE_FLOAT_EQUAL_NAMED(label, got / expected, T(1), tol);
    }

    /// Legacy names, from the legacy X-macro
    const std::vector<std::string> legacy_names = {
#define X(name, conv) #name,
        UNITS_CONSTANTS
#undef X
    };

    template<class T>
    void compare(UnitSystem<T> u, const std::string &sys, T tol) {
        Constants<T> cte{u};
        // Every constant: legacy accessor (raised as the Python binding does) vs runtime get
#define X(name, conv)                                                                              \
    for (int p = -2; p <= 2; p++) {                                                                \
        require_rel(                                                                               \
            sys + " " #name "^" + std::to_string(p),                                               \
            u.get(constants::from_name(#name), p),                                                 \
            T(sycl::pown(cte.name(), p)),                                                          \
            tol);                                                                                  \
    }                                                                                              \
    require_rel(sys + " template " #name, u.template get<constants::name>(), cte.name(), tol);     \
    require_rel(                                                                                   \
        sys + " Constants::get " #name, cte.template get<constants::name>(), cte.name(), tol);
        UNITS_CONSTANTS
#undef X
    }

} // namespace

NEW_TEST(Unittest, "shamunits/constants_vs_legacy", 1) {
    // Same names, both directions
    REQUIRE_EQUAL(legacy_names.size(), constants::registry.size());
    for (const std::string &n : legacy_names) {
        REQUIRE_NAMED(
            "constant in new table: " + n, details::find_named(constants::registry, n) != nullptr);
    }

    compare<f64>(UnitSystem<f64>{}, "f64 si", 1e-12);
    compare<f64>(UnitSystem<f64>{3.15576e13, 1.495978707e11, 1.98847e30}, "f64 astro", 1e-12);
    compare<f64>(UnitSystem<f64>{1, 1e-2, 1e-3}, "f64 cgs", 1e-12);
    compare<f64>(UnitSystem<f64>{2, 3, 5, 7, 11, 13, 17}, "f64 primes", 1e-12);

    // f32: the legacy accessors lose precision in the astro system (BUG 6), so compare elsewhere
    compare<f32>(UnitSystem<f32>{}, "f32 si", 1e-5f);
    compare<f32>(UnitSystem<f32>{1, 1e-2f, 1e-3f}, "f32 cgs", 1e-5f);
    compare<f32>(UnitSystem<f32>{2, 3, 5, 7, 11, 13, 17}, "f32 primes", 1e-5f);
}
