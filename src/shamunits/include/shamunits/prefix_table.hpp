// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2026 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file prefix_table.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief The table of unit prefixes: a prefix is a dimensionless Unit
 *
 * A prefix multiplies a unit once: `prefix::mega * units::year` is one Myr, and
 * `upow(prefix::kilo * units::metre, 2)` is one km2.
 */

#include "shamunits/Unit.hpp"
#include "shamunits/details/registry.hpp"
#include <string_view>
#include <array>

namespace shamunits::prefix {

    inline constexpr Unit<> tera  = scale(1e12);  ///< T
    inline constexpr Unit<> giga  = scale(1e9);   ///< G
    inline constexpr Unit<> mega  = scale(1e6);   ///< M
    inline constexpr Unit<> kilo  = scale(1e3);   ///< k
    inline constexpr Unit<> hecto = scale(1e2);   ///< hect
    inline constexpr Unit<> deca  = scale(1e1);   ///< dec
    inline constexpr Unit<> none  = scale(1.);    ///< no prefix ("None", "_")
    inline constexpr Unit<> centi = scale(1e-2);  ///< c
    inline constexpr Unit<> milli = scale(1e-3);  ///< m
    inline constexpr Unit<> micro = scale(1e-6);  ///< mu
    inline constexpr Unit<> nano  = scale(1e-9);  ///< n
    inline constexpr Unit<> pico  = scale(1e-12); ///< p
    inline constexpr Unit<> femto = scale(1e-15); ///< f

    /// Name registry: Python passes pref="None" by default, so "None" must stay
    inline constexpr auto registry = std::to_array<NamedUnit>({
        {"tera", "T", tera},
        {"giga", "G", giga},
        {"mega", "M", mega},
        {"kilo", "k", kilo},
        {"hecto", "hect", hecto},
        {"deca", "dec", deca},
        {"None", "_", none},
        {"centi", "c", centi},
        {"milli", "m", milli},
        {"micro", "mu", micro},
        {"nano", "n", nano},
        {"pico", "p", pico},
        {"femto", "f", femto},
    });

    static_assert(details::names_are_valid(registry), "empty or duplicate prefix name");

    /// Prefix from its long or short name; throws std::invalid_argument if unknown
    inline Unit<> from_name(std::string_view name) {
        return details::unit_from_name(registry, name, "prefix");
    }

} // namespace shamunits::prefix
