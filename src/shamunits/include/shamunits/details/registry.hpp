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
 * @file registry.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Name tables mapping long and short names to units, with constexpr lookup
 *
 * Each category (units, prefixes, constants) owns one registry, a constexpr array of
 * NamedUnit next to its table. Names only need to be unique within one registry, so "m" can be
 * both metre (units) and milli (prefixes). Host-only: unlike Unit.hpp this uses the standard
 * library and exceptions.
 */

#include "shamunits/Unit.hpp"
#include <string_view>
#include <span>
#include <stdexcept>
#include <string>

namespace shamunits {

    /// One registry entry: a unit and the two names it can be looked up by
    struct NamedUnit {
        std::string_view long_name;  ///< e.g. "metre", "astronomical_unit"
        std::string_view short_name; ///< e.g. "m", "au"
        Unit<> unit;                 ///< the unit, always in the reference precision (double)

        /// Constructor, so registry rows stay compact: `{"metre", "m", metre}`
        constexpr NamedUnit(std::string_view lname, std::string_view sname, Unit<> u) noexcept
            : long_name(lname), short_name(sname), unit(u) {}
    };

} // namespace shamunits

namespace shamunits::details {

    /// Entry whose long or short name is `name`, or nullptr
    constexpr const NamedUnit *find_named(
        std::span<const NamedUnit> registry, std::string_view name) noexcept {
        for (const NamedUnit &e : registry) {
            if (e.long_name == name || e.short_name == name) {
                return &e;
            }
        }
        return nullptr;
    }

    /// Unit named `name` in `registry`; throws std::invalid_argument when unknown
    inline Unit<> unit_from_name(
        std::span<const NamedUnit> registry,
        std::string_view name,
        std::string_view kind = "unit") {
        if (const NamedUnit *e = find_named(registry, name); e != nullptr) {
            return e->unit;
        }
        throw std::invalid_argument(
            "unknown " + std::string(kind) + " name : '" + std::string(name) + "'");
    }

    /// True when no name is empty and no name appears twice, long and short names together
    constexpr bool names_are_valid(std::span<const NamedUnit> registry) noexcept {
        for (std::size_t i = 0; i < registry.size(); i++) {
            const NamedUnit &a = registry[i];
            if (a.long_name.empty() || a.short_name.empty()) {
                return false;
            }
            for (std::size_t j = i + 1; j < registry.size(); j++) {
                const NamedUnit &b = registry[j];
                if (a.long_name == b.long_name || a.long_name == b.short_name
                    || a.short_name == b.long_name || a.short_name == b.short_name) {
                    return false;
                }
            }
        }
        return true;
    }

} // namespace shamunits::details
