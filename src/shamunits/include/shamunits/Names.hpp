// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file Names.hpp
 * @author David Fang (david.fang@ikmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "details/utils.hpp"
#include <unordered_map>
#include <stdexcept>
#include <string>

/// \brief Definition of all units
///
/// This defines units and their short name
#define XMAC_UNITS                                                                                 \
    X1(second, s) /*base units*/                                                                   \
    X1(metre, m)                                                                                   \
    X1(kilogram, kg)                                                                               \
    X1(Ampere, A)                                                                                  \
    X1(Kelvin, K)                                                                                  \
    X1(mole, mol)                                                                                  \
    X1(candela, cd)                                                                                \
    /*derived units*/                                                                              \
    X1(Hertz, Hz)    /* hertz : frequency (s-1) */                                                 \
    X1(Newtown, N)   /* (kg.m.s-2)*/                                                               \
    X1(Pascal, Pa)   /* (kg.m-1.s-2) = (N/m2)*/                                                    \
    X1(Joule, J)     /* (kg.m2.s-2) = (N.m = Pa.m3)*/                                              \
    X1(Watt, W)      /* (kg.m2.s-3) = (J/s)*/                                                      \
    X1(Coulomb, C)   /* (s.A)*/                                                                    \
    X1(Volt, V)      /* (kg.m2.s-3.A-1) = (W/A) = (J/C)*/                                          \
    X1(Farad, F)     /* (kg-1.m-2.s4.A2) = (C/V) = (C2/J)*/                                        \
    X1(Ohm, ohm)     /* (kg.m2.s-3.A-2) = (V/A) = (J.s/C2)*/                                       \
    X1(Siemens, S)   /* (kg-1.m-2.s3.A2) = (ohm-1)*/                                               \
    X1(Weber, Wb)    /* (kg.m2.s-2.A-1) = (V.s)*/                                                  \
    X1(Tesla, T)     /* (kg.s-2.A-1) = (Wb/m2)*/                                                   \
    X1(Henry, H)     /* (kg.m2.s-2.A-2) = (Wb/A)*/                                                 \
    X1(lumens, lm)   /* (cd.sr) = (cd.sr)*/                                                        \
    X1(lux, lx)      /* (cd.sr.m-2) = (lm/m2)*/                                                    \
    X1(Bequerel, Bq) /* (s-1)*/                                                                    \
    X1(Gray, Gy)     /* (m2.s-2) = (J/kg)*/                                                        \
    X1(Sievert, Sv)  /* (m2.s-2) = (J/kg)*/                                                        \
    X1(katal, kat)   /* (mol.s-1) */                                                               \
    /*relative units*/                                                                             \
    X1(minutes, mn)                                                                                \
    X1(hours, hr)                                                                                  \
    X1(days, dy)                                                                                   \
    X1(years, yr)                                                                                  \
    X1(astronomical_unit, au)                                                                      \
    X1(light_year, ly)                                                                             \
    X1(parsec, pc)                                                                                 \
    X1(solar_radius, rsol)                                                                         \
    X1(earth_radius, rearth)                                                                       \
    X1(electron_volt, eV)                                                                          \
    X1(ergs, erg)                                                                                  \
    X1(british_pint, pint)

/// Definition of all prefixes
#define XMAC_UNIT_PREFIX                                                                           \
    X(tera, T, 12)                                                                                 \
    X(giga, G, 9)                                                                                  \
    X(mega, M, 6)                                                                                  \
    X(kilo, k, 3)                                                                                  \
    X(hecto, hect, 2)                                                                              \
    X(deca, dec, 1)                                                                                \
    X(None, _, 0)                                                                                  \
    /*X(deci  ,deci_, -1)*/                                                                        \
    X(centi, c, -2)                                                                                \
    X(milli, m, -3)                                                                                \
    X(micro, mu, -6)                                                                               \
    X(nano, n, -9)                                                                                 \
    X(pico, p, -12)                                                                                \
    X(femto, f, -15)

namespace shamunits {

    /// Enum of all prefixes
    enum UnitPrefix {
    /// Macro expending to all units prefixes in the enum
    // clang-format off
        #define X(longname, shortname, value) longname = value, shortname = value,
        XMAC_UNIT_PREFIX
        #undef X
        // clang-format on
    };

    /// Get the value of a prefix
    template<class T, UnitPrefix p>
    inline constexpr T get_prefix_val() {
        return details::pow_constexpr_fast_inv<p, T>(10, 1e-1);
    }

    /// Map to convert from a prefix name to a prefix enum value
    /// Ideally this should be replaced by cpp reflexion one day
    static const std::unordered_map<std::string, UnitPrefix> map_name_to_unit_prefix{
    // clang-format off
        #define X(longname, shortname, value) {#longname, longname}, {#shortname, shortname},
        XMAC_UNIT_PREFIX
        #undef X
        // clang-format on
    };

    /// Map to convert from unit prefix to prefix name in string
    /// Ideally this should be replaced by cpp reflexion one day
    static const std::unordered_map<UnitPrefix, std::string> map_u_to_name_prefix = {
    // clang-format off
        #define X(longname, shortname, value) {shortname, #shortname},
        XMAC_UNIT_PREFIX
        #undef X
        // clang-format on
    };

    /// Get the prefix name for a UnitPrefix enum value
    inline const std::string get_unit_prefix_name(UnitPrefix p) {

        map_u_to_name_prefix.find(p);

        if (auto search = map_u_to_name_prefix.find(p); search != map_u_to_name_prefix.end()) {
            return search->second;
        }

        return "[Unknown Unit prefix name]";
    }

    /// Get the UnitPrefix enum value from a prefix name as a string
    inline const UnitPrefix unit_prefix_from_name(std::string p) {

        map_name_to_unit_prefix.find(p);

        if (auto search = map_name_to_unit_prefix.find(p);
            search != map_name_to_unit_prefix.end()) {
            return search->second;
        }

        throw std::invalid_argument("this unit prefix name is unknown");
        return None; // to silence a warning
    }

    namespace units {

        // clang-format off
        /// List of all units name
        enum UnitName {
            /// Macro expanding to all unit names
            #define X1(longname, shortname) longname, shortname = longname,
            XMAC_UNITS
            #undef X1
        };

        /// Map to convert from string to unit name
        static const std::unordered_map<std::string, UnitName> map_name_to_unit{
            /// Macro expanding to the string->UnitName map
            #define X1(longname, shortname) {#longname, longname}, {#shortname, shortname},
            XMAC_UNITS
            #undef X1
        };

        /// Map to convert from unit name to string
        static const std::unordered_map<UnitName, std::string> map_u_to_name = {
            /// Macro expanding to the UnitName->string map
            #define X1(longname, shortname) {shortname, #shortname},
            XMAC_UNITS
            #undef X1
        };
        // clang-format on

        /// Get the unit name for a UnitName enum value
        inline const std::string get_unit_name(UnitName p) {

            map_u_to_name.find(p);

            if (auto search = map_u_to_name.find(p); search != map_u_to_name.end()) {
                return search->second;
            }

            return "[Unknown Unit name]";
        }

        /// Get the UnitName enum value from a unit name as a string
        inline const UnitName unit_from_name(std::string p) {

            auto search = map_name_to_unit.find(p);

            if (search != map_name_to_unit.end()) {
                return search->second;
            }

            throw std::invalid_argument("this unit name is unknown");
            return s; // to silence a warning
        }

    } // namespace units

} // namespace shamunits
