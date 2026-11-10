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
 * @file unit_table.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief The table of units: one constexpr Unit per unit, plus its name registry
 *
 * To add a unit: declare it below from existing units, then add one row to the registry. Both
 * are checked at compile time (dimensions, unique names). Factors are Unit<> (double); combine
 * them with double literals only (`60. * second`, not `60 * second`).
 */

#include "shamunits/Unit.hpp"
#include "shamunits/details/registry.hpp"
#include <string_view>
#include <array>

namespace shamunits::units {

    // SI base units
    inline constexpr Unit<> second   = base_unit({.second = 1});   ///< s
    inline constexpr Unit<> metre    = base_unit({.metre = 1});    ///< m
    inline constexpr Unit<> kilogram = base_unit({.kilogram = 1}); ///< kg
    inline constexpr Unit<> ampere   = base_unit({.ampere = 1});   ///< A
    inline constexpr Unit<> kelvin   = base_unit({.kelvin = 1});   ///< K
    inline constexpr Unit<> mole     = base_unit({.mole = 1});     ///< mol
    inline constexpr Unit<> candela  = base_unit({.candela = 1});  ///< cd

    // SI derived units
    inline constexpr Unit<> hertz     = 1. / second;                        ///< Hz = s-1
    inline constexpr Unit<> newton    = kilogram * metre / upow(second, 2); ///< N = kg.m.s-2
    inline constexpr Unit<> pascal    = newton / upow(metre, 2);            ///< Pa = N.m-2
    inline constexpr Unit<> joule     = newton * metre;                     ///< J = N.m
    inline constexpr Unit<> watt      = joule / second;                     ///< W = J.s-1
    inline constexpr Unit<> coulomb   = second * ampere;                    ///< C = s.A
    inline constexpr Unit<> volt      = watt / ampere;                      ///< V = W.A-1
    inline constexpr Unit<> farad     = coulomb / volt;                     ///< F = C.V-1
    inline constexpr Unit<> ohm       = volt / ampere;                      ///< ohm = V.A-1
    inline constexpr Unit<> siemens   = 1. / ohm;                           ///< S = ohm-1
    inline constexpr Unit<> weber     = volt * second;                      ///< Wb = V.s
    inline constexpr Unit<> tesla     = weber / upow(metre, 2);             ///< T = Wb.m-2
    inline constexpr Unit<> henry     = weber / ampere;                     ///< H = Wb.A-1
    inline constexpr Unit<> lumen     = candela;                            ///< lm = cd.sr
    inline constexpr Unit<> lux       = lumen / upow(metre, 2);             ///< lx = lm.m-2
    inline constexpr Unit<> becquerel = 1. / second;                        ///< Bq = s-1
    inline constexpr Unit<> gray      = joule / kilogram;                   ///< Gy = J.kg-1
    inline constexpr Unit<> sievert   = joule / kilogram;                   ///< Sv = J.kg-1
    inline constexpr Unit<> katal     = mole / second;                      ///< kat = mol.s-1

    // Other units
    inline constexpr Unit<> minute            = 60. * second;              ///< mn
    inline constexpr Unit<> hour              = 3600. * second;            ///< hr
    inline constexpr Unit<> day               = 86400. * second;           ///< dy
    inline constexpr Unit<> year              = 31557600. * second;        ///< yr (Julian)
    inline constexpr Unit<> astronomical_unit = 149597870700. * metre;     ///< au
    inline constexpr Unit<> light_year        = 9460730472580800. * metre; ///< ly
    inline constexpr Unit<> parsec            = 3.0857e16 * metre;         ///< pc
    inline constexpr Unit<> solar_radius      = 6.957e8 * metre;           ///< rsol
    inline constexpr Unit<> earth_radius      = 6.378137e6 * metre;        ///< rearth (equatorial)
    inline constexpr Unit<> solar_mass        = 1.98847e30 * kilogram;     ///< sol_mass
    inline constexpr Unit<> electron_volt     = 1.602176634e-19 * joule;   ///< eV
    inline constexpr Unit<> erg               = 1e-7 * joule;              ///< erg
    inline constexpr Unit<> british_pint      = 5.68e-4 * upow(metre, 3);  ///< pint (0.568 L)

    /// Name registry: long and short names. Python and saved scripts resolve these strings, so
    /// they must stay
    inline constexpr auto registry = std::to_array<NamedUnit>({
        {"second", "s", second},
        {"metre", "m", metre},
        {"kilogram", "kg", kilogram},
        {"Ampere", "A", ampere},
        {"Kelvin", "K", kelvin},
        {"mole", "mol", mole},
        {"candela", "cd", candela},
        {"Hertz", "Hz", hertz},
        {"Newton", "N", newton},
        {"Pascal", "Pa", pascal},
        {"Joule", "J", joule},
        {"Watt", "W", watt},
        {"Coulomb", "C", coulomb},
        {"Volt", "V", volt},
        {"Farad", "F", farad},
        {"Ohm", "ohm", ohm},
        {"Siemens", "S", siemens},
        {"Weber", "Wb", weber},
        {"Tesla", "T", tesla},
        {"Henry", "H", henry},
        {"lumens", "lm", lumen},
        {"lux", "lx", lux},
        {"Becquerel", "Bq", becquerel},
        {"Gray", "Gy", gray},
        {"Sievert", "Sv", sievert},
        {"katal", "kat", katal},
        {"minutes", "mn", minute},
        {"hours", "hr", hour},
        {"days", "dy", day},
        {"years", "yr", year},
        {"astronomical_unit", "au", astronomical_unit},
        {"light_year", "ly", light_year},
        {"parsec", "pc", parsec},
        {"solar_radius", "rsol", solar_radius},
        {"earth_radius", "rearth", earth_radius},
        {"solar_mass", "sol_mass", solar_mass},
        {"electron_volt", "eV", electron_volt},
        {"ergs", "erg", erg},
        {"british_pint", "pint", british_pint},
    });

    static_assert(details::names_are_valid(registry), "empty or duplicate unit name");

    // Dimension cross-checks: catch a wrong composition at compile time
    static_assert(pascal.dim == Dimension{.second = -2, .metre = -1, .kilogram = 1});
    static_assert(joule.dim == Dimension{.second = -2, .metre = 2, .kilogram = 1});
    static_assert(watt.dim == Dimension{.second = -3, .metre = 2, .kilogram = 1});
    static_assert(volt.dim == Dimension{.second = -3, .metre = 2, .kilogram = 1, .ampere = -1});
    static_assert(farad.dim == Dimension{.second = 4, .metre = -2, .kilogram = -1, .ampere = 2});
    static_assert(ohm.dim == Dimension{.second = -3, .metre = 2, .kilogram = 1, .ampere = -2});
    static_assert(siemens.dim == Dimension{.second = 3, .metre = -2, .kilogram = -1, .ampere = 2});
    static_assert(weber.dim == Dimension{.second = -2, .metre = 2, .kilogram = 1, .ampere = -1});
    static_assert(tesla.dim == Dimension{.second = -2, .kilogram = 1, .ampere = -1});
    static_assert(henry.dim == Dimension{.second = -2, .metre = 2, .kilogram = 1, .ampere = -2});
    static_assert(lux.dim == Dimension{.metre = -2, .candela = 1});
    static_assert(gray.dim == Dimension{.second = -2, .metre = 2});
    static_assert(katal.dim == Dimension{.second = -1, .mole = 1});
    static_assert(electron_volt.dim == joule.dim && erg.dim == joule.dim);
    static_assert(british_pint.dim == Dimension{.metre = 3});

    /// Unit from its long or short name; throws std::invalid_argument if unknown
    inline Unit<> from_name(std::string_view name) {
        return details::unit_from_name(registry, name, "unit");
    }

} // namespace shamunits::units
