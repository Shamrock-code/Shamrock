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
 * @file constant_table.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief The table of physical constants: each constant is a Unit (its SI value and dimension)
 *
 * A constant is the same thing as a unit: `usys.get<constants::G>()` is G in the system `usys`.
 * Derived constants are written from other constants, so their dimension is derived and checked
 * rather than retyped. To add a constant: declare it below, then add one registry row; it then
 * appears in Python with no binding edit. Combine with double literals only.
 */

#include "shamunits/Unit.hpp"
#include "shamunits/details/registry.hpp"
#include "shamunits/unit_table.hpp"
#include <string_view>
#include <array>

namespace shamunits::details {
    /// pi, as a plain number for building constants (same value as shamunits::pi<double>)
    inline constexpr double pi = 3.141592653589793116;
} // namespace shamunits::details

namespace shamunits::constants {

    // SI defining constants
    inline constexpr Unit<> delta_nu_cs = 9192631770. * units::hertz; ///< Cs hyperfine frequency
    inline constexpr Unit<> c = 299792458. * units::metre / units::second;     ///< speed of light
    inline constexpr Unit<> h = 6.62607015e-34 * units::joule * units::second; ///< Planck constant
    inline constexpr Unit<> e = 1.602176634e-19 * units::coulomb;            ///< elementary charge
    inline constexpr Unit<> k = 1.380649e-23 * units::joule / units::kelvin; ///< Boltzmann constant
    // NOLINTNEXTLINE(readability-identifier-naming): physics symbol, also the Python method name
    inline constexpr Unit<> Na = 6.02214076e23 / units::mole; ///< Avogadro constant
    // NOLINTNEXTLINE(readability-identifier-naming): physics symbol, also the Python method name
    inline constexpr Unit<> Kcd = 683. * units::lumen / units::watt; ///< luminous efficacy

    // Times
    inline constexpr Unit<> second = units::second; ///< second
    inline constexpr Unit<> minute = units::minute; ///< minute
    inline constexpr Unit<> hour   = units::hour;   ///< hour
    inline constexpr Unit<> day    = units::day;    ///< day
    inline constexpr Unit<> year   = units::year;   ///< Julian year

    // Lengths
    inline constexpr Unit<> astronomical_unit = units::astronomical_unit; ///< au
    inline constexpr Unit<> au           = units::astronomical_unit; ///< alias of astronomical_unit
    inline constexpr Unit<> light_year   = units::light_year;        ///< light year
    inline constexpr Unit<> parsec       = units::parsec;            ///< parsec
    inline constexpr Unit<> solar_radius = units::solar_radius;      ///< solar radius
    inline constexpr Unit<> earth_radius = units::earth_radius;      ///< Earth equatorial radius
    inline constexpr Unit<> planck_length = 1.61625518e-35 * units::metre; ///< Planck length

    // Masses
    inline constexpr Unit<> proton_mass = 1.67262192e-27 * units::kilogram; ///< proton mass
    /// electron mass, from the proton/electron mass ratio
    inline constexpr Unit<> electron_mass = proton_mass * (1. / 1836.1526734311);
    inline constexpr Unit<> earth_mass    = 5.9722e24 * units::kilogram;     ///< Earth mass
    inline constexpr Unit<> jupiter_mass  = 1.898e27 * units::kilogram;      ///< Jupiter mass
    inline constexpr Unit<> sol_mass      = units::solar_mass;               ///< solar mass
    inline constexpr Unit<> planck_mass   = 2.17643424e-8 * units::kilogram; ///< Planck mass
    inline constexpr Unit<> dalton = 1.66053906892e-27 * units::kilogram;    ///< atomic mass unit

    // Densities
    /// density of Guinness, 1.017 g.cm-3
    inline constexpr Unit<> guiness_density
        = (1.017 * 1000.) * units::kilogram / upow(units::metre, 3);

    // Derived constants
    // NOLINTNEXTLINE(readability-identifier-naming): physics symbol, also the Python method name
    inline constexpr Unit<> G
        = 6.6743015e-11 * units::newton * upow(units::metre, 2) / upow(units::kilogram, 2);
    inline constexpr Unit<> hbar = h / (2. * details::pi); ///< reduced Planck
    inline constexpr Unit<> mu_0
        = 1.2566370621219e-6 * units::newton / upow(units::ampere, 2); ///< vacuum permeability
    // NOLINTNEXTLINE(readability-identifier-naming): physics symbol, also the Python method name
    inline constexpr Unit<> Z_0       = mu_0 * c;       ///< vacuum impedance
    inline constexpr Unit<> epsilon_0 = 1. / (Z_0 * c); ///< vacuum permittivity
    inline constexpr Unit<> ke        = 1. / (4. * details::pi * epsilon_0); ///< Coulomb constant
    inline constexpr Unit<> kb
        = 1.380649e-23 * units::joule / units::kelvin; ///< Boltzmann constant
    /// Stefan-Boltzmann constant
    inline constexpr Unit<> sigma
        = 5.670374419e-8 * units::watt / upow(units::metre, 2) / upow(units::kelvin, 4);

    /// Name registry: one name per constant (long and short are the same). These are the Python
    /// method names (Constants(usys).G())
    inline constexpr auto registry = std::to_array<NamedUnit>({
        {"delta_nu_cs", "delta_nu_cs", delta_nu_cs},
        {"c", "c", c},
        {"h", "h", h},
        {"e", "e", e},
        {"k", "k", k},
        {"Na", "Na", Na},
        {"Kcd", "Kcd", Kcd},
        {"second", "second", second},
        {"minute", "minute", minute},
        {"hour", "hour", hour},
        {"day", "day", day},
        {"year", "year", year},
        {"au", "au", au},
        {"astronomical_unit", "astronomical_unit", astronomical_unit},
        {"light_year", "light_year", light_year},
        {"parsec", "parsec", parsec},
        {"solar_radius", "solar_radius", solar_radius},
        {"earth_radius", "earth_radius", earth_radius},
        {"planck_length", "planck_length", planck_length},
        {"proton_mass", "proton_mass", proton_mass},
        {"electron_mass", "electron_mass", electron_mass},
        {"earth_mass", "earth_mass", earth_mass},
        {"jupiter_mass", "jupiter_mass", jupiter_mass},
        {"sol_mass", "sol_mass", sol_mass},
        {"planck_mass", "planck_mass", planck_mass},
        {"dalton", "dalton", dalton},
        {"guiness_density", "guiness_density", guiness_density},
        {"G", "G", G},
        {"hbar", "hbar", hbar},
        {"mu_0", "mu_0", mu_0},
        {"Z_0", "Z_0", Z_0},
        {"epsilon_0", "epsilon_0", epsilon_0},
        {"ke", "ke", ke},
        {"kb", "kb", kb},
        {"sigma", "sigma", sigma},
    });

    static_assert(details::names_are_valid(registry), "empty or duplicate constant name");

    // Dimension checks: derived constants must land on the expected dimension
    static_assert(h.dim == (units::joule * units::second).dim && hbar.dim == h.dim);
    static_assert(G.dim == Dimension{.second = -2, .metre = 3, .kilogram = -1});
    static_assert(mu_0.dim == (units::newton / upow(units::ampere, 2)).dim);
    static_assert(Z_0.dim == units::ohm.dim);
    static_assert(epsilon_0.dim == (units::farad / units::metre).dim);
    static_assert(ke.dim == (units::newton * upow(units::metre, 2) / upow(units::coulomb, 2)).dim);
    static_assert(sigma.dim == (units::watt / upow(units::metre, 2) / upow(units::kelvin, 4)).dim);
    static_assert(Kcd.dim == (units::lumen / units::watt).dim && Na.dim == (1. / units::mole).dim);
    static_assert(guiness_density.dim == (units::kilogram / upow(units::metre, 3)).dim);
    static_assert(k == kb && au == astronomical_unit);

    /// Constant from its name; throws std::invalid_argument if unknown
    inline Unit<> from_name(std::string_view name) {
        return shamunits::details::unit_from_name(registry, name, "constant");
    }

} // namespace shamunits::constants
