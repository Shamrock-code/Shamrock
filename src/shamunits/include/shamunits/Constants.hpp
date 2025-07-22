// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2024 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file Constants.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 */

#include "ConvertionConstants.hpp"
#include "Names.hpp"
#include "UnitSystem.hpp"
#include "details/utils.hpp"
#include <stdexcept>

/// @brief Macro to add a constant to the \ref shamunits::Constants class
/// For exemple one can add a new constant by adding the following statment inside
/// shamunits::Constants
/// \code{.cpp}
///     addconstant(delta_nu_cs) { return Cget(Si::delta_nu_cs,1) * Uget(Hertz, 1); }
/// \endcode
#define addconstant(name)                                                                          \
    template<int power = 1>                                                                        \
    inline constexpr T name()

/// Utility macro to get the value of a unit in the current unit system with its power
#define Uget(unitname, mult_pow) units.template get<None, units::unitname, (mult_pow) * power>()

/// Utility macro to get the value of a constant in the current unit system with its power
#define Cget(constant_name, mult_pow)                                                              \
    details::pow_constexpr_fast_inv<(mult_pow) * power>(constant_name, 1 / constant_name)

/// X macro to list all constants conversion & bindings
#define UNITS_CONSTANTS                                                                            \
    /* si base ctes  */                                                                            \
    X(delta_nu_cs /**/, Uget(Hertz, 1))                                                            \
    X(c /************/, Uget(m, 1) * Uget(s, -1))                                                  \
    X(h /************/, Uget(Joule, 1) * Uget(s, -1))                                              \
    X(e /************/, Uget(Coulomb, 1))                                                          \
    X(k /************/, Uget(Joule, 1) * Uget(Kelvin, -1))                                         \
    X(Na /***********/, Uget(mole, -1))                                                            \
    X(Kcd /**********/, Uget(lm, 1) * Uget(Watt, -1))                                              \
    /* times */                                                                                    \
    X(hour /**/, Uget(s, 1))                                                                       \
    X(day /***/, Uget(s, 1))                                                                       \
    X(year /**/, Uget(s, 1))                                                                       \
    /* sizes */                                                                                    \
    X(au /*****************/, Uget(m, 1))                                                          \
    X(astronomical_unit /**/, Uget(m, 1))                                                          \
    X(light_year /*********/, Uget(m, 1))                                                          \
    X(parsec /*************/, Uget(m, 1))                                                          \
    X(planck_length /******/, Uget(m, 1))                                                          \
    /* masses */                                                                                   \
    X(proton_mass /****/, Uget(kg, 1))                                                             \
    X(electron_mass /**/, Uget(kg, 1))                                                             \
    X(earth_mass /*****/, Uget(kg, 1))                                                             \
    X(jupiter_mass /***/, Uget(kg, 1))                                                             \
    X(sol_mass /*******/, Uget(kg, 1))                                                             \
    X(planck_mass /****/, Uget(kg, 1))                                                             \
    /* densities */                                                                                \
    X(guiness_density, Uget(kg, 1) * Uget(m, -1))                                                  \
    /* derived ctes  */                                                                            \
    X(G /**********/, Uget(N, 1) * Uget(m, 2) * Uget(kg, -2))                                      \
    X(hbar /*******/, Uget(Joule, 1) * Uget(s, -1))                                                \
    X(mu_0 /*******/, Uget(N, 1) * Uget(A, -2))                                                    \
    X(Z_0 /********/, Uget(Ohm, 1))                                                                \
    X(epsilon_0 /**/, Uget(F, 1) * Uget(m, -1))                                                    \
    X(ke /*********/, Uget(N, 1) * Uget(m, 2) * Uget(Coulomb, -2))                                 \
    X(kb /*********/, Uget(J, 1) * Uget(K, -1))                                                    \
    X(sigma /******/, Uget(W, 1) * Uget(m, -2) * Uget(K, -4))

namespace shamunits {

    /**
     * @brief Value of pi
     * Usage : `auto pi = shamunits::pi<T>;`
     */
    template<class T>
    constexpr T pi = 3.141592653589793116;

    /// Fine structure constant
    template<class T>
    constexpr T fine_structure = 0.0072973525693;

    /// Mass ration between the proton and electron
    template<class T>
    constexpr T proton_electron_ratio = 1836.1526734311;

    /// Mass ration between the electron and proton
    template<class T>
    constexpr T electron_proton_ratio = 1 / proton_electron_ratio<T>;

    /**
     * @brief Physical constants
     */
    template<class T>
    struct Constants {

        /// Alias to the conversion constants
        using Conv = ConvertionConstants<T>;

        /// Physical constant in SI units
        struct Si {

            // si system base constants
            static constexpr T delta_nu_cs = 9192631770;      // (s-1)
            static constexpr T c           = 299792458;       // (m.s-1)
            static constexpr T h           = 6.62607015e-34;  // (J.s-1)
            static constexpr T e           = 1.602176634e-19; // (C)
            static constexpr T k           = 1.380649e-23;    // (J.K-1 )
            static constexpr T Na          = 6.02214076e23;   // (mol-1 )
            static constexpr T Kcd         = 683;             // (lm.W-1)

            // other constants in si units
            static constexpr T G         = 6.6743015e-11;               // (N.m2.kg-2)
            static constexpr T hbar      = h / (2 * pi<T>);             // (J.s-1)
            static constexpr T mu_0      = 1.2566370621219e-6;          // (N.A-2)
            static constexpr T Z_0       = mu_0 * c;                    // (Ohm)
            static constexpr T epsilon_0 = 1 / (Z_0 * c);               // (F.m-1)
            static constexpr T ke        = 1 / (4 * pi<T> * epsilon_0); // (N.m2.C-2)
            static constexpr T kb        = 1.380649e-23;                // (J.K-1)

            /// Stephan Boltzmann constant (W.m-2.K-4)
            static constexpr T sigma = 5.670374419e-8;

            static constexpr T hour = Conv::hr_to_s; //(s)
            static constexpr T day  = Conv::dy_to_s; //(s)
            static constexpr T year = Conv::yr_to_s; //(s)

            static constexpr T astronomical_unit = Conv::au_to_m;     //(m)
            static constexpr T au                = astronomical_unit; //(m)
            static constexpr T light_year        = Conv::ly_to_m;     //(m)
            static constexpr T parsec            = Conv::pc_to_m;     //(m)
            static constexpr T planck_length     = 1.61625518e-35;    //(m)

            static constexpr T proton_mass   = 1.67262192e-27;                         //(kg)
            static constexpr T electron_mass = proton_mass * electron_proton_ratio<T>; //(kg)
            static constexpr T earth_mass    = 5.9722e24;                              //(kg)
            static constexpr T jupiter_mass  = 1.898e27;                               //(kg)
            static constexpr T sol_mass      = 1.98847e30;                             //(kg)
            static constexpr T planck_mass   = 2.17643424e-8;                          //(kg)

            static constexpr T guiness_density = Conv::gcm3_to_guiness_density * 1000; //(kg.m-3)
        };

        /// Current unit system of the constants
        const UnitSystem<T> units;

        /// Construct the \ref shamunits::Constants class with a unit system
        explicit Constants(const UnitSystem<T> units) : units(units) {}

/// Define the constant conversions functions
#define X(name, conv)                                                                              \
    addconstant(name) { return Cget(Si::name, 1) * conv; }
        UNITS_CONSTANTS
#undef X
    };

} // namespace shamunits

/**
 * \fn shamunits::Constants::delta_nu_cs()
 * \brief get delta_nu_cs in the current unit system @ref units (s-1)
 *
 * \fn shamunits::Constants::Si::delta_nu_cs()
 * \brief get delta_nu_cs in the si unit system (s-1)
 *
 * \fn shamunits::Constants::c()
 * \brief get c in the current unit system @ref units (m.s-1)
 *
 * \fn shamunits::Constants::Si::c()
 * \brief get c in the si unit system (m.s-1)
 *
 * \fn shamunits::Constants::h()
 * \brief get h in the current unit system @ref units (J.s-1)
 *
 * \fn shamunits::Constants::Si::h()
 * \brief get h in the si unit system (J.s-1)
 *
 * \fn shamunits::Constants::e()
 * \brief get e in the current unit system @ref units (C)
 *
 * \fn shamunits::Constants::Si::e()
 * \brief get e in the si unit system (C)
 *
 * \fn shamunits::Constants::k()
 * \brief get k in the current unit system @ref units (J.K-1 )
 *
 * \fn shamunits::Constants::Si::k()
 * \brief get k in the si unit system (J.K-1 )
 *
 * \fn shamunits::Constants::Na()
 * \brief get Na in the current unit system @ref units (mol-1 )
 *
 * \fn shamunits::Constants::Si::Na()
 * \brief get Na in the si unit system (mol-1 )
 *
 * \fn shamunits::Constants::Kcd()
 * \brief get Kcd in the current unit system @ref units (lm.W-1)
 *
 * \fn shamunits::Constants::Si::Kcd()
 * \brief get Kcd in the si unit system (lm.W-1)
 */

/**
 * \fn shamunits::Constants::G()
 * \brief get the value of G in the current unit system @ref units
 *
 * \fn shamunits::Constants::Si::G()
 * \brief get the value of G in the si unit system (N.m2.kg-2)
 *
 * \fn shamunits::Constants::hbar()
 * \brief get the value of hbar in the current unit system @ref units
 *
 * \fn shamunits::Constants::Si::hbar()
 * \brief get the value of hbar in the si unit system (J.s-1)
 *
 * \fn shamunits::Constants::mu_0()
 * \brief get the value of mu_0 in the current unit system @ref units
 *
 * \fn shamunits::Constants::Si::mu_0()
 * \brief get the value of mu_0 in the si unit system (N.A-2)
 *
 * \fn shamunits::Constants::Z_0()
 * \brief get the value of Z_0 in the current unit system @ref units
 *
 * \fn shamunits::Constants::Si::Z_0()
 * \brief get the value of Z_0 in the si unit system (Ohm)
 *
 * \fn shamunits::Constants::epsilon_0()
 * \brief get the value of epsilon_0 in the current unit system @ref units
 *
 * \fn shamunits::Constants::Si::epsilon_0()
 * \brief get the value of epsilon_0 in the si unit system (F.m-1)
 *
 * \fn shamunits::Constants::ke()
 * \brief get the value of ke in the current unit system @ref units
 *
 * \fn shamunits::Constants::Si::ke()
 * \brief get the value of ke in the si unit system (N.m2.C-2)
 *
 * \fn shamunits::Constants::kb()
 * \brief get the value of kb in the current unit system @ref units
 *
 * \fn shamunits::Constants::Si::kb()
 * \brief get the value of kb in the si unit system (J.K-1)
 *
 */

/**
 * \fn shamunits::Constants::sigma()
 * \brief get the value of sigma (Stephan Boltzmann constant) in the current unit system @ref units
 *
 * \fn shamunits::Constants::Si::sigma()
 * \brief get the value of sigma (Stephan Boltzmann constant) in the si unit system (W.m-2.K-4)
 */

/**
 * \fn shamunits::Constants::hour()
 * \brief get the value of a hour in the time unit of the current unit system @ref units
 *
 * \fn shamunits::Constants::Si::hour()
 * \brief get the value of a hour in the time unit of the si unit system (s)
 *
 * \fn shamunits::Constants::day()
 * \brief get the value of a day in the time unit of the current unit system @ref units
 *
 * \fn shamunits::Constants::Si::day()
 * \brief get the value of a day in the time unit of the si unit system (s)
 *
 * \fn shamunits::Constants::year()
 * \brief get the value of a year in the time unit of the current unit system @ref units
 *
 * \fn shamunits::Constants::Si::year()
 * \brief get the value of a year in the time unit of the si unit system (s)
 */

/**
 * \fn shamunits::Constants::astronomical_unit()
 * \brief get the value of an astronomical_unit in the distance unit of the current unit system
 * @ref units
 *
 * \fn shamunits::Constants::Si::astronomical_unit()
 * \brief get the value of an astronomical_unit in the distance unit of the si unit system (m)
 *
 * \fn shamunits::Constants::au()
 * \brief get the value of an au in the distance unit of the current unit system @ref units
 *
 * \fn shamunits::Constants::Si::au()
 * \brief get the value of an au in the distance unit of the si unit system (m)
 *
 * \fn shamunits::Constants::light_year()
 * \brief get the value of a light_year in the distance unit of the current unit system @ref units
 *
 * \fn shamunits::Constants::Si::light_year()
 * \brief get the value of a light_year in the distance unit of the si unit system (m)
 *
 * \fn shamunits::Constants::parsec()
 * \brief get the value of a parsec in the distance unit of the current unit system @ref units
 *
 * \fn shamunits::Constants::Si::parsec()
 * \brief get the value of a parsec in the distance unit of the si unit system (m)
 *
 * \fn shamunits::Constants::planck_length()
 * \brief get the value of a planck_length in the distance unit of the current unit system @ref
 * units
 *
 * \fn shamunits::Constants::Si::planck_length()
 * \brief get the value of a planck_length in the distance unit of the si unit system (m)
 */

/**
 * \fn shamunits::Constants::proton_mass()
 * \brief get the value of a proton_mass in the mass unit of the current unit system @ref units
 *
 * \fn shamunits::Constants::Si::proton_mass()
 * \brief get the value of a proton_mass in the mass unit of the si unit system (kg)
 *
 * \fn shamunits::Constants::electron_mass()
 * \brief get the value of a electron_mass in the mass unit of the current unit system @ref units
 *
 * \fn shamunits::Constants::Si::electron_mass()
 * \brief get the value of a electron_mass in the mass unit of the si unit system (kg)
 *
 * \fn shamunits::Constants::earth_mass()
 * \brief get the value of a earth_mass in the mass unit of the current unit system @ref units
 *
 * \fn shamunits::Constants::Si::earth_mass()
 * \brief get the value of a earth_mass in the mass unit of the si unit system (kg)
 *
 * \fn shamunits::Constants::jupiter_mass()
 * \brief get the value of a jupiter_mass in the mass unit of the current unit system @ref units
 *
 * \fn shamunits::Constants::Si::jupiter_mass()
 * \brief get the value of a jupiter_mass in the mass unit of the si unit system (kg)
 *
 * \fn shamunits::Constants::sol_mass()
 * \brief get the value of a sol_mass in the mass unit of the current unit system @ref units
 *
 * \fn shamunits::Constants::Si::sol_mass()
 * \brief get the value of a sol_mass in the mass unit of the si unit system (kg)
 *
 * \fn shamunits::Constants::planck_mass()
 * \brief get the value of a planck_mass in the mass unit of the current unit system @ref units
 *
 * \fn shamunits::Constants::Si::planck_mass()
 * \brief get the value of a planck_mass in the mass unit of the si unit system (kg)
 */

/**
 * \fn shamunits::Constants::guiness_density()
 * \brief get the value of the guiness density in the density unit of the current unit system @ref
 * units
 *
 * \fn shamunits::Constants::Si::guiness_density()
 * \brief get the value of a guiness_density in the density of the si unit system (kg.m-3)
 */

#undef Uget
#undef addconstant
