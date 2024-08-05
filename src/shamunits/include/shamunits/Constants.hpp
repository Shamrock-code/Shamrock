// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file Constants.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "ConvertionConstants.hpp"
#include "Names.hpp"
#include "UnitSystem.hpp"
#include "details/utils.hpp"
#include <stdexcept>

/// @brief Macro to add a constant to the \ref shamunits::Constants class
/// For exemple one can add a new constant by adding the following statment inside
/// shamunits::Constants \code{.cpp} addconstant(delta_nu_cs) { return Cget(Si::delta_nu_cs,1) *
/// Uget(Hertz, 1); } \endcode
///
#define addconstant(name)                                                                          \
    template<int power = 1>                                                                        \
    inline constexpr T name()

/// Utility macro to get the value of a unit in the current unit system with its power
#define Uget(unitname, mult_pow) units.template get<None, units::unitname, (mult_pow) * power>()

/// Utility macro to get the value of a constant in the current unit system with its power
#define Cget(constant_name, mult_pow)                                                              \
    details::pow_constexpr_fast_inv<(mult_pow) * power>(constant_name, 1 / constant_name)

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
            static constexpr T hbar      = 1.054571817e-34;             // (J.s-1)
            static constexpr T mu_0      = 1.2566370621219e-6;          //
            static constexpr T Z_0       = mu_0 * c;                    //
            static constexpr T epsilon_0 = 1 / (Z_0 * c);               //
            static constexpr T ke        = 1 / (4 * pi<T> * epsilon_0); //

            static constexpr T hour = Conv::hr_to_s; //(s)
            static constexpr T day  = Conv::dy_to_s; //(s)
            static constexpr T year = Conv::yr_to_s; //(s)

            static constexpr T astronomical_unit = Conv::au_to_m;  //(m)
            static constexpr T light_year        = Conv::ly_to_m;  //(m)
            static constexpr T parsec            = Conv::pc_to_m;  //(m)
            static constexpr T planck_length     = 1.61625518e-35; //(m)

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

        // clang-format off
        
        addconstant(delta_nu_cs) { return Cget(Si::delta_nu_cs,1) * Uget(Hertz, 1); }
        addconstant(c)           { return Cget(Si::c,1)   * Uget(m, 1)* Uget(s, -1); }
        addconstant(h)           { return Cget(Si::h,1)   * Uget(Joule, 1) * Uget(s, -1); }
        addconstant(e)           { return Cget(Si::e,1)   * Uget(Coulomb, 1); }
        addconstant(k)           { return Cget(Si::k,1)   * Uget(Joule, 1) * Uget(Kelvin, -1); }
        addconstant(Na)          { return Cget(Si::Na,1)  * Uget(mole, -1); }
        addconstant(Kcd)         { return Cget(Si::Kcd,1) * Uget(lm, 1)    * Uget(Watt, -1); }

        
        addconstant(year)         { return Cget(Si::year,1) * Uget(s,1) ; }

        addconstant(au)         { return Cget(Si::astronomical_unit,1) * Uget(s,1) ; }

        addconstant(G)         { return Cget(Si::G,1) * Uget(N,1) * Uget(m,2) * Uget(kg,-2)  ; }

        addconstant(earth_mass)         { return Cget(Si::earth_mass,1) * Uget(kg,1) ; }
        addconstant(jupiter_mass)         { return Cget(Si::jupiter_mass,1) * Uget(kg,1) ; }
        addconstant(sol_mass)         { return Cget(Si::sol_mass,1) * Uget(kg,1) ; }
        addconstant(guiness_density)         { return Cget(Si::guiness_density,1) *Uget(kg,1)* Uget(m,-3) ; }

        // clang-format on
    };

} // namespace shamunits

/**
 * \fn shamunits::Constants::delta_nu_cs()
 * \brief get delta_nu_cs in the current unit system @ref units
 *
 * \fn shamunits::Constants::Si::delta_nu_cs()
 * \brief get delta_nu_cs in the si unit system
 *
 * \fn shamunits::Constants::c()
 * \brief get c in the current unit system @ref units
 *
 * \fn shamunits::Constants::Si::c()
 * \brief get c in the si unit system
 *
 * \fn shamunits::Constants::h()
 * \brief get h in the current unit system @ref units
 *
 * \fn shamunits::Constants::Si::h()
 * \brief get h in the si unit system
 *
 * \fn shamunits::Constants::e()
 * \brief get e in the current unit system @ref units
 *
 * \fn shamunits::Constants::Si::e()
 * \brief get e in the si unit system
 *
 * \fn shamunits::Constants::k()
 * \brief get k in the current unit system @ref units
 *
 * \fn shamunits::Constants::Si::k()
 * \brief get k in the si unit system
 *
 * \fn shamunits::Constants::Na()
 * \brief get Na in the current unit system @ref units
 *
 * \fn shamunits::Constants::Si::Na()
 * \brief get Na in the si unit system
 *
 * \fn shamunits::Constants::Kcd()
 * \brief get Kcd in the current unit system @ref units
 *
 * \fn shamunits::Constants::Si::Kcd()
 * \brief get Kcd in the si unit system
 *
 * \fn shamunits::Constants::year()
 * \brief get the value of a year in the time unit of the current unit system @ref units
 *
 * \fn shamunits::Constants::Si::year()
 * \brief get the value of a year in the time unit of the si unit system
 *
 * \fn shamunits::Constants::au()
 * \brief get the value of an au in the distance unit of the current unit system @ref units
 *
 * \fn shamunits::Constants::Si::astronomical_unit()
 * \brief get the value of an au in the distance unit of the si unit system
 *
 * \fn shamunits::Constants::G()
 * \brief get the value of G in the current unit system @ref units
 *
 * \fn shamunits::Constants::Si::G()
 * \brief get the value of G in the si unit system
 *
 * \fn shamunits::Constants::earth_mass()
 * \brief get the value of a earth_mass in the mass unit of the current unit system @ref units
 *
 * \fn shamunits::Constants::Si::earth_mass()
 * \brief get the value of a earth_mass in the mass unit of the si unit system
 *
 * \fn shamunits::Constants::jupiter_mass()
 * \brief get the value of a jupiter_mass in the mass unit of the current unit system @ref units
 *
 * \fn shamunits::Constants::Si::jupiter_mass()
 * \brief get the value of a jupiter_mass in the mass unit of the si unit system
 *
 * \fn shamunits::Constants::sol_mass()
 * \brief get the value of a sol_mass in the mass unit of the current unit system @ref units
 *
 * \fn shamunits::Constants::Si::sol_mass()
 * \brief get the value of a sol_mass in the mass unit of the si unit system
 *
 * \fn shamunits::Constants::guiness_density()
 * \brief get the value of the guiness density in the density unit of the current unit system @ref
 * units
 *
 * \fn shamunits::Constants::Si::guiness_density()
 * \brief get the value of a guiness_density in the density of the si unit system
 *
 * \fn shamunits::Constants::Si::hbar()
 * \brief get the value of a hbar in the si unit system
 *
 * \fn shamunits::Constants::Si::mu_0()
 * \brief get the value of a mu_0 in the si unit system
 *
 * \fn shamunits::Constants::Si::Z_0()
 * \brief get the value of a Z_0 in the si unit system
 *
 * \fn shamunits::Constants::Si::epsilon_0()
 * \brief get the value of a epsilon_0 in the si unit system
 *
 * \fn shamunits::Constants::Si::ke()
 * \brief get the value of a ke in the si unit system
 *
 * \fn shamunits::Constants::Si::hour()
 * \brief get the duration of an hour in the si unit system
 *
 * \fn shamunits::Constants::Si::day()
 * \brief get the duration of a day in the si unit system
 *
 * \fn shamunits::Constants::Si::light_year()
 * \brief get the lenght of a light_year in the si unit system
 *
 * \fn shamunits::Constants::Si::parsec()
 * \brief get the lenght of a parsec in the si unit system
 *
 * \fn shamunits::Constants::Si::planck_length()
 * \brief get the lenght of a planck_length in the si unit system
 *
 * \fn shamunits::Constants::Si::proton_mass()
 * \brief get the mass of a proton in the si unit system
 *
 * \fn shamunits::Constants::Si::electron_mass()
 * \brief get the mass of a electron in the si unit system
 *
 * \fn shamunits::Constants::Si::planck_mass()
 * \brief get the value of a planck mass in the si unit system
 *
 */

#undef Uget
#undef addconstant
