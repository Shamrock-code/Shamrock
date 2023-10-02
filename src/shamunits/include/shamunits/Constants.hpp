// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "ConvertionConstants.hpp"
#include "Names.hpp"
#include "UnitSystem.hpp"
#include <stdexcept>
#include "details/utils.hpp"

#define addconstant(name)                                                                          \
    template<int power = 1>                                                                        \
    inline constexpr T name()
#define Uget(unitname, mult_pow) units.template get<None, units::unitname, (mult_pow)*power>()
#define Cget(constant_name, mult_pow)                                                              \
    details::pow_constexpr_fast_inv<(mult_pow)*power>(constant_name, 1 / constant_name)

namespace shamunits {

    template<class T>
    constexpr T pi = 3.141592653589793116;
    template<class T>
    constexpr T fine_structure = 0.0072973525693;
    template<class T>
    constexpr T proton_electron_ratio = 1836.1526734311;
    template<class T>
    constexpr T electron_proton_ratio = 1 / proton_electron_ratio<T>;

    template<class T>
    struct Constants {

        using Conv = ConvertionConstants<T>;

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
            static constexpr T planck_lenght     = 1.61625518e-35; //(m)

            static constexpr T proton_mass   = 1.67262192e-27;                         //(kg)
            static constexpr T electron_mass = proton_mass * electron_proton_ratio<T>; //(kg)
            static constexpr T earth_mass    = 5.9722e24;                              //(kg)
            static constexpr T jupiter_mass  = 1.898e27;                               //(kg)
            static constexpr T sol_mass      = 1.98847e30;                             //(kg)
            static constexpr T planck_mass   = 2.17643424e-8;                          //(kg)
        };

        const UnitSystem<T> units;
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

        // clang-format on
    };

} // namespace shamunits

#undef Uget
#undef addconstant