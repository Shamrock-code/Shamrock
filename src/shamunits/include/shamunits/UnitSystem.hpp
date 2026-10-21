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
 * @file UnitSystem.hpp
 * @author David Fang (david.fang@ikmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

#include "ConversionConstants.hpp"
#include "Names.hpp"
#include <cmath>

/// Macro to add a unit getter
#define addget(uname)                                                                              \
    template<                                                                                      \
        UnitPrefix pref = None,                                                                    \
        legacy_units::UnitName u,                                                                  \
        int power                                       = 1,                                       \
        std::enable_if_t<u == legacy_units::uname, int> = 0>                                       \
    inline constexpr T get() const noexcept

/// Shorthand for unit getter
#define Uget(unitname, mult_pow) get<None, legacy_units::unitname, (mult_pow) * power>()
/// Shorthand for constant getter
#define Cget(constant_name, mult_pow)                                                              \
    details::pow_constexpr_fast_inv<(mult_pow) * power>(constant_name, T(1) / constant_name)
/// Shorthand for prefix handling
#define PREF Cget((get_prefix_val<T, pref>()), 1)

namespace shamunits {

    /// \brief Defines a unit system
    ///
    /// Example of usage :
    /// \code {.cpp}
    ///    using namespace shamunits;
    ///
    ///    //create si units
    ///    UnitSystem<double> si {};
    ///
    ///    // get the value of au^2 in the unit system
    ///    // but it is quite big :)
    ///    std::cout << si.get<legacy_units::astronomical_unit,2>() << std::endl;
    ///
    ///    double sol_mass = Constants<double>(si).sol_mass();
    ///
    ///    /*
    ///    * create a unit system with time in Myr, length in au, mass in solar masses
    ///    */
    ///    UnitSystem<double> astro_units {
    ///        si.get<mega, legacy_units::years>(),
    ///        si.get<legacy_units::astronomical_unit>(),
    ///        si.get<legacy_units::kilogram>()*sol_mass,
    ///    };
    ///
    ///    //this time it returns 1 because the base length is the astronomical unit
    ///    std::cout << astro_units.get<legacy_units::astronomical_unit,2>() << std::endl;
    ///
    ///    Constants<double> astro_cte {astro_units};
    ///
    ///    // in those units G is 3.94781e+13
    ///    std::cout << astro_cte.G() << std::endl;
    /// @endcode
    ///
    ///
    template<class T>
    class UnitSystem {

        template<int power>
        inline static constexpr T pow_constexpr(T a, T a_inv) noexcept {
            return details::pow_constexpr_fast_inv<power>(a, a_inv);
        }

        inline T pown(T a, int n) { return std::pow(a, n); }

        using Uconvert = ConversionConstants<T>;

        public:
        /**
         * @fn s
         * \brief value of a second in this system
         *
         * @fn m
         * \brief value of a meter in this system
         *
         * @fn kg
         * \brief value of a kilogram in this system
         *
         * @fn A
         * \brief value of a Ampere in this system
         *
         * @fn K
         * \brief value of a Kelvin in this system
         *
         * @fn mol
         * \brief value of a mol in this system
         *
         * @fn cd
         * \brief value of a candela in this system
         *
         * @fn s_inv
         * \brief inverse value of a second in this system
         *
         * @fn m_inv
         * \brief inverse value of a meter in this system
         *
         * @fn kg_inv
         * \brief inverse value of a kilogram in this system
         *
         * @fn A_inv
         * \brief inverse value of a Ampere in this system
         *
         * @fn K_inv
         * \brief inverse value of a Kelvin in this system
         *
         * @fn mol_inv
         * \brief inverse value of a mol in this system
         *
         * @fn cd_inv
         * \brief inverse value of a candela in this system
         *
         */

        T s, m, kg, A, K, mol, cd;
        T s_inv, m_inv, kg_inv, A_inv, K_inv, mol_inv, cd_inv;

        /**
         * @brief Construct a new UnitSystem from either supplied base units or SI by default
         *
         * @param unit_time
         * @param unit_length
         * @param unit_mass
         * @param unit_current
         * @param unit_temperature
         * @param unit_qte
         * @param unit_lumint
         */
        constexpr explicit UnitSystem(
            T unit_time        = 1,
            T unit_length      = 1,
            T unit_mass        = 1,
            T unit_current     = 1,
            T unit_temperature = 1,
            T unit_qte         = 1,
            T unit_lumint      = 1)
            : s(1 / unit_time), m(1 / unit_length), kg(1 / unit_mass), A(1 / unit_current),
              K(1 / unit_temperature), mol(1 / unit_qte), cd(1 / unit_lumint), s_inv(unit_time),
              m_inv(unit_length), kg_inv(unit_mass), A_inv(unit_current), K_inv(unit_temperature),
              mol_inv(unit_qte), cd_inv(unit_lumint) {}

#ifdef DOXYGEN
        /**
         * @brief get the value of a units in the current system
         * \todo better doc with example
         * @tparam pref unit prefix (default to none)
         * @tparam u the units
         * @tparam power the power of the units (also applied to the prefix)
         * @return constexpr T the value in this system
         */
        template<UnitPrefix pref = None, legacy_units::UnitName u, int power = 1>
        inline constexpr T get() const noexcept;
#else

        // clang-format off
        addget(second)    { return PREF* pow_constexpr<power>(s  , s_inv);   }
        addget(metre)     { return PREF* pow_constexpr<power>(m  , m_inv);   }
        addget(kilogram)  { return PREF* pow_constexpr<power>(kg , kg_inv);  }
        addget(Ampere)    { return PREF* pow_constexpr<power>(A  , A_inv);   }
        addget(Kelvin)    { return PREF* pow_constexpr<power>(K  , K_inv);   }
        addget(mole)      { return PREF* pow_constexpr<power>(mol, mol_inv); }
        addget(candela)   { return PREF* pow_constexpr<power>(cd , cd_inv);  }

        addget(Hertz)   { return PREF* Uget(s, -1); }
        //addget(mps)     { return PREF* Uget(m, 1)       * Uget(s, -1); }
        addget(Newton) { return PREF* Uget(kg, 1)      * Uget(m, 1)  * Uget(s, -2); }
        addget(Pascal)  { return PREF* Uget(kg, 1)      * Uget(m, -1) * Uget(s, -2); }
        addget(Joule)   { return PREF* Uget(Newton, 1) * Uget(m, 1); }
        addget(Watt)    { return PREF* Uget(Joule, 1)   * Uget(s, -1); }
        addget(Coulomb) { return PREF* Uget(s, 1)       * Uget(A, 1); }
        addget(Volt)    { return PREF* Uget(Watt, 1)    * Uget(A, -1); }
        addget(Farad)   { return PREF* Uget(Coulomb, 1) * Uget(Volt, -1); }
        addget(Ohm)     { return PREF* Uget(Volt, 1)    * Uget(Ampere, -1); }
        addget(Siemens) { return PREF* Uget(Ohm, -1); }
        addget(Weber)   { return PREF* Uget(Volt, 1)    * Uget(second, 1); }
        addget(Tesla)   { return PREF* Uget(Weber, 1)   * Uget(m, -2); }
        addget(Henry)   { return PREF* Uget(Weber, 1)   * Uget(A, -1); }
        addget(lumens)  { return PREF* Uget(candela, 1); }
        addget(lux)     { return PREF* Uget(lumens, 1)  * Uget(m, -2); }
        addget(Bequerel){ return PREF* Uget(s, -1); }
        addget(Gray)    { return PREF* Uget(m, 2)       *Uget(s, -2)  ; }
        addget(Sievert) { return PREF* Uget(m, 2)       *Uget(s, -2)  ; }
        addget(katal)   { return PREF* Uget(mol, 1)     *Uget(s, -1)  ; }


        // alternative base units
        addget(minutes){ return PREF* Uget(s, 1) * Cget(Uconvert::mn_to_s, 1); }
        addget(hours)  { return PREF* Uget(s, 1) * Cget(Uconvert::hr_to_s, 1); }
        addget(days)   { return PREF* Uget(s, 1) * Cget(Uconvert::dy_to_s, 1); }
        addget(years)  { return PREF* Uget(s, 1) * Cget(Uconvert::yr_to_s, 1); }

        addget(astronomical_unit) { return PREF* Uget(m, 1) * Cget(Uconvert::au_to_m, 1); }
        addget(light_year)        { return PREF* Uget(m, 1) * Cget(Uconvert::ly_to_m, 1); }
        addget(parsec)            { return PREF* Uget(m, 1) * Cget(Uconvert::pc_to_m, 1); }
        addget(solar_radius)      { return PREF* Uget(m, 1) * Cget(Uconvert::rsol_to_m, 1); }
        addget(earth_radius)      { return PREF* Uget(m, 1) * Cget(Uconvert::rearth_to_m, 1); }
        addget(solar_mass)        { return PREF* Uget(kg, 1) * Cget(Uconvert::sol_mass_to_kg, 1); }

        addget(eV) {return PREF* Uget(Joule, 1) * Cget(Uconvert::eV_to_J,1);}
        addget(erg) {return PREF* Uget(Joule, 1) * Cget(Uconvert::erg_to_J,1);}

        addget(pint) {return PREF* Uget(m, 3) * Cget(Uconvert::litre_to_pint/1000,1);}
        #endif

        // clang-format on

        /**
         * @brief To the si units (convert a system unit to si ones). This function is the inverse
         * of get \todo better doc with example
         * @tparam pref unit prefix (default to none)
         * @tparam u the units
         * @tparam power the power of the units (also applied to the prefix)
         * @return constexpr T the value in si units
         */
        template<UnitPrefix pref = None, legacy_units::UnitName u, int power = 1>
        inline constexpr T to() {
            return get<pref, u, -power>();
        }

        /// Simplified signature for \ref get without prefix
        template<legacy_units::UnitName u, int power = 1>
        inline constexpr T get() {
            return get<None, u, power>();
        }

        /// Simplified signature for \ref to without prefix
        template<legacy_units::UnitName u, int power = 1>
        inline constexpr T to() {
            return to<None, u, power>();
        }

        private:
        template<UnitPrefix pref = None>
        inline T getter_1(legacy_units::UnitName name) {
            switch (name) {

            case legacy_units::second  : return get<pref, legacy_units::second>(); break;
            case legacy_units::metre   : return get<pref, legacy_units::metre>(); break;
            case legacy_units::kilogram: return get<pref, legacy_units::kilogram>(); break;
            case legacy_units::Ampere  : return get<pref, legacy_units::Ampere>(); break;
            case legacy_units::Kelvin  : return get<pref, legacy_units::Kelvin>(); break;
            case legacy_units::mole    : return get<pref, legacy_units::mole>(); break;
            case legacy_units::candela : return get<pref, legacy_units::candela>(); break;
            // case legacy_units::mps: return get<pref, legacy_units::mps>(); break;
            case legacy_units::Hertz   : return get<pref, legacy_units::Hertz>(); break;
            case legacy_units::Newton  : return get<pref, legacy_units::Newton>(); break;
            case legacy_units::Pascal  : return get<pref, legacy_units::Pascal>(); break;
            case legacy_units::Joule   : return get<pref, legacy_units::Joule>(); break;
            case legacy_units::Watt    : return get<pref, legacy_units::Watt>(); break;
            case legacy_units::Coulomb : return get<pref, legacy_units::Coulomb>(); break;
            case legacy_units::Volt    : return get<pref, legacy_units::Volt>(); break;
            case legacy_units::Farad   : return get<pref, legacy_units::Farad>(); break;
            case legacy_units::Ohm     : return get<pref, legacy_units::Ohm>(); break;
            case legacy_units::Siemens : return get<pref, legacy_units::Siemens>(); break;
            case legacy_units::Weber   : return get<pref, legacy_units::Weber>(); break;
            case legacy_units::Tesla   : return get<pref, legacy_units::Tesla>(); break;
            case legacy_units::Henry   : return get<pref, legacy_units::Henry>(); break;
            case legacy_units::lumens  : return get<pref, legacy_units::lumens>(); break;
            case legacy_units::lux     : return get<pref, legacy_units::lux>(); break;
            case legacy_units::Bequerel: return get<pref, legacy_units::Bequerel>(); break;
            case legacy_units::Gray    : return get<pref, legacy_units::Gray>(); break;
            case legacy_units::Sievert : return get<pref, legacy_units::Sievert>(); break;
            case legacy_units::katal   : return get<pref, legacy_units::katal>(); break;
            case legacy_units::minutes : return get<pref, legacy_units::minutes>(); break;
            case legacy_units::hours   : return get<pref, legacy_units::hours>(); break;
            case legacy_units::days    : return get<pref, legacy_units::days>(); break;
            case legacy_units::years   : return get<pref, legacy_units::years>(); break;
            case legacy_units::astronomical_unit:
                return get<pref, legacy_units::astronomical_unit>();
                break;
            case legacy_units::light_year  : return get<pref, legacy_units::light_year>(); break;
            case legacy_units::parsec      : return get<pref, legacy_units::parsec>(); break;
            case legacy_units::solar_radius: return get<pref, legacy_units::solar_radius>(); break;
            case legacy_units::earth_radius: return get<pref, legacy_units::earth_radius>(); break;
            case legacy_units::solar_mass  : return get<pref, legacy_units::solar_mass>(); break;
            case legacy_units::eV          : return get<pref, legacy_units::eV>(); break;
            case legacy_units::erg         : return get<pref, legacy_units::erg>(); break;
            case legacy_units::pint        : return get<pref, legacy_units::pint>(); break;
            }
        }

        inline T getter_2(UnitPrefix pref, legacy_units::UnitName name) {
            switch (pref) {

            case tera : return getter_1<tera>(name); break;
            case giga : return getter_1<giga>(name); break;
            case mega : return getter_1<mega>(name); break;
            case kilo : return getter_1<kilo>(name); break;
            case hecto: return getter_1<hecto>(name); break;
            case deca : return getter_1<deca>(name); break;
            case None : return getter_1<None>(name); break;
            // case deci: return getter_1<deci>(name); break;
            case centi: return getter_1<centi>(name); break;
            case milli: return getter_1<milli>(name); break;
            case micro: return getter_1<micro>(name); break;
            case nano : return getter_1<nano>(name); break;
            case pico : return getter_1<pico>(name); break;
            case femto: return getter_1<femto>(name); break;
            }
        }

        public:
        /// Non templated version of \ref get
        inline T runtime_get(UnitPrefix pref, legacy_units::UnitName name, int power) {
            return pown(getter_2(pref, name), power);
        }

        /// Non templated version of \ref to
        inline T runtime_to(UnitPrefix pref, legacy_units::UnitName name, int power) {
            return pown(getter_2(pref, name), -power);
        }
    };

} // namespace shamunits

#undef addget
#undef PREF
#undef Uget
#undef Cget
