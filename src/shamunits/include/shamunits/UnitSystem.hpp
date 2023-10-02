// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "Names.hpp"
#include "ConvertionConstants.hpp"
#include <cmath>

#define addget(uname)                                                                              \
    template<UnitPrefix pref = None,                                                               \
             units::UnitName u,                                                                    \
             int power                                = 1,                                         \
             std::enable_if_t<u == units::uname, int> = 0>                                         \
    inline constexpr T get() const noexcept

#define Uget(unitname, mult_pow) get<pref,units::unitname, (mult_pow)*power>()
#define Cget(constant_name, mult_pow)                                                              \
    details::pow_constexpr_fast_inv<(mult_pow)*power>(constant_name, T(1) / constant_name)
#define PREF Cget((get_prefix_val<T,pref>()), 1)

namespace shamunits {

    template<class T>
    class UnitSystem {

        template<int power>
        inline static constexpr T pow_constexpr(T a, T a_inv) noexcept {
            return details::pow_constexpr_fast_inv<power>(a, a_inv);
        }

        inline T pown(T a, int n){
            return std::pow(a,n);
        }

        using Uconvert = ConvertionConstants<T>;

        public:

        T s, m, kg, A, K, mol, cd;
        T s_inv, m_inv, kg_inv, A_inv, K_inv, mol_inv, cd_inv;

        explicit UnitSystem(T unit_time = 1 ,
                   T unit_lenght = 1 ,
                   T unit_mass = 1 ,
                   T unit_current = 1 ,
                   T unit_temperature = 1 ,
                   T unit_qte = 1 ,
                   T unit_lumint = 1 )
            : s(1 / unit_time), m(1 / unit_lenght), kg(1 / unit_mass), A(1 / unit_current),
              K(1 / unit_temperature), mol(1 / unit_qte), cd(1 / unit_lumint), s_inv(unit_time),
              m_inv(unit_lenght), kg_inv(unit_mass), A_inv(unit_current), K_inv(unit_temperature),
              mol_inv(unit_qte), cd_inv(unit_lumint) {}

        // clang-format off
        addget(second)    { return PREF* pow_constexpr<power>(s  , s_inv);   }
        addget(metre)     { return PREF* pow_constexpr<power>(m  , m_inv);   }
        addget(kilogramm) { return PREF* pow_constexpr<power>(kg , kg_inv);  }
        addget(Ampere)    { return PREF* pow_constexpr<power>(A  , A_inv);   }
        addget(Kelvin)    { return PREF* pow_constexpr<power>(K  , K_inv);   }
        addget(mole)      { return PREF* pow_constexpr<power>(mol, mol_inv); }
        addget(candela)   { return PREF* pow_constexpr<power>(cd , cd_inv);  }
        
        addget(Hertz)   { return PREF* Uget(s, -1); }
        //addget(mps)     { return PREF* Uget(m, 1)       * Uget(s, -1); }
        addget(Newtown) { return PREF* Uget(kg, 1)      * Uget(m, 1)  * Uget(s, -2); }
        addget(Pascal)  { return PREF* Uget(kg, 1)      * Uget(m, -1) * Uget(s, -2); }
        addget(Joule)   { return PREF* Uget(Newtown, 1) * Uget(m, 1); }
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

        addget(eV) {return PREF* Uget(Joule, 1) * Cget(Uconvert::eV_to_J,1);}
        addget(erg) {return PREF* Uget(Joule, 1) * Cget(Uconvert::erg_to_J,1);}

        // clang-format on

        template<UnitPrefix pref = None,                                                             
             units::UnitName u,                                                                    
             int power                                = 1>  
        inline constexpr T to() {
            return get<u, -power>();
        }

        template<                                                             
             units::UnitName u,                                                                    
             int power                                = 1>  
        inline constexpr T get(){
            return get<None,u,power>();
        }

        private:

        template<UnitPrefix pref = None>
        inline T getter_1(units::UnitName name){
            switch (name) {

            case units::second: return get<pref, units::second>(); break;
            case units::metre: return get<pref, units::metre>(); break;
            case units::kilogramm: return get<pref, units::kilogramm>(); break;
            case units::Ampere: return get<pref, units::Ampere>(); break;
            case units::Kelvin: return get<pref, units::Kelvin>(); break;
            case units::mole: return get<pref, units::mole>(); break;
            case units::candela: return get<pref, units::candela>(); break;
            //case units::mps: return get<pref, units::mps>(); break;
            case units::Hertz: return get<pref, units::Hertz>(); break;
            case units::Newtown: return get<pref, units::Newtown>(); break;
            case units::Pascal: return get<pref, units::Pascal>(); break;
            case units::Joule: return get<pref, units::Joule>(); break;
            case units::Watt: return get<pref, units::Watt>(); break;
            case units::Coulomb: return get<pref, units::Coulomb>(); break;
            case units::Volt: return get<pref, units::Volt>(); break;
            case units::Farad: return get<pref, units::Farad>(); break;
            case units::Ohm: return get<pref, units::Ohm>(); break;
            case units::Siemens: return get<pref, units::Siemens>(); break;
            case units::Weber: return get<pref, units::Weber>(); break;
            case units::Tesla: return get<pref, units::Tesla>(); break;
            case units::Henry: return get<pref, units::Henry>(); break;
            case units::lumens: return get<pref, units::lumens>(); break;
            case units::lux: return get<pref, units::lux>(); break;
            case units::Bequerel: return get<pref, units::Bequerel>(); break;
            case units::Gray: return get<pref, units::Gray>(); break;
            case units::Sievert: return get<pref, units::Sievert>(); break;
            case units::katal: return get<pref, units::katal>(); break;
            case units::minutes: return get<pref, units::minutes>(); break;
            case units::hours: return get<pref, units::hours>(); break;
            case units::days: return get<pref, units::days>(); break;
            case units::years: return get<pref, units::years>(); break;
            case units::astronomical_unit: return get<pref, units::astronomical_unit>(); break;
            case units::light_year: return get<pref, units::light_year>(); break;
            case units::parsec: return get<pref, units::parsec>(); break;
            case units::eV: return get<pref, units::eV>(); break;
            case units::erg:  return get<pref, units::erg>(); break;
            }
        }

        inline T getter_2(UnitPrefix pref ,units::UnitName name){
            switch (pref) {

            case tera: return getter_1<tera>(name); break;
            case giga: return getter_1<giga>(name); break;
            case mega: return getter_1<mega>(name); break;
            case kilo: return getter_1<kilo>(name); break;
            case hecto: return getter_1<hecto>(name); break;
            case deca: return getter_1<deca>(name); break;
            case None: return getter_1<None>(name); break;
            //case deci: return getter_1<deci>(name); break;
            case centi: return getter_1<centi>(name); break;
            case milli: return getter_1<milli>(name); break;
            case micro: return getter_1<micro>(name); break;
            case nano: return getter_1<nano>(name); break;
            case pico: return getter_1<pico>(name); break;
            case femto:  return getter_1<femto>(name); break;
            }
        }

        public:

        inline T runtime_get(UnitPrefix pref ,units::UnitName name, int power){
            return pown(getter_2(pref, name),power);
        }

        inline T runtime_to(UnitPrefix pref ,units::UnitName name, int power){
            return pown(getter_2(pref, name),-power);
        }
    };

} // namespace shamunits

#undef addget
#undef PREF
#undef Uget
#undef Cget