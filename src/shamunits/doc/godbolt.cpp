// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////
// shamrock units lib
// Go to the bottom of the file to try
////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////

namespace shamunits::details {

    template<int power, class T>
    inline constexpr T pow_constexpr_fast_inv(T a, T a_inv) noexcept {

        if constexpr (power < 0) {
            return pow_constexpr_fast_inv<-power>(a_inv, a);
        } else if constexpr (power == 0) {
            return T{1};
        } else if constexpr (power % 2 == 0) {
            T tmp = pow_constexpr_fast_inv<power / 2>(a, a_inv);
            return tmp * tmp;
        } else if constexpr (power % 2 == 1) {
            T tmp = pow_constexpr_fast_inv<(power - 1) / 2>(a, a_inv);
            return tmp * tmp * a;
        }
    }
} // namespace shamunits::details

#include <unordered_map>
#include <stdexcept>

#define XMAC_UNITS                                                                                 \
    /*base units*/                                                                                 \
    X1(second, s)                                                                                  \
    X1(metre, m)                                                                                   \
    X1(kilogramm, kg)                                                                              \
    X1(Ampere, A)                                                                                  \
    X1(Kelvin, K)                                                                                  \
    X1(mole, mol)                                                                                  \
    X1(candela, cd)                                                                                \
    /*derived units*/                                                                              \
    X1(Hertz, Hz)    /* hertz : frequency (s−1) */                                                 \
    X1(Newtown, N)   /* (kg⋅m⋅s−2)*/                                                               \
    X1(Pascal, Pa)   /* (kg⋅m−1⋅s−2) (N/m2)*/                                                      \
    X1(Joule, J)     /* (kg⋅m2⋅s−2) (N⋅m = Pa⋅m3)*/                                                \
    X1(Watt, W)      /* (kg⋅m2⋅s−3) (J/s)*/                                                        \
    X1(Coulomb, C)   /* (s⋅A)*/                                                                    \
    X1(Volt, V)      /* (kg⋅m2⋅s−3⋅A−1) (W/A) = (J/C)*/                                            \
    X1(Farad, F)     /* (kg−1⋅m−2⋅s4⋅A2) (C/V) = (C2/J)*/                                          \
    X1(Ohm, ohm)     /* (kg⋅m2⋅s−3⋅A−2) (V/A) = (J⋅s/C2)*/                                         \
    X1(Siemens, S)   /* (kg−1⋅m−2⋅s3⋅A2) (ohm−1)*/                                                 \
    X1(Weber, Wb)    /* (kg⋅m2⋅s−2⋅A−1) (V⋅s)*/                                                    \
    X1(Tesla, T)     /* (kg⋅s−2⋅A−1) (Wb/m2)*/                                                     \
    X1(Henry, H)     /* (kg⋅m2⋅s−2⋅A−2) (Wb/A)*/                                                   \
    X1(lumens, lm)   /* (cd.sr) (cd.sr)*/                                                          \
    X1(lux, lx)      /* (cd.sr.m−2) (lm/m2)*/                                                      \
    X1(Bequerel, Bq) /* (s−1)*/                                                                    \
    X1(Gray, Gy)     /* (m2.s−2) (J/kg)*/                                                          \
    X1(Sievert, Sv)  /* (m2.s−2) (J/kg)*/                                                          \
    X1(katal, kat)   /* (mol.s-1) */                                                               \
    /*relative units*/                                                                             \
    X1(minutes, mn)                                                                                \
    X1(hours, hr)                                                                                  \
    X1(days, dy)                                                                                   \
    X1(years, yr)                                                                                  \
    X1(astronomical_unit, au)                                                                      \
    X1(light_year, ly)                                                                             \
    X1(parsec, pc)                                                                                 \
    X1(electron_volt, eV)                                                                          \
    X1(ergs, erg)

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

    enum UnitPrefix {
#define X(longname, shortname, value) longname = value, shortname = value,
        XMAC_UNIT_PREFIX
#undef X
    };

    template<class T, UnitPrefix p>
    inline constexpr T get_prefix_val() {
        return details::pow_constexpr_fast_inv<p, T>(10, 1e-1);
    }

    namespace units {

        enum UnitName {
#define X1(longname, shortname) longname, shortname = longname,
            XMAC_UNITS
#undef X1
        };

    } // namespace units

} // namespace shamunits

namespace shamunits {

    template<class T>
    struct ConvertionConstants {

        static constexpr T au_to_m = 149597870700;     //(m)
        static constexpr T ly_to_m = 9460730472580800; //(m)
        static constexpr T pc_to_m = 3.0857e16;        //(m)

        static constexpr T mn_to_s  = 60;            //(s)
        static constexpr T hr_to_s  = 3600;          //(s)
        static constexpr T dy_to_s  = 24 * hr_to_s;  //(s)
        static constexpr T yr_to_s  = 31557600;      //(s)
        static constexpr T Myr_to_s = 1e6 * yr_to_s; //(s)
        static constexpr T Gyr_to_s = 1e9 * yr_to_s; //(s)

        static constexpr T eV_to_J  = 1.602176634e-19; // (J)
        static constexpr T erg_to_J = 1e-7;            // (J)

        static constexpr T K_degC_offset = 273.15;
    };

} // namespace shamunits

#include <cmath>

#define addget(uname)                                                                              \
    template<                                                                                      \
        UnitPrefix pref = None,                                                                    \
        units::UnitName u,                                                                         \
        int power                                = 1,                                              \
        std::enable_if_t<u == units::uname, int> = 0>                                              \
    inline constexpr T get() const noexcept

#define Uget(unitname, mult_pow) get<pref, units::unitname, (mult_pow) * power>()
#define Cget(constant_name, mult_pow)                                                              \
    details::pow_constexpr_fast_inv<(mult_pow) * power>(constant_name, T(1) / constant_name)
#define PREF Cget((get_prefix_val<T, pref>()), 1)

namespace shamunits {

    template<class T>
    class UnitSystem {

        template<int power>
        inline static constexpr T pow_constexpr(T a, T a_inv) noexcept {
            return details::pow_constexpr_fast_inv<power>(a, a_inv);
        }

        inline T pown(T a, int n) { return std::pow(a, n); }

        using Uconvert = ConvertionConstants<T>;

        public:
        T s, m, kg, A, K, mol, cd;
        T s_inv, m_inv, kg_inv, A_inv, K_inv, mol_inv, cd_inv;

        explicit UnitSystem(
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

        template<UnitPrefix pref = None, units::UnitName u, int power = 1>
        inline constexpr T to() {
            return get<u, -power>();
        }

        template<units::UnitName u, int power = 1>
        inline constexpr T get() {
            return get<None, u, power>();
        }

        template<units::UnitName u, int power = 1>
        inline constexpr T to() {
            return to<None, u, power>();
        }

        private:
        template<UnitPrefix pref = None>
        inline T getter_1(units::UnitName name) {
            switch (name) {

            case units::second: return get<pref, units::second>(); break;
            case units::metre: return get<pref, units::metre>(); break;
            case units::kilogramm: return get<pref, units::kilogramm>(); break;
            case units::Ampere: return get<pref, units::Ampere>(); break;
            case units::Kelvin: return get<pref, units::Kelvin>(); break;
            case units::mole: return get<pref, units::mole>(); break;
            case units::candela: return get<pref, units::candela>(); break;
            // case units::mps: return get<pref, units::mps>(); break;
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
            case units::erg: return get<pref, units::erg>(); break;
            }
        }

        inline T getter_2(UnitPrefix pref, units::UnitName name) {
            switch (pref) {

            case tera: return getter_1<tera>(name); break;
            case giga: return getter_1<giga>(name); break;
            case mega: return getter_1<mega>(name); break;
            case kilo: return getter_1<kilo>(name); break;
            case hecto: return getter_1<hecto>(name); break;
            case deca: return getter_1<deca>(name); break;
            case None: return getter_1<None>(name); break;
            // case deci: return getter_1<deci>(name); break;
            case centi: return getter_1<centi>(name); break;
            case milli: return getter_1<milli>(name); break;
            case micro: return getter_1<micro>(name); break;
            case nano: return getter_1<nano>(name); break;
            case pico: return getter_1<pico>(name); break;
            case femto: return getter_1<femto>(name); break;
            }
        }

        public:
        inline T runtime_get(UnitPrefix pref, units::UnitName name, int power) {
            return pown(getter_2(pref, name), power);
        }

        inline T runtime_to(UnitPrefix pref, units::UnitName name, int power) {
            return pown(getter_2(pref, name), -power);
        }
    };

} // namespace shamunits

#undef addget
#undef PREF
#undef Uget
#undef Cget

#define addconstant(name)                                                                          \
    template<int power = 1>                                                                        \
    inline constexpr T name()
#define Uget(unitname, mult_pow) units.template get<None, units::unitname, (mult_pow) * power>()
#define Cget(constant_name, mult_pow)                                                              \
    details::pow_constexpr_fast_inv<(mult_pow) * power>(constant_name, 1 / constant_name)

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
            static constexpr T planck_length     = 1.61625518e-35; //(m)

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

////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////
// user source starts here
////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////

#include <iostream>
int main(void) {

    using namespace shamunits;

    // create si units
    UnitSystem<double> si{};

    // get the value of au^2 in the unit system
    // but it is quite big :)
    std::cout << si.get<units::astronomical_unit, 2>() << std::endl;

    double sol_mass = Constants<double>(si).sol_mass();

    /*
     * create a unit system with time in Myr, length in au, mass in solar masses
     */
    UnitSystem<double> astro_units{
        si.get<mega, units::years>(),
        si.get<units::astronomical_unit>(),
        si.get<units::kilogramm>() * sol_mass,
    };

    // this time it returns 1 because the base length is the astronomical unit
    std::cout << astro_units.get<units::astronomical_unit, 2>() << std::endl;

    Constants<double> astro_cte{astro_units};

    // in those units G is 3.94781e+25
    std::cout << astro_cte.G() << std::endl;

    // now if the code return a value in astro_units
    // we can convert it to any units like so
    double value = 12; // here 12 Myr

    // print : value = 3.15576e+19 s
    std::cout << "value = " << astro_units.to<units::second>() << " s" << std::endl;
}
