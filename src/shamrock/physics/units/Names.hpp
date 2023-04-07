// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

#include "shambase/floats.hpp"
namespace shamrock {

    enum UnitPrefix {
        tera  = 12,  // e12
        giga  = 9,   // e9
        mega  = 6,   // e6
        kilo  = 3,   // e3
        hecto = 2,   // e2
        deca  = 1,   // e1
        None  = 0,   // 1
        deci  = -1,  // e-1
        centi = -2,  // e-2
        milli = -3,  // e-3
        micro = -6,  // e-6
        nano  = -9,  // e-9
        pico  = -12, // e-12
        femto = -15, // e-15
    };

    template<class T>
    inline constexpr T get_prefix_val(UnitPrefix p) {
        return shambase::pow_constexpr_fast_inv<p, T>(10, 1e-1);
    }

    inline const std::string get_prefix_str(UnitPrefix p) {
        switch (p) {
        case tera: return "T"; break;
        case giga: return "G"; break;
        case mega: return "M"; break;
        case kilo: return "k"; break;
        case hecto: return "x100"; break;
        case deca: return "x10"; break;
        case None: return ""; break;
        case deci: return "/10"; break;
        case centi: return "c"; break;
        case milli: return "m"; break;
        case micro: return "mu"; break;
        case nano: return "n"; break;
        case pico: return "p"; break;
        case femto: return "f"; break;
        }
        return "";
    }

    namespace units {

        enum UnitName {

            /*
             * Base si units
             */
            second,
            metre,
            kilogramm,
            Ampere,
            Kelvin,
            mole,
            candela,

            s   = second,
            m   = metre,
            kg  = kilogramm,
            A   = Ampere,
            K   = Kelvin,
            mol = mole,
            cd  = candela,

            /*
             * si derived units
             */

            mps, ///< meter per second (m.s-1)

            Hertz,    ///< hertz : frequency (s−1)
            Newtown,  ///< (kg⋅m⋅s−2)
            Pascal,   ///< (kg⋅m−1⋅s−2) 	(N/m2)
            Joule,    ///< (kg⋅m2⋅s−2) 	(N⋅m = Pa⋅m3)
            Watt,     ///< (kg⋅m2⋅s−3) 	(J/s)
            Coulomb,  ///< (s⋅A)
            Volt,     ///< (kg⋅m2⋅s−3⋅A−1) 	(W/A) = (J/C)
            Farad,    ///< (kg−1⋅m−2⋅s4⋅A2) 	(C/V) = (C2/J)
            Ohm,      ///< (kg⋅m2⋅s−3⋅A−2) 	(V/A) = (J⋅s/C2)
            Siemens,  ///< (kg−1⋅m−2⋅s3⋅A2) 	(ohm−1)
            Weber,    ///< (kg⋅m2⋅s−2⋅A−1) 	(V⋅s)
            Tesla,    ///< (kg⋅s−2⋅A−1) 	(Wb/m2)
            Henry,    ///< (kg⋅m2⋅s−2⋅A−2) 	(Wb/A)
            lumens,   ///< (cd⋅sr) 	(cd⋅sr)
            lux,      ///< (cd⋅sr⋅m−2) 	(lm/m2)
            Bequerel, ///< (s−1)
            Gray,     ///< (m2⋅s−2) 	(J/kg)
            Sievert,  ///< (m2⋅s−2) 	(J/kg)
            katal,    ///< (mol⋅s−1)

            Hz  = Hertz,
            N   = Newtown,
            Pa  = Pascal,
            J   = Joule,
            W   = Watt,
            C   = Coulomb,
            V   = Volt,
            F   = Farad,
            ohm = Ohm,
            S   = Siemens,
            Wb  = Weber,
            T   = Tesla,
            H   = Henry,
            lm  = lumens,
            lx  = lux,
            Bq  = Bequerel,
            Gy  = Gray,
            Sv  = Sievert,
            kat = katal,

            /*
             * alternative base units
             */

            // other times units
            minute,
            hours,
            days,
            years,

            mn = minute,
            hr = hours,
            dy = days,
            yr = years,

            // other lenght units
            astronomical_unit,
            light_year,
            parsec,

            au = astronomical_unit,
            ly = light_year,
            pc = parsec,

            /*
             * alternative derived units
             */
            eV,
            electron_volt = eV, // (J)
            erg,                // (J)
        };

        inline const std::string get_prefix_str(UnitName p) {
            switch (p) {
            case second: return "s"; break;
            case metre: return "m"; break;
            case kilogramm: return "kg"; break;
            case Ampere: return "A"; break;
            case Kelvin: return "K"; break;
            case mole: return "mol"; break;
            case candela: return "cd"; break;
            case mps: return "m.s^{-1}"; break;
            case Hertz: return "Hz"; break;
            case Newtown: return "N"; break;
            case Pascal: return "P"; break;
            case Joule: return "J"; break;
            case Watt: return "W"; break;
            case Coulomb: return "C"; break;
            case Volt: return "V"; break;
            case Farad: return "F"; break;
            case Ohm: return "Ohm"; break;
            case Siemens: return "S"; break;
            case Weber: return "Wb"; break;
            case Tesla: return "T"; break;
            case Henry: return "H"; break;
            case lumens: return "lm"; break;
            case lux: return "lx"; break;
            case Bequerel: return "Bq"; break;
            case Gray: return "G"; break;
            case Sievert: return "S"; break;
            case katal: return "kat"; break;
            case minute: return "m"; break;
            case hours: return "h"; break;
            case days: return "dy"; break;
            case years: return "yr"; break;
            case astronomical_unit: return "au"; break;
            case light_year: return "ly"; break;
            case parsec: return "pc"; break;
            case eV: return "eV"; break;
            case erg: return "erg"; break;

            default: return ""; break;
            }
            return "";
        }

    } // namespace units

} // namespace shamrock