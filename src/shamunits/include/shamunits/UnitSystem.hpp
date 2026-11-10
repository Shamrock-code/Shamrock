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
 * @brief A unit system: the values of the seven SI base units in code units
 */

#include "Unit.hpp"
#include "details/utils.hpp"

namespace shamunits {

    /**
     * @brief A unit system, defined by the SI value of its base units
     *
     * Units, prefixes and constants come from shamunits/unit_table.hpp,
     * shamunits/prefix_table.hpp and shamunits/constant_table.hpp, and compose freely:
     * @code{.cpp}
     * using namespace shamunits;
     *
     * constexpr UnitSystem<double> si{};
     *
     * // time in Myr, length in au, mass in solar masses
     * constexpr UnitSystem<double> astro{
     *     si.get<prefix::mega * units::year>(),
     *     si.get<units::astronomical_unit>(),
     *     si.get<constants::sol_mass>(),
     * };
     *
     * astro.get<upow(units::astronomical_unit, 2)>(); // 1
     * astro.get<constants::G>();                      // 3.947813e13
     * astro.to<units::second>();                      // one code time unit in s
     * astro.get<constants::G, float>();               // result in another precision
     * astro.get(units::from_name("pc"), -1);          // unit known at run time
     * @endcode
     */
    template<class T>
    class UnitSystem {
        public:
        T s;       ///< value of a second in this system
        T m;       ///< value of a metre in this system
        T kg;      ///< value of a kilogram in this system
        T A;       ///< value of an ampere in this system
        T K;       ///< value of a kelvin in this system
        T mol;     ///< value of a mole in this system
        T cd;      ///< value of a candela in this system
        T s_inv;   ///< inverse value of a second in this system (the time unit in s)
        T m_inv;   ///< inverse value of a metre in this system (the length unit in m)
        T kg_inv;  ///< inverse value of a kilogram in this system (the mass unit in kg)
        T A_inv;   ///< inverse value of an ampere in this system
        T K_inv;   ///< inverse value of a kelvin in this system
        T mol_inv; ///< inverse value of a mole in this system
        T cd_inv;  ///< inverse value of a candela in this system

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

        // ---------------------------------------------------------------------------------------
        // Getters. Units come from shamunits::units, prefixes from shamunits::prefix and constants
        // from shamunits::constants, and compose freely: get<prefix::mega * units::year>(),
        // get<upow(units::metre, 2)>(). get<u>(), to<u>() and get(u) never divide.
        //
        // Precision: the product is accumulated in Tacc, the widest of T, Tret and double, and
        // converted to Tret once at the end. Units and their factors are double by design, so
        // this costs nothing for UnitSystem<f64>; for f32 it keeps intermediate powers (e.g.
        // s^-3 = 3e40 in a Myr system) from overflowing, so the result is the correctly rounded
        // value whenever it fits in Tret. The factor is applied last.
        /// Accumulation type of the Unit-value getters: the widest of T, Tret and double
        template<class Tret>
        using acc_t = decltype(T{} * Tret{} * double{});

        // ---------------------------------------------------------------------------------------

        /**
         * @brief Value of the compile-time unit `u` in this system
         *
         * `u` is a by-value template parameter, so every exponent is a constant expression:
         * dimensions with exponent 0 expand to nothing. The parameter is `Unit u` (deduced) on
         * purpose; it compiles on clang 15. Never replace it with a raw `double` template
         * parameter, which clang < 18 rejects.
         *
         * @code{.cpp}
         * astro.get<units::astronomical_unit>();              // in this system's lengths
         * si.get<prefix::mega * units::year>();               // 3.15576e13
         * usys.get<constants::G, f32>();                      // result as f32
         * @endcode
         */
        template<Unit u, class Tret = T>
        constexpr Tret get() const noexcept {
            using details::pow_constexpr_fast_inv;
            using Tacc = acc_t<Tret>;
            return Tret(
                pow_constexpr_fast_inv<u.dim.second>(Tacc(s), Tacc(s_inv))
                * pow_constexpr_fast_inv<u.dim.metre>(Tacc(m), Tacc(m_inv))
                * pow_constexpr_fast_inv<u.dim.kilogram>(Tacc(kg), Tacc(kg_inv))
                * pow_constexpr_fast_inv<u.dim.ampere>(Tacc(A), Tacc(A_inv))
                * pow_constexpr_fast_inv<u.dim.kelvin>(Tacc(K), Tacc(K_inv))
                * pow_constexpr_fast_inv<u.dim.mole>(Tacc(mol), Tacc(mol_inv))
                * pow_constexpr_fast_inv<u.dim.candela>(Tacc(cd), Tacc(cd_inv))
                * Tacc(u.si_factor));
        }

        /// Inverse of get<u>(): one unit of this system expressed in `u`, e.g.
        /// to<units::kilogram>() is the code mass unit in kg. The reciprocal of the factor is a
        /// constant expression, so this does not divide either.
        template<Unit u, class Tret = T>
        constexpr Tret to() const noexcept {
            return get<upow(u, -1), Tret>();
        }

        /// Value of a unit only known at run time (e.g. looked up by name). Exponents are runtime
        /// values; the stored *_inv members serve negative exponents, so there is no division.
        template<class Tret = T, class Tu>
        constexpr Tret get(Unit<Tu> u) const noexcept {
            using details::ipow;
            using Tacc = acc_t<Tret>;
            return Tret(
                ipow(Tacc(s), u.dim.second, Tacc(s_inv)) * ipow(Tacc(m), u.dim.metre, Tacc(m_inv))
                * ipow(Tacc(kg), u.dim.kilogram, Tacc(kg_inv))
                * ipow(Tacc(A), u.dim.ampere, Tacc(A_inv))
                * ipow(Tacc(K), u.dim.kelvin, Tacc(K_inv))
                * ipow(Tacc(mol), u.dim.mole, Tacc(mol_inv))
                * ipow(Tacc(cd), u.dim.candela, Tacc(cd_inv)) * Tacc(u.si_factor));
        }

        /// get(u) for `u` raised to a runtime power: exactly get(upow(u, power)). The only entry
        /// point that can divide (in upow, for power < 0); this is the Python path.
        template<class Tret = T, class Tu>
        constexpr Tret get(Unit<Tu> u, int power) const noexcept {
            return get<Tret>(upow(u, power));
        }

        /// Inverse of get(u)
        template<class Tret = T, class Tu>
        constexpr Tret to(Unit<Tu> u) const noexcept {
            return get<Tret>(upow(u, -1));
        }

        /// Inverse of get(u, power)
        template<class Tret = T, class Tu>
        constexpr Tret to(Unit<Tu> u, int power) const noexcept {
            return get<Tret>(upow(u, -power));
        }
    };

} // namespace shamunits
