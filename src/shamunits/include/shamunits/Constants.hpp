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
 * @file Constants.hpp
 * @author David Fang (david.fang@ikmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief Physical constants in a unit system, plus a few dimensionless numbers
 *
 * The constants themselves are the values of shamunits::constants (constant_table.hpp).
 * Constants<T> is kept for backward compatibility: it is a thin wrapper around a UnitSystem,
 * and Constants<T>{usys}.get<constants::G>() is the same as usys.get<constants::G>().
 */

#include "UnitSystem.hpp"
#include "constant_table.hpp"

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
     * @brief Physical constants in a unit system
     *
     * \code{.cpp}
     * shamunits::Constants<f64> cte{usys};
     * f64 G  = cte.get<shamunits::constants::G>();
     * f64 G2 = cte.get(shamunits::constants::from_name("G"), 2);
     * \endcode
     */
    template<class T>
    struct Constants {

        /// Current unit system of the constants
        const UnitSystem<T> units;

        /// Construct the \ref shamunits::Constants class with a unit system
        constexpr explicit Constants(const UnitSystem<T> units) : units(units) {}

        /// Value of the constant (or unit) `u` in this unit system, e.g. get<constants::G>().
        /// Same as units.get<u, Tret>(); see UnitSystem::get.
        template<Unit u, class Tret = T>
        constexpr Tret get() const noexcept {
            return units.template get<u, Tret>();
        }

        /// Value of a constant only known at run time, e.g. get(constants::from_name("G"))
        template<class Tret = T, class Tu>
        constexpr Tret get(Unit<Tu> u) const noexcept {
            return units.template get<Tret>(u);
        }

        /// get(u) raised to a runtime power: the Python path, Constants(usys).G(power=2)
        template<class Tret = T, class Tu>
        constexpr Tret get(Unit<Tu> u, int power) const noexcept {
            return units.template get<Tret>(u, power);
        }
    };

} // namespace shamunits
