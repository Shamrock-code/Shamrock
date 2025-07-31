// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright (c) 2021-2025 Timothée David--Cléris <tim.shamrock@proton.me>
// SPDX-License-Identifier: CeCILL Free Software License Agreement v2.1
// Shamrock is licensed under the CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file ConvertionConstants.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 */

namespace shamunits {

    /// Conversion constants from a units to another one
    /// \todo Could be also a XMacro next to names definitions
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

        static constexpr T litre_to_pint = 0.568;

        /// We first want to thanks E. Lynch for his usefull contribution to the devellopement of
        ///    numerical methods and new standard of measurment
        /// \verbatim
        /// todo find a more precise measurment of the guiness density (shall be done in ireland)
        /// protocol :
        ///  1 - start by ordering the most precise pint of guiness (in ireland)
        ///  2 - weight the pint
        ///  4 - proceed to drink the beer (including any leftover foam)
        ///  54 - weight the pint again
        ///  745 - proceed to compute the dedzabfgzi if the zad beer
        ///  current estimation is sourced from
        ///     - Tinseth, Glenn. Javascript Beer Specs Calculator. The Real Beer Page. 1997.
        /// \endverbatim
        static constexpr T gcm3_to_guiness_density = 1.017; // in g.cm-3
    };

} // namespace shamunits

/**
 * @fn shamunits::ConvertionConstants::au_to_m()
 * @brief conversion factor from au to meters
 *
 * @fn shamunits::ConvertionConstants::ly_to_m()
 * @brief conversion factor from light years to meters
 *
 * @fn shamunits::ConvertionConstants::pc_to_m()
 * @brief conversion factor from parsecs to meters
 *
 * @fn shamunits::ConvertionConstants::mn_to_s()
 * @brief conversion factor from minutes to seconds
 *
 * @fn shamunits::ConvertionConstants::hr_to_s()
 * @brief conversion factor from hours to seconds
 *
 * @fn shamunits::ConvertionConstants::dy_to_s()
 * @brief conversion factor from days to seconds
 *
 * @fn shamunits::ConvertionConstants::yr_to_s()
 * @brief conversion factor from years to seconds
 *
 * @fn shamunits::ConvertionConstants::Myr_to_s()
 * @brief conversion factor from $10^6$ years to seconds
 *
 * @fn shamunits::ConvertionConstants::Gyr_to_s()
 * @brief conversion factor from $10^9$ years to seconds
 *
 * @fn shamunits::ConvertionConstants::eV_to_J()
 * @brief conversion factor from electron volts to joules
 *
 * @fn shamunits::ConvertionConstants::erg_to_J()
 * @brief conversion factor from ergs to joules
 *
 * @fn shamunits::ConvertionConstants::K_degC_offset()
 * @brief conversion offset from kelvin degrees to celsius
 *
 * @fn shamunits::ConvertionConstants::litre_to_pint()
 * @brief conversion offset from litre to british pint
 *
 */
