// -------------------------------------------------------//
//
// SHAMROCK code for hydrodynamics
// Copyright(C) 2021-2023 Timothée David--Cléris <timothee.david--cleris@ens-lyon.fr>
// Licensed under CeCILL 2.1 License, see LICENSE for more information
//
// -------------------------------------------------------//

#pragma once

/**
 * @file eos_config.hpp
 * @author Timothée David--Cléris (timothee.david--cleris@ens-lyon.fr)
 * @brief
 */

#include "shambackends/sycl.hpp"
#include "shamunits/Constants.hpp"
#include "shamunits/UnitSystem.hpp"

namespace shamphys {

    /**
     * @brief Configuration struct for adiabatic equation of state
     *
     * @tparam Tscal Scalar type
     */
    template<class Tscal>
    struct EOS_Config_Adiabatic {
        Tscal gamma;
    };

    /**
     * @brief equal operator for EOS_Config_Adiabatic
     */
    template<class Tscal>
    inline bool
    operator==(const EOS_Config_Adiabatic<Tscal> &lhs, const EOS_Config_Adiabatic<Tscal> &rhs) {
        return lhs.gamma == rhs.gamma;
    }

    /**
     * @brief Configuration struct for locally isothermal equation of state from Lodato Price 2007
     *
     * @tparam Tscal Scalar type
     */
    template<class Tscal>
    struct EOS_Config_LocallyIsothermal_LP07 {
        Tscal cs0 = 0.005;
        Tscal q   = -2;
        Tscal r0  = 10;
    };

    /**
     * @brief equal operator for EOS_Config_LocallyIsothermal_LP07
     */
    template<class Tscal>
    inline bool operator==(
        const EOS_Config_LocallyIsothermal_LP07<Tscal> &lhs,
        const EOS_Config_LocallyIsothermal_LP07<Tscal> &rhs) {
        return (lhs.cs0 == rhs.cs0) && (lhs.q == rhs.q) && (lhs.r0 == rhs.r0);
    }

} // namespace shamphys
