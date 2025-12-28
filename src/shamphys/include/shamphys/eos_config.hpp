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
 * @file eos_config.hpp
 * @author David Fang (david.fang@ikmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 */

#include "shambackends/sycl.hpp"
#include "shamunits/Constants.hpp"
#include "shamunits/UnitSystem.hpp"

namespace shamphys {

    /**
     * @brief Configuration struct for isothermal equation of state
     *
     * @tparam Tscal Scalar type
     *
     * This struct holds the configuration for the isothermal equation of state.
     * It contains the soundspeed cs = sqrt(RT).
     *
     * The equation of state is given by:
     * \f$ p = c_s^2 \rho \f$
     */
    template<class Tscal>
    struct EOS_Config_Isothermal {
        /// Soundspeed
        Tscal cs;
    };

    /**
     * @brief Configuration struct for adiabatic equation of state
     *
     * @tparam Tscal Scalar type
     *
     * This struct holds the configuration for the adiabatic equation of state.
     * It contains the adiabatic index, which is a dimensionless quantity that
     * determines the behavior of the gas.
     *
     * The equation of state is given by:
     * \f$ p = \rho^\gamma \f$
     */
    template<class Tscal>
    struct EOS_Config_Adiabatic {
        /// Adiabatic index
        Tscal gamma;
    };

    /**
     * @brief Equal operator for the EOS_Config_Adiabatic struct
     *
     * @tparam Tscal Scalar type
     * @param lhs First EOS_Config_Adiabatic struct to compare
     * @param rhs Second EOS_Config_Adiabatic struct to compare
     *
     * This function checks if two EOS_Config_Adiabatic structs are equal by comparing their
     * gamma values.
     *
     * @return true if the two structs have the same gamma value, false otherwise
     */
    template<class Tscal>
    inline bool operator==(
        const EOS_Config_Adiabatic<Tscal> &lhs, const EOS_Config_Adiabatic<Tscal> &rhs) {
        return lhs.gamma == rhs.gamma;
    }

    /**
     * @brief Configuration struct for polytropic equation of state
     *
     * @tparam Tscal Scalar type
     *
     * This struct holds the configuration for the polytropic equation of state.
     * It contains K and the adiabatic index, which are dimensionless quantities that
     * determines the behavior of the gas.
     *
     * The equation of state is given by:
     * \f$ P = K\rho^\gamma \f$
     */
    template<class Tscal>
    struct EOS_Config_Polytropic {
        Tscal K;
        Tscal gamma;
    };

    /**
     * @brief Equal operator for the EOS_Config_Polytropic struct
     *
     * @tparam Tscal Scalar type
     * @param lhs First EOS_Config_Polytropic struct to compare
     * @param rhs Second EOS_Config_Polytropic struct to compare
     *
     * This function checks if two EOS_Config_Polytropic structs are equal by comparing their K and
     * gamma values.
     *
     * @return true if the two structs have the same K and gamma values, false otherwise
     */
    template<class Tscal>
    inline bool operator==(
        const EOS_Config_Polytropic<Tscal> &lhs, const EOS_Config_Polytropic<Tscal> &rhs) {
        return (lhs.K == rhs.K) && (lhs.gamma == rhs.gamma);
    }

    /**
     * @brief Configuration struct for the locally isothermal equation of state from Lodato Price
     * 2007
     *
     * @tparam Tscal Scalar type
     *
     * The equation of state is given by:
     * \f$ p = (c_{s,0} (r / r_0)^{-q})^2 \rho \f$
     */
    template<class Tscal>
    struct EOS_Config_LocallyIsothermal_LP07 {
        /// Soundspeed at the reference radius
        Tscal cs0 = 0.005;

        /// Power exponent of the soundspeed profile
        Tscal q = 2;

        /// Reference radius
        Tscal r0 = 10;
    };

    /**
     * @brief Equal operator for the EOS_Config_LocallyIsothermal_LP07 struct
     *
     * @tparam Tscal Scalar type
     * @param lhs First EOS_Config_LocallyIsothermal_LP07 struct to compare
     * @param rhs Second EOS_Config_LocallyIsothermal_LP07 struct to compare
     *
     * This function checks if two EOS_Config_LocallyIsothermal_LP07 structs are equal by
     * comparing their cs0, q, and r0 values.
     *
     * @return true if the two structs have the same cs0, q, and r0 values, false otherwise
     */
    template<class Tscal>
    inline bool operator==(
        const EOS_Config_LocallyIsothermal_LP07<Tscal> &lhs,
        const EOS_Config_LocallyIsothermal_LP07<Tscal> &rhs) {
        return (lhs.cs0 == rhs.cs0) && (lhs.q == rhs.q) && (lhs.r0 == rhs.r0);
    }

    /**
     * @brief Configuration struct for the locally isothermal equation of state from Farris 2014
     *
     * @tparam Tscal Scalar type
     *
     * Note that the notation in the original paper are confusing and a clearer version is to use
     * the form in The Santa Barbara Binary−disk Code Comparison, Duffel et al. 2024
     *
     * The equation of state is given by:
     * \f$ c_s = (H(r)/r) \sqrt(- \phi_{\rm grav}) \f$
     */
    template<class Tscal>
    struct EOS_Config_LocallyIsothermalDisc_Farris2014 {
        Tscal h_over_r = 0.05;
    };

    /**
     * @brief Equal operator for the EOS_Config_LocallyIsothermalDisc_Farris2014 struct
     *
     * @tparam Tscal Scalar type
     * @param lhs First EOS_Config_LocallyIsothermalDisc_Farris2014 struct to compare
     * @param rhs Second EOS_Config_LocallyIsothermalDisc_Farris2014 struct to compare
     *
     * This function checks if two EOS_Config_LocallyIsothermalDisc_Farris2014 structs are equal by
     * comparing their cs0, q, and r0 values.
     *
     * @return true if the two structs have the same cs0, q, and r0 values, false otherwise
     */
    template<class Tscal>
    inline bool operator==(
        const EOS_Config_LocallyIsothermalDisc_Farris2014<Tscal> &lhs,
        const EOS_Config_LocallyIsothermalDisc_Farris2014<Tscal> &rhs) {
        return (lhs.h_over_r == rhs.h_over_r);
    }

    /**
     * @brief Configuration struct for the Tillotson equation of state
     *
     * @tparam Tscal Scalar type
     *
     * The equation of state is given by:
     * TODO
     */
    template<class Tscal>
    struct EOS_Config_Tillotson {
        /// Parameters
        Tscal rho0;
        Tscal E0;
        Tscal A;
        Tscal B;
        Tscal a;
        Tscal b;
        Tscal alpha;
        Tscal beta;
        Tscal u_iv;
        Tscal u_cv;
    };

    /**
     * @brief Equal operator for the EOS_Config_Tillotson struct
     *
     * @tparam Tscal Scalar type
     * @param lhs First EOS_Config_Tillotson struct to compare
     * @param rhs Second EOS_Config_Tillotson struct to compare
     *
     * This function checks if two EOS_Config_Tillotson structs are equal by
     * comparing their parameters values
     *
     * @return true if the two structs have the same parameter values, false otherwise
     */
    template<class Tscal>
    inline bool operator==(
        const EOS_Config_Tillotson<Tscal> &lhs, const EOS_Config_Tillotson<Tscal> &rhs) {
        return (lhs.rho0 == rhs.rho0) && (lhs.A == rhs.A) && (lhs.B == rhs.B) && (lhs.a == rhs.a)
               && (lhs.b == rhs.b) && (lhs.E0 == rhs.E0) && (lhs.alpha == rhs.alpha)
               && (lhs.beta == rhs.beta) && (lhs.u_iv == rhs.u_iv) && (lhs.u_cv == rhs.u_cv);
    }

} // namespace shamphys
