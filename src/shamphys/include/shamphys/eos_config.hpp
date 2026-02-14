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
     * @brief Configuration struct for Fermi equation of state
     *
     * @tparam Tscal Scalar type
     *
     * This struct holds the configuration for Fermi equation of state.
     * It contains mu_e the mean molecular weight which is dimensionless quantities that
     * determines the behavior of the gas.
     *
     * The equation of state is given by:
     * \f$ P = P(\rho, \mu_e) \f$, see `shamphys::EOS_Fermi` for the exact formula.
     */
    template<class Tscal>
    struct EOS_Config_Fermi {
        Tscal mu_e; ///< mu_e is the mean molecular weight
    };

    /**
     * @brief Equal operator for the EOS_Config_Fermi struct
     *
     * @tparam Tscal Scalar type
     * @param lhs First EOS_Config_Fermi struct to compare
     * @param rhs Second EOS_Config_Fermi struct to compare
     *
     * This function checks if two EOS_Config_Fermi structs are equal by comparing their mu_e
     * values.
     *
     * @return true if the two structs have the same mu_e values, false otherwise
     */
    template<class Tscal>
    inline bool operator==(const EOS_Config_Fermi<Tscal> &lhs, const EOS_Config_Fermi<Tscal> &rhs) {
        return lhs.mu_e == rhs.mu_e;
    }

    /**
     * @brief Configuration struct for the Tillotson equation of state
     *
     * @tparam Tscal Scalar type
     *
     * The equation of state is detailed in EOS_Tillotson
     */
    template<class Tscal>
    struct EOS_Config_Tillotson {
        Tscal rho0;  ///< Tillotson EoS \f$ \rho_0 \f$ parameter
        Tscal E0;    ///< Tillotson EoS \f$ E_0 \f$ parameter
        Tscal A;     ///< Tillotson EoS \f$ A \f$ parameter
        Tscal B;     ///< Tillotson EoS \f$ B \f$ parameter
        Tscal a;     ///< Tillotson EoS \f$ a \f$ parameter
        Tscal b;     ///< Tillotson EoS \f$ b \f$ parameter
        Tscal alpha; ///< Tillotson EoS \f$\alpha\f$ parameter
        Tscal beta;  ///< Tillotson EoS \f$\beta\f$ parameter
        Tscal u_iv;  ///< Tillotson EoS \f$ u_{\mathrm{iv}} \f$ parameter: energy of incipient
                     ///< vaporization
        Tscal u_cv;  ///< Tillotson EoS \f$ u_{\mathrm{cv}} \f$ parameter: energy of complete
        ///< vaporization
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
        Tscal cs0;

        /// Power exponent of the soundspeed profile
        Tscal q;

        /// Reference radius
        Tscal r0;
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
     * the form in The Santa Barbara Binary-disk Code Comparison, Duffel et al. 2024
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
     * @brief Configuration struct for the locally isothermal equation of state extended from Farris
     * 2014 to include for the q index of the disc.
     *
     * This EOS should match with ieos 13 and 14 of phantom.
     *
     * The equation in phantom is a bit weird so re-derived it here.
     *
     * Farris 2014 EOS which only corresponds to q=1/2:
     *
     * \f$
     * c_s = \frac{H_0}{r_0}\left(\frac{G M_1}{r_1} + \frac{G M_2}{r_2}\right)
     * \f$
     *
     * However the extension of that EOS to q != 1/2 was only introduced in Ragussa et al 2016, if
     * I'm right with:
     *
     * \f$
     * c_s = \frac{H_0}{r_0} \left(\frac{G M_1}{r_1} + \frac{G M_2}{r_2}\right)^{q}
     * \f$
     *
     * But as is the units are broken if q is not 1/2 so you need to compensate with
     * \f$r_0 \Omega_0\f$
     *
     * \f$c_s = \frac{H_0}{r_0}\frac{1}{(r_0 \Omega_0)^{2q - 1}}\left(\frac{G M_1}{r_1}
     * + \frac{G M_2}{r_2}\right)^{q} \f$
     *
     * \f$= c_{s0} \frac{1}{(r_0 \Omega_0)^{q}}\left(\frac{G M_1}{r_1}
     * + \frac{G M_2}{r_2}\right)^{q}\f$
     *
     * \f$= c_{s0}\frac{1}{r_0^{q}}\left[\frac{1}{\sum_i M_i}\sum_i \frac{M_i}{r_i}\right]^{q}\f$
     *
     * @tparam Tscal Scalar type
     */
    template<class Tscal>
    struct EOS_Config_LocallyIsothermalDisc_ExtendedFarris2014 {
        /// Soundspeed at the reference radius
        Tscal cs0;

        /// Power exponent of the soundspeed profile
        Tscal q;

        /// Reference radius
        Tscal r0;

        /// Number of sinks to consider for the equation of state
        u32 n_sinks;
    };

    /**
     * @brief Equal operator for the EOS_Config_LocallyIsothermalDisc_ExtendedFarris2014 struct
     *
     * @tparam Tscal Scalar type
     * @param lhs First EOS_Config_LocallyIsothermalDisc_ExtendedFarris2014 struct to compare
     * @param rhs Second EOS_Config_LocallyIsothermalDisc_ExtendedFarris2014 struct to compare
     *
     * This function checks if two EOS_Config_LocallyIsothermalDisc_ExtendedFarris2014 structs are
     equal by comparing their cs0, q, r0, and n_sinks values.

     * @return true if the two structs have the same cs0, q, r0, and n_sinks values, false otherwise
    */
    template<class Tscal>
    inline bool operator==(
        const EOS_Config_LocallyIsothermalDisc_ExtendedFarris2014<Tscal> &lhs,
        const EOS_Config_LocallyIsothermalDisc_ExtendedFarris2014<Tscal> &rhs) {
        return (lhs.cs0 == rhs.cs0) && (lhs.q == rhs.q) && (lhs.r0 == rhs.r0)
               && (lhs.n_sinks == rhs.n_sinks);
    }

} // namespace shamphys
