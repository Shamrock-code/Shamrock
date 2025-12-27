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
 * @file SedovTaylor.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Guo Yansong (guo.yansong.ngy@gmail.com)
 * @brief Sedov-Taylor blast wave analytical solution
 *
 * References:
 * - Sedov, L.I. (1959) "Similarity and Dimensional Methods in Mechanics"
 * - Taylor, G.I. (1950) "The Formation of a Blast Wave by a Very Intense Explosion"
 */

#include "shambase/aliases_float.hpp"
#include "shambase/aliases_int.hpp"
#include <cmath>
#include <vector>

namespace shamphys {

    /**
     * @brief Represents a Sedov-Taylor solution using pre-computed reference data.
     *
     * This class provides a way to calculate the values of density, velocity, and pressure at a
     * given position using interpolation from reference data at a fixed time.
     *
     * gamma = 5./3.
     * t = 0.1
     * \int u_inj = 1
     */
    class SedovTaylor {
        public:
        inline SedovTaylor() {}

        struct field_val {
            f64 rho, vx, P;
        };

        field_val get_value(f64 x);
    };

    /**
     * @brief Time-dependent Sedov-Taylor blast wave analytical solution.
     *
     * This class computes the self-similar Sedov-Taylor solution for a point explosion
     * in a uniform medium. Unlike SedovTaylor which uses pre-computed data, this class
     * calculates the solution analytically for any time t.
     *
     * The solution describes a spherical blast wave expanding into a uniform medium
     * with the shock radius following: R_s = xi_0 * (E * t^2 / rho_0)^(1/(n+2))
     * where n is the number of dimensions.
     */
    class SedovTaylorAnalytical {
        public:
        /**
         * @brief Construct a new Sedov-Taylor analytical solution.
         *
         * @param gamma Adiabatic index (ratio of specific heats), typically 5/3 or 1.4
         * @param E_blast Total blast energy
         * @param rho_0 Ambient (pre-shock) density
         * @param ndim Number of spatial dimensions (1, 2, or 3)
         */
        SedovTaylorAnalytical(
            f64 gamma = 5.0 / 3.0, f64 E_blast = 1.0, f64 rho_0 = 1.0, u32 ndim = 3);

        /**
         * @brief Field values at a point (density, radial velocity, pressure).
         */
        struct FieldValues {
            f64 rho; ///< Density
            f64 v_r; ///< Radial velocity
            f64 P;   ///< Pressure
        };

        /**
         * @brief Compute the shock radius at time t.
         *
         * R_s = xi_0 * (E_blast * t^2 / rho_0)^(1/(ndim+2))
         *
         * @param t Time since explosion
         * @return Shock radius
         */
        f64 shock_radius(f64 t) const;

        /**
         * @brief Compute the shock velocity at time t.
         *
         * v_s = dR_s/dt = (2/(ndim+2)) * R_s / t
         *
         * @param t Time since explosion
         * @return Shock velocity
         */
        f64 shock_velocity(f64 t) const;

        /**
         * @brief Compute the post-shock (immediately behind shock) density.
         *
         * rho_s = rho_0 * (gamma + 1) / (gamma - 1)
         *
         * @return Post-shock density
         */
        f64 post_shock_density() const;

        /**
         * @brief Compute field values at radius r and time t.
         *
         * @param r Radial distance from explosion center
         * @param t Time since explosion
         * @return FieldValues containing density, velocity, and pressure
         */
        FieldValues get_value(f64 r, f64 t) const;

        /**
         * @brief Compute radial profiles at time t.
         *
         * @param t Time since explosion
         * @param r_values Vector of radial positions to evaluate
         * @param[out] rho_out Density values at each r
         * @param[out] v_out Velocity values at each r
         * @param[out] P_out Pressure values at each r
         */
        void solution_at_time(
            f64 t,
            const std::vector<f64> &r_values,
            std::vector<f64> &rho_out,
            std::vector<f64> &v_out,
            std::vector<f64> &P_out) const;

        /// Get the adiabatic index
        f64 get_gamma() const { return gamma; }

        /// Get the blast energy
        f64 get_E_blast() const { return E_blast; }

        /// Get the ambient density
        f64 get_rho_0() const { return rho_0; }

        /// Get the number of dimensions
        u32 get_ndim() const { return ndim; }

        /// Get the Sedov constant xi_0
        f64 get_xi_0() const { return xi_0; }

        private:
        f64 gamma;   ///< Adiabatic index
        f64 E_blast; ///< Blast energy
        f64 rho_0;   ///< Ambient density
        u32 ndim;    ///< Number of dimensions
        f64 xi_0;    ///< Sedov constant

        /**
         * @brief Compute the Sedov constant xi_0 for given gamma and dimensions.
         *
         * Uses tabulated values for common cases, falls back to default for others.
         */
        f64 compute_xi_0() const;
    };

} // namespace shamphys
