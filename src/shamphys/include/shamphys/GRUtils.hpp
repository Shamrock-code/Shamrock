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
 * @file GRUtils.hpp
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief
 */

#include "shambase/aliases_int.hpp"
#include "shambase/assert.hpp"
#include "shambackends/math.hpp"
#include "shamunits/Constants.hpp"
#include <experimental/mdspan>
#include <shambackends/sycl.hpp>

namespace shamphys {

    template<
        class Tvec,
        class Tscal,
        class SizeType,
        class Layout1,
        class Layout2,
        class Accessor1,
        class Accessor2>
    struct GR_physics {

        /**
         * @brief Compute the alpha factor from the spacetime metric.
         *
         * Alpha is defined as 1 / sqrt(-g00) where g00 is the
         * 0,0 component of the spacetime metric.
         *
         * @param gcon The spacetime metric as a 4x4 matrix.
         * @return The alpha factor.
         */
        inline static constexpr Tscal get_alpha(
            std::mdspan<Tscal, std::extents<SizeType, 4, 4>, Layout2, Accessor2> gcon) {
            // alpha = 1 / sqrt(-g00)
            return 1. / sycl::sqrt(-gcon(0, 0));
        }

        /**
         * @brief Compute the betaUP factor from the spacetime metric.
         *
         * BetaUP is defined as (g01, g02, g03) / (alpha^2) where
         * g0i is the i-th component of the first row of the spacetime
         * metric and alpha is the Lorentz factor.
         *
         * @param gcon The spacetime metric as a 4x4 matrix.
         * @return The betaUP factor.
         */
        inline static constexpr Tvec get_betaUP(
            std::mdspan<Tscal, std::extents<SizeType, 4, 4>, Layout2, Accessor2> gcon) {
            Tscal alpha  = get_alpha(gcon);
            Tscal alpha2 = alpha * alpha;
            Tvec betaUP  = {0., 0., 0.};

            betaUP[0] = gcon(0, 1) * alpha2;
            betaUP[1] = gcon(0, 2) * alpha2;
            betaUP[2] = gcon(0, 3) * alpha2;

            return betaUP;
        }

        /**
         * @brief Compute the betaDOWN factor from the spacetime metric.
         *
         * BetaDOWN is defined as (g01, g02, g03) where g0i is the i-th
         * component of the first row of the spacetime metric.
         *
         * @param gcov The spacetime metric as a 4x4 matrix.
         * @return The betaDOWN factor.
         */
        inline static constexpr Tvec get_betaDOWN(
            std::mdspan<Tscal, std::extents<SizeType, 4, 4>, Layout2, Accessor2> gcov) {
            Tvec betaDOWN = {0., 0., 0.};

            betaDOWN[0] = gcov(0, 1);
            betaDOWN[1] = gcov(0, 2);
            betaDOWN[2] = gcov(0, 3);

            return betaDOWN;
        }

        /**
         * @brief Compute the dot product of two vectors in the spacetime metric.
         *
         * @param a The first vector.
         * @param b The second vector.
         * @param gcov The spacetime metric as a 4x4 matrix.
         * @return The dot product of the two vectors in the spacetime metric.
         */
        inline static constexpr Tscal GR_dot(
            Tvec a,
            Tvec b,
            std::mdspan<Tscal, std::extents<SizeType, 4, 4>, Layout2, Accessor2> gcov) {

            Tscal result = 0;
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    result += gcov(i + 1, j + 1) * a[i] * b[j];
                }
            }
            return result;
        }

        /**
         * @brief Compute the dot product of two spatial vectors in the spacetime metric.
         *
         * @param a The first spatial vector.
         * @param b The second spatial vector.
         * @param gamma_ij The spatial part of the spacetime metric as a 3x3 matrix.
         * @return The dot product of the two spatial vectors in the spacetime metric.
         */
        inline static constexpr Tscal GR_dot_spatial(
            Tvec a,
            Tvec b,
            std::mdspan<Tscal, std::extents<SizeType, 3, 3>, Layout2, Accessor2> gamma_ij) {

            Tscal result = 0;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    result += gamma_ij(i + 1, j + 1) * a[i] * b[j];
                }
            }
            return result;
        }

        /**
         * @brief Compute the spatial part of the spacetime metric from the full 4x4 metric.
         *
         * @param gcov The spacetime metric as a 4x4 matrix.
         * @return The spatial part of the spacetime metric as a 3x3 matrix.
         */
        inline static constexpr std::mdspan<Tscal, std::extents<SizeType, 3, 3>, Layout2, Accessor2>
        get_gammaijDOWN(std::mdspan<Tscal, std::extents<SizeType, 4, 4>, Layout2, Accessor2> gcov) {

            std::mdspan<Tscal, std::extents<SizeType, 3, 3>, Layout2, Accessor2> gamma_ij;
            for (int i = 1; i < 4; i++) {
                for (int j = 1; j < 4; j++) {
                    gamma_ij(i - 1, j - 1) = gcov(i, j);
                }
            }

            return gamma_ij;
        }

        /**
         * @brief Compute the contravariant spatial part of the spacetime metric from the full 4x4
         * metric.
         *
         * @param gcon The spacetime metric as a 4x4 matrix.
         * @return The contravariant spatial part of the spacetime metric as a 3x3 matrix.
         */
        inline static constexpr std::mdspan<Tscal, std::extents<SizeType, 3, 3>, Layout2, Accessor2>
        get_gammaijUP(std::mdspan<Tscal, std::extents<SizeType, 4, 4>, Layout2, Accessor2> gcon) {

            Tscal alpha = get_alpha(gcon);
            Tvec betaUP = get_betaUP(gcon);
            std::mdspan<Tscal, std::extents<SizeType, 3, 3>, Layout2, Accessor2> gamma_ij;
            for (int i = 1; i < 4; i++) {
                for (int j = 1; j < 4; j++) {
                    gamma_ij(i - 1, j - 1)
                        = gcon(i, j) + betaUP[i - 1] * betaUP[j - 1] / (-gcon(0, 0));
                }
            }

            return gamma_ij;
        }

        /**
         * @brief Compute the determinant of the spatial part of the spacetime metric.
         *
         * This function takes a 3x3 matrix representing the spatial part of the
         * spacetime metric and returns its determinant.
         *
         * @param gamma_ij The spatial part of the spacetime metric as a 3x3 matrix.
         * @return The determinant of the spatial part of the spacetime metric.
         */
        inline static constexpr Tscal get_gamma(
            std::mdspan<Tscal, std::extents<SizeType, 3, 3>, Layout2, Accessor2> gamma_ij) {

            return mat_det_33(gamma_ij);
        }

        /**
         * @brief Compute the Lorentz factor of a fluid element given its momentum and the spacetime
         * metric.
         *
         * This function takes the momentum of a fluid element, its enthalpy, and the spacetime
         * metric as inputs and returns the Lorentz factor of the fluid element.
         *
         * @param momentum The momentum of the fluid element.
         * @param enthalpy The enthalpy of the fluid element.
         * @param gcov The spacetime metric as a 4x4 matrix.
         * @return The Lorentz factor of the fluid element.
         */
        inline static constexpr Tscal get_lorentz_factor(
            Tvec momentum,
            Tscal enthalpy,
            std::mdspan<Tscal, std::extents<SizeType, 4, 4>, Layout2, Accessor2> gcov) {

            return sycl::sqrt(1. + GR_dot(momentum, momentum, gcov) / (enthalpy * enthalpy));
        }

        /// @brief Compute the time component of the 4-velocity of a fluid element given its
        /// momentum, enthalpy, and the spacetime metric.
        ///
        /// This function takes the momentum of a fluid element, its enthalpy, and the spacetime
        /// metric as inputs and returns the time component of the 4-velocity of the fluid element.
        ///
        /// @param momentum The momentum of the fluid element.
        /// @param enthalpy The enthalpy of the fluid element.
        /// @param gcon The spacetime metric as a 4x4 matrix.
        /// @param gcov The spacetime metric as a 4x4 matrix.
        /// @return The time component of the 4-velocity of the fluid element.
        inline static constexpr Tscal get_U0(
            Tvec momentum,
            Tscal enthalpy,
            std::mdspan<Tscal, std::extents<SizeType, 4, 4>, Layout2, Accessor2> gcon,
            std::mdspan<Tscal, std::extents<SizeType, 4, 4>, Layout2, Accessor2> gcov) {

            Tscal alpha          = get_alpha(gcon);
            Tscal lorentz_factor = get_lorentz_factor(momentum, enthalpy, gcov);
            return lorentz_factor / alpha;
        }

        /**
         * @brief Compute the fluid velocity in the local rest frame.
         *
         * This function takes the fluid velocity in the global frame and the spacetime metric as
         * inputs and returns the fluid velocity in the local rest frame.
         *
         * @param vxyz The fluid velocity in the global frame.
         * @param gcon The spacetime metric as a 4x4 matrix.
         * @return The fluid velocity in the local rest frame.
         */
        inline static constexpr Tvec get_V(
            Tvec vxyz, std::mdspan<Tscal, std::extents<SizeType, 4, 4>, Layout2, Accessor2> gcon) {
            Tscal alpha = get_alpha(gcon);
            Tvec betaUP = get_betaUP(gcon);
            Tvec V      = (vxyz + betaUP) / alpha;

            return V;
        }

        /// placeholder function for more complex metrics
        inline static constexpr Tscal get_sqrt_g(
            Tvec vxyz, std::mdspan<Tscal, std::extents<SizeType, 4, 4>, Layout2, Accessor2> gcov) {
            // for unusual metric, need to compute the determinant
            // for now (Kerr), this is enough
            return 1;
        }

        /// placeholder function for more complex metrics
        inline static constexpr Tscal get_sqrt_gamma(
            Tvec vxyz, std::mdspan<Tscal, std::extents<SizeType, 4, 4>, Layout2, Accessor2> gcov) {
            // for unusual metric, need to compute the determinant of the spatial metric
            // for now (Kerr), this is enough
            return 1;
        }
    };

} // namespace shamphys
