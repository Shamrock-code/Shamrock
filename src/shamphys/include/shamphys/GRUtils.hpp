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

        inline static constexpr Tscal get_alpha(
            std::mdspan<Tscal, std::extents<SizeType, 4, 4>, Layout2, Accessor2> gcon) {
            // alpha = 1 / sqrt(-g00)
            return 1. / sycl::sqrt(-gcon(0, 0));
        }

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

        inline static constexpr Tvec get_betaDOWN(
            std::mdspan<Tscal, std::extents<SizeType, 4, 4>, Layout2, Accessor2> gcov) {
            Tvec betaDOWN = {0., 0., 0.};

            betaDOWN[0] = gcov(0, 1);
            betaDOWN[1] = gcov(0, 2);
            betaDOWN[2] = gcov(0, 3);

            return betaDOWN;
        }

        inline static constexpr Tscal GR_dot(
            std::mdspan<Tscal, std::extents<SizeType, 1, 4>, Layout2, Accessor2> a,
            std::mdspan<Tscal, std::extents<SizeType, 1, 4>, Layout2, Accessor2> b,
            std::mdspan<Tscal, std::extents<SizeType, 4, 4>, Layout2, Accessor2> gcov) {

            Tscal result = 0;
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    result += gcov(i + 1, j + 1) * a[i] * b[j];
                }
            }
            return result;
        }

        inline static constexpr Tscal GR_dot_spatial(
            Tvec a,
            Tvec b,
            std::mdspan<Tscal, std::extents<SizeType, 3, 3>, Layout2, Accessor2> gamma_ij) {

            Tscal result = 0;
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    result += gamma_ij(i, j) * a[i] * b[j];
                }
            }
            return result;
        }

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

        inline static constexpr Tscal get_gamma(
            std::mdspan<Tscal, std::extents<SizeType, 4, 4>, Layout2, Accessor2> gcov) {

            std::mdspan<Tscal, std::extents<SizeType, 3, 3>, Layout2, Accessor2> gamma_ij;
            for (int i = 1; i < 4; i++) {
                for (int j = 1; j < 4; j++) {
                    gamma_ij(i, j) = gcov(i, j);
                }
            }

            return mat_det_33(gamma_ij);
        }

        inline static constexpr Tscal get_lorentz_factor(
            Tvec momentum,
            Tscal enthalpy,
            std::mdspan<Tscal, std::extents<SizeType, 4, 4>, Layout2, Accessor2> gcov) {

            return sycl::sqrt(1. + GR_dot(momentum, momentum, gcov) / (enthalpy * enthalpy));
        }

        inline static constexpr Tscal get_U0(
            Tvec momentum,
            Tscal enthalpy,
            std::mdspan<Tscal, std::extents<SizeType, 4, 4>, Layout2, Accessor2> gcon,
            std::mdspan<Tscal, std::extents<SizeType, 4, 4>, Layout2, Accessor2> gcov) {

            // U0 = gamma / alpha wrong
            Tscal alpha          = get_alpha(gcon);
            Tscal lorentz_factor = get_lorentz_factor(momentum, enthalpy, gcov);
            return lorentz_factor / alpha;
        }

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
