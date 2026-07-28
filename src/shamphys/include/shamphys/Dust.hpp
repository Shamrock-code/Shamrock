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
 * @file Dust.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Epstein drag stopping time for spherical dust grains
 */

#include "shambase/assert.hpp"
#include "shambase/constants.hpp"
#include "shambackends/sycl.hpp"

namespace shamphys {

    /**
     * @brief Epstein drag supersonic correction factor
     *
     * Corrects the Epstein drag for supersonic drift between dust and gas.
     * When the drift speed exceeds the gas thermal speed, the relative
     * motion must be accounted for in the drag force.
     *
     * \f[
     *     f(\Delta v, c_s) = \sqrt{1 + \frac{9\pi}{128} \frac{\Delta v^2}{c_s^2}}
     * \f]
     *
     * Corresponds to Eq. 249 in the PHANTOM paper.
     *
     * @param delta_v Drift speed between dust and gas
     * @param cs      Gas sound speed
     * @return Supersonic correction factor f(delta_v, cs)
     */
    template<class T>
    inline T epstein_supersonic_correction(T delta_v, T cs) noexcept {

        SHAM_ASSERT(cs > 0);

        auto div = delta_v / cs;
        return sycl::sqrt(T(1.0) + T(9.0 / 128.0) * shambase::constants::pi<T> * div * div);
    }

    /**
     * @brief Epstein drag stopping time for spherical dust grains
     *
     * \f[
     *     t_s = \frac{\rho_{\rm grain} \, s_{\rm grain}}{\rho \, c_s \, f}
     *           \sqrt{\frac{\pi \gamma}{8}}
     * \f]
     *
     * Corresponds to Eq. 250 in the PHANTOM paper.
     *
     * where \f$\rho = \rho_{\rm g} + \rho_{\rm d}\f$ is the total density,
     * \f$f\f$ is the supersonic correction (1.0 for the subsonic case),
     * \f$\gamma\f$ is the adiabatic index,
     * \f$\rho_{\rm grain}\f$ is the grain internal density, and
     * \f$s_{\rm grain}\f$ is the grain radius.
     *
     * @param rho_grain   Internal density of the dust grain
     * @param s_grain     Radius of the dust grain
     * @param rho         Total density (\f$\rho_{\rm g} + \rho_{\rm d}\f$)
     * @param cs          Gas sound speed
     * @param gamma       Adiabatic index
     * @param f           Supersonic correction factor (default 1.0)
     * @return Stopping time
     */
    template<class T>
    inline T epstein_stopping_time(
        T rho_grain, T s_grain, T rho, T cs, T gamma, T f = T(1.0)) noexcept {

        SHAM_ASSERT(rho * cs * f > 0);
        SHAM_ASSERT(gamma > 0);

        return (rho_grain * s_grain / (rho * cs * f))
               * sycl::sqrt(shambase::constants::pi<T> * gamma / T(8));
    }

} // namespace shamphys
