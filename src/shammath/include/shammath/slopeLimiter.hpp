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
 * @file slopeLimiter.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief Math header to compute slope limiters
 *
 */

#include "shambackends/math.hpp"

namespace shammath {

    /**
     * @brief Van leer slope limiter
     *
     *
     * \f[
     *   \\phi(f,g) = \frac{f g + \vert f g \vert}{f + g + \epsilon}
     * \f]
     *
     * \todo clarify the definition, normally this is done as adimensionalized function but here it
     * is not.
     *
     * @tparam T
     * @param f
     * @param g
     * @return T
     */
    template<class T>
    inline T van_leer_slope(T f, T g) {
        T tmp = f * g;
        if (tmp > 0) {
            return 2 * tmp / (f + g);
        } else {
            return 0;
        }
    }

    template<class T>
    inline T van_leer_slope_symetric(T sR, T sL) {
        T abs_sR = sham::abs(sR);
        T abs_sL = sham::abs(sL);
        T sgn_sR = sycl::sign(sR);
        T sgn_sL = sycl::sign(sL);

        T tmp = abs_sR * abs_sL;

        if (tmp > 0) {
            return tmp * (sgn_sL + sgn_sR) / (abs_sL + abs_sR);
        } else {
            return 0;
        }
    }

    template<class T>
    inline T minmod(T sR, T sL) {
        T r = sL / sR;
        return ((r > 0) ? ((r < 1) ? r : 1) : 0) * sR;
    }

} // namespace shammath
