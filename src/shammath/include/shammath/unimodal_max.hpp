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
 * @file unimodal_max.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @author Yona Lapeyre (yona.lapeyre@ens-lyon.fr)
 * @brief Maximum value of a unimodal function on an interval
 *
 */

#include "shambase/aliases_int.hpp"
#include <algorithm>
#include <cmath>

namespace shammath {

    /**
     * @brief Maximum value of a unimodal function on [a, b]
     *
     * Dichotomic search keeping the half containing the max, until the estimated max is
     * converged to the relative tolerance ``rel_tol``. The max may sit on a boundary if the
     * function is monotonic, so the boundary values are included in the result.
     *
     * Typical use is the bound of a rejection sampling, where the caller should still check
     * that the sampled values never exceed the returned max (with a margin matching
     * ``rel_tol``), as a function that is not unimodal would make the search miss its max.
     *
     * @param f the function, assumed unimodal on [a, b]
     * @param a lower bound of the interval
     * @param b upper bound of the interval
     * @param rel_tol relative convergence criterion on the max value
     * @param max_iter maximum number of iterations
     * @return the estimated maximum value of f on [a, b]
     */
    template<class T, class Func>
    inline T unimodal_max_value(Func &&f, T a, T b, T rel_tol = T(1e-8), u32 max_iter = 200) {

        const T fa = f(a);
        const T fb = f(b);

        T f_prev = std::max(fa, fb);
        T f_est  = f_prev;

        for (u32 iter = 0; iter < max_iter; iter++) {
            T m     = (a + b) / 2;
            T delta = (b - a) * T(1e-3);

            T f1 = f(m - delta);
            T f2 = f(m + delta);

            // keep the half containing the max
            if (f1 < f2) {
                a = m - delta;
            } else {
                b = m + delta;
            }

            f_est = std::max(f1, f2);

            if (std::fabs(f_est - f_prev) <= rel_tol * std::fabs(f_est)) {
                break;
            }
            f_prev = f_est;
        }

        // the max may sit on the boundary if f is monotonic
        return std::max(f_est, std::max(fa, fb));
    }

} // namespace shammath
