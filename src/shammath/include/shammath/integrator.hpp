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
 * @file integrator.hpp
 * @author David Fang (david.fang@ikmail.com)
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include <functional>

namespace shammath {

    template<class T, class Lambda>
    inline constexpr T integ_riemann_sum(T start, T end, T step, Lambda &&fct) {
        T acc = {};

        for (T x = start; x < end; x += step) {
            acc += fct(x) * step;
        }
        return acc;
    }

    template<class T, class Lambda>
    inline constexpr T integ_trapezoidal(T start, T end, T step, Lambda &&fct) {
        T acc   = {};
        T fprev = 0;
        T f     = 0;
        for (T x = start; x < end; x += step) {
            f = fct(x);
            acc += 0.5 * (f + fprev) * step;
            fprev = f;
        }
        return acc;
    }

} // namespace shammath
