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
 * @file solve.hpp
 * @author Timothée David--Cléris (tim.shamrock@proton.me)
 * @brief
 *
 */

#include <cmath>
#include <functional>

namespace shammath {

    template<class T>
    float newtown_rhaphson(std::function<T(T)> &&f, std::function<T(T)> &&df, T epsilon_c, T x_0) {

        auto iterate_newtown = [](T f, T df, T xk) -> T {
            return xk - (f / df);
        };

        T xk      = x_0;
        T epsilon = 100000;

        while (epsilon > epsilon_c) {
            T xkp1 = iterate_newtown(f(xk), df(xk), xk);

            epsilon = std::fabs(xk - xkp1);

            xk = xkp1;
        }

        return xk;
    }

} // namespace shammath
